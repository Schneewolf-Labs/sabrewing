/* moe_matmul.h — shared f32 GEMV/GEMM for the MoE engines.
 * Phase 1 of the MoE-runtime refactor (docs/moe-runtime-plan.md).
 *
 * y[S,O] = x[S,I] @ W[O,I]^T  (W row-major [out,in], HF Linear convention).
 * Batched over S rows — the general/prefill-friendly form (inkling was already
 * batched; laguna called the S=1 special case). One canonical implementation:
 * AVX-512 FMA (f32 accumulate) for speed, with an `exact` double-accumulate
 * fallback used by the token-exact oracle. For S=1 this is bit-identical to
 * laguna's previous single-row matmul (same per-output reduction order); the
 * batched path reduces each (o,s) independently, so S>1 is just the same dot
 * repeated — no cross-row reduction that could perturb numerics.
 *
 * This is the ONLY oracle-covered kernel (the tiny oracle runs bits=0 = f32).
 * The bf16/int8/int4 kernels have real per-engine contract differences (laguna
 * keeps activations f32; inkling rounds them to bf16) and are NOT unified here —
 * they need a dedicated kernel test harness first. */
#ifndef MOE_MATMUL_H
#define MOE_MATMUL_H
#include <stdint.h>
#include <stdlib.h>
#include "moe_math.h"          /* bf16_f32 */
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ---------- register-blocked batched core (S > 1) ----------
 *
 * The batched kernels used to reduce every (o,s) output with ONE dependent FMA
 * chain, so a prefill chunk or a decode batch ran at FMA *latency*, not throughput:
 * ~45-80 GFLOP/s on a 4-core AVX-512 box where the units sustain several times
 * that. This core computes a 4-output x up-to-4-row tile at once — 16 independent
 * accumulators, each weight chunk loaded once for 4 rows and each activation chunk
 * once for 4 outputs.
 *
 * Numerics are untouched: every output still owns exactly one accumulator that
 * sees the same 16-wide FMAs in the same ascending order, then the same
 * _mm512_reduce_add_ps, then the same scalar tail, then the same scale multiply.
 * Tiling only interleaves independent reductions, so S>1 stays BIT-IDENTICAL to
 * the S=1 kernels (tools/kernel_check.c asserts it for every format).
 *
 * Weights reach the core as f32 rows: f32 weights directly, the quantized formats
 * via a per-thread dequant of the tile's 4 rows (amortized over up to MOE_SCHUNK
 * activation rows). Activations are walked in MOE_SCHUNK-row chunks so a long
 * prefill's x stays cache-resident across all output tiles. */
#if defined(__AVX512F__)
#define MOE_TO     4     /* output rows per tile */
#define MOE_TS     4     /* activation rows per tile */
#define MOE_SCHUNK 32    /* activation rows per cache chunk (32 x 3072 f32 = 384 KB) */

/* finish one output: the S=1 epilogue, verbatim */
static inline float moe_tile_fin(__m512 acc, const float *x, const float *w, int i, int I, float sc) {
    float sm = _mm512_reduce_add_ps(acc);
    for (; i < I; i++) sm += x[i] * w[i];
    return sm * sc;
}

/* MOE_TO outputs x ns rows; ns is a compile-time constant after inlining */
static inline __attribute__((always_inline))
void moe_tile_4xn(float *y, int64_t ldy, const float *x, int64_t ldx, int I,
                  const float *w, int64_t ldw, const float *sc, const int ns) {
    __m512 acc[MOE_TO][MOE_TS];
    for (int a = 0; a < MOE_TO; a++) for (int b = 0; b < ns; b++) acc[a][b] = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= I; i += 16) {
        __m512 wv[MOE_TO];
        for (int a = 0; a < MOE_TO; a++) wv[a] = _mm512_loadu_ps(w + a * ldw + i);
        for (int b = 0; b < ns; b++) {
            __m512 xv = _mm512_loadu_ps(x + b * ldx + i);
            for (int a = 0; a < MOE_TO; a++) acc[a][b] = _mm512_fmadd_ps(xv, wv[a], acc[a][b]);
        }
    }
    for (int b = 0; b < ns; b++)
        for (int a = 0; a < MOE_TO; a++)
            y[b * ldy + a] = moe_tile_fin(acc[a][b], x + b * ldx, w + a * ldw, i, I, sc ? sc[a] : 1.f);
}

/* mo (<= MOE_TO) outputs x S rows. y/x/w/sc are already offset to the tile. */
static void moe_tile(float *y, int64_t ldy, const float *x, int64_t ldx, int S, int I,
                     const float *w, int64_t ldw, int mo, const float *sc) {
    int s = 0;
    if (mo == MOE_TO) {
        for (; s + 4 <= S; s += 4) moe_tile_4xn(y + s * ldy, ldy, x + s * ldx, ldx, I, w, ldw, sc, 4);
        switch (S - s) {
        case 3: moe_tile_4xn(y + s * ldy, ldy, x + s * ldx, ldx, I, w, ldw, sc, 3); break;
        case 2: moe_tile_4xn(y + s * ldy, ldy, x + s * ldx, ldx, I, w, ldw, sc, 2); break;
        case 1: moe_tile_4xn(y + s * ldy, ldy, x + s * ldx, ldx, I, w, ldw, sc, 1); break;
        }
        return;
    }
    for (; s < S; s++)                                  /* ragged last tile: one chain each */
        for (int a = 0; a < mo; a++) {
            const float *xs = x + s * ldx, *wa = w + a * ldw;
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 16 <= I; i += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i), _mm512_loadu_ps(wa + i), acc);
            y[s * ldy + a] = moe_tile_fin(acc, xs, wa, i, I, sc ? sc[a] : 1.f);
        }
}

/* Expand weight rows [o, o+n) to f32 into dst[n][I]. NULL = rows are already f32. */
typedef void (*moe_deq_fn)(float *dst, const void *W, const void *aux, int64_t o, int n, int I);

/* y[S,O] = x[S,I] @ W^T (* scale[o] if scale), any weight format via deq. */
static void moe_gemm_tiled(float *y, const float *x, int S, int I, int O,
                           const void *W, moe_deq_fn deq, const void *aux, const float *scale) {
    int nt = (O + MOE_TO - 1) / MOE_TO;
    #pragma omp parallel if(MOE_MM_PAR(S, O))
    {
        float *wf = deq ? (float*)malloc((size_t)MOE_TO * I * sizeof(float)) : NULL;
        for (int s0 = 0; s0 < S; s0 += MOE_SCHUNK) {
            int sn = S - s0 < MOE_SCHUNK ? S - s0 : MOE_SCHUNK;
            #pragma omp for schedule(static)
            for (int t = 0; t < nt; t++) {
                int o = t * MOE_TO, mo = O - o < MOE_TO ? O - o : MOE_TO;
                const float *w = (const float*)W + (int64_t)o * I;
                if (deq) { deq(wf, W, aux, o, mo, I); w = wf; }
                moe_tile(y + (int64_t)s0 * O + o, O, x + (int64_t)s0 * I, I, sn, I,
                         w, I, mo, scale ? scale + o : NULL);
            }
        }
        free(wf);
    }
}

/* bf16 -> f32 is exact (top 16 bits), so the tiled path matches the in-loop convert */
static void moe_deq_bf16(float *dst, const void *W, const void *aux, int64_t o, int n, int I) {
    (void)aux;
    const uint16_t *w = (const uint16_t*)W + o * I;
    for (int64_t i = 0; i < (int64_t)n * I; i++) dst[i] = bf16_f32(w[i]);
}
#endif

#if !defined(__AVX512F__)
/* ---------- portable fast path (no AVX-512) ----------
 * Every fast path above is AVX-512, and the Makefile's default ARCH is x86-64-v3, so a
 * plain `make` (or an AVX2-only Zen 2/3 box, or arm64) used to run every GEMM through the
 * scalar double-accumulate ORACLE loop: 4-8 GFLOP/s. Sixteen independent float lanes in
 * plain C vectorize on whatever the compiler targets (AVX2: two ymm, NEON: four q). Not
 * bit-identical to the AVX-512 build (different reduction tree) — nothing non-exact is
 * across builds — but S>1 and S=1 run the same function, so batched stays bit-identical
 * to single-row within a build. */
#define MOE_LANES 16
static inline float moe_dot_lanes(const float *x, const float *w, int I) {
    float acc[MOE_LANES] = {0};
    int i = 0;
    for (; i + MOE_LANES <= I; i += MOE_LANES)
        for (int l = 0; l < MOE_LANES; l++) acc[l] += x[i + l] * w[i + l];
    float s = 0;
    for (int l = 0; l < MOE_LANES; l++) s += acc[l];
    for (; i < I; i++) s += x[i] * w[i];
    return s;
}
/* y[S,O] = (x[S,I] . wrow(o)) * scale[o]: deq expands one weight row to f32 (NULL = W is
 * f32 already), once per output row for all S activation rows. */
typedef void (*moe_row_deq_fn)(float *dst, const void *W, const void *aux, int64_t o, int I);
static void moe_gemm_lanes(float *y, const float *x, int S, int I, int O, const void *W,
                           moe_row_deq_fn deq, const void *aux, const float *scale) {
    #pragma omp parallel if(MOE_MM_PAR(S, O))
    {
        float *wf = deq ? (float*)malloc((size_t)I * sizeof(float)) : NULL;
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            const float *w = (const float*)W + (int64_t)o * I;
            if (deq) { deq(wf, W, aux, o, I); w = wf; }
            float sc = scale ? scale[o] : 1.f;
            for (int s = 0; s < S; s++)
                y[(int64_t)s * O + o] = moe_dot_lanes(x + (int64_t)s * I, w, I) * sc;
        }
        free(wf);
    }
}
static void moe_rdeq_bf16(float *dst, const void *W, const void *aux, int64_t o, int I) {
    (void)aux;
    const uint16_t *w = (const uint16_t*)W + o * I;
    for (int i = 0; i < I; i++) dst[i] = bf16_f32(w[i]);
}
#endif

static void matmul_f32(float *y, const float *x, const float *W, int S, int I, int O, int exact) {
#if defined(__AVX512F__)
    if (!exact && S > 1) { moe_gemm_tiled(y, x, S, I, O, W, NULL, NULL, NULL); return; }
    if (!exact) {
        #pragma omp parallel for schedule(static) if(MOE_MM_PAR(S, O))
        for (int o = 0; o < O; o++) {
            const float *w = W + (int64_t)o * I;
            for (int s = 0; s < S; s++) {
                const float *xs = x + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                int i = 0;
                for (; i + 16 <= I; i += 16)
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i), _mm512_loadu_ps(w + i), acc);
                float sm = _mm512_reduce_add_ps(acc);
                for (; i < I; i++) sm += xs[i] * w[i];
                y[(int64_t)s * O + o] = sm;
            }
        }
        return;
    }
#endif
#if !defined(__AVX512F__)
    if (!exact) { moe_gemm_lanes(y, x, S, I, O, W, NULL, NULL, NULL); return; }
#endif
    /* exact double-accumulate reference (the oracle path) */
    #pragma omp parallel for schedule(static) if(MOE_MM_PAR(S, O))
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            double sm = 0; for (int i = 0; i < I; i++) sm += (double)xs[i] * w[i];
            y[(int64_t)s * O + o] = (float)sm;
        }
    }
}

/* bf16-weight GEMM: W is raw bf16 [O,I]. Two activation contracts:
 *   round_x=0 — activations kept f32, weight expanded bf16->f32 EXACT, f32 FMA
 *               (laguna's default; strictly more accurate).
 *   round_x=1 — activations ALSO rounded to bf16, hardware vdpbf16ps dot where
 *               available (inkling's default; matches an HF bf16-activations ref).
 * exact forces the double-accumulate scalar reference (round_x=0 contract). For
 * S=1 the round_x=0 SIMD path is bit-identical to laguna's old matmul_bf16; the
 * round_x=1 dpbf16 path is bit-identical to inkling's matmul_h fast path. */
static void matmul_bf16_k(float *y, const float *x, const uint16_t *W, int S, int I, int O, int round_x, int exact) {
#if defined(__AVX512BF16__) && defined(__AVX512F__)
    if (round_x && !exact && I % 32 == 0) {
        uint16_t *xh = (uint16_t*)malloc((size_t)S * I * sizeof(uint16_t));
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I; uint16_t *xd = xh + (int64_t)s * I;
            for (int i = 0; i < I; i += 32) {
                __m512 a = _mm512_loadu_ps(xs + i), b = _mm512_loadu_ps(xs + i + 16);
                _mm512_storeu_si512(xd + i, (__m512i)_mm512_cvtne2ps_pbh(b, a));
            }
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint16_t *w = W + (int64_t)o * I;
            for (int s = 0; s < S; s++) {
                const uint16_t *xs = xh + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                for (int i = 0; i < I; i += 32)
                    acc = _mm512_dpbf16_ps(acc, (__m512bh)_mm512_loadu_si512(xs + i),
                                                (__m512bh)_mm512_loadu_si512(w + i));
                y[(int64_t)s * O + o] = _mm512_reduce_add_ps(acc);
            }
        }
        free(xh);
        return;
    }
#endif
#if defined(__AVX512F__)
    /* tiled from 4 rows: below that the bf16->f32 row expansion costs about what it saves */
    if (!exact && S >= 4) { moe_gemm_tiled(y, x, S, I, O, W, moe_deq_bf16, NULL, NULL); return; }
    if (!exact) {   /* weight bf16->f32 exact, activations f32 */
        #pragma omp parallel for schedule(static) if(MOE_MM_PAR(S, O))
        for (int o = 0; o < O; o++) {
            const uint16_t *w = W + (int64_t)o * I;
            for (int s = 0; s < S; s++) {
                const float *xs = x + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                int i = 0;
                for (; i + 16 <= I; i += 16) {
                    __m512i we = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(w + i))), 16);
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i), _mm512_castsi512_ps(we), acc);
                }
                float sm = _mm512_reduce_add_ps(acc);
                for (; i < I; i++) sm += xs[i] * bf16_f32(w[i]);
                y[(int64_t)s * O + o] = sm;
            }
        }
        return;
    }
#else
    if (!exact) { moe_gemm_lanes(y, x, S, I, O, W, moe_rdeq_bf16, NULL, NULL); return; }
#endif
    #pragma omp parallel for schedule(static) if(MOE_MM_PAR(S, O))
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            double sm = 0; for (int i = 0; i < I; i++) sm += (double)xs[i] * bf16_f32(w[i]);
            y[(int64_t)s * O + o] = (float)sm;
        }
    }
}

#endif
