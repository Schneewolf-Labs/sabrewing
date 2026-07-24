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
#include "moe_math.h"          /* bf16_f32 */
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

static void matmul_f32(float *y, const float *x, const float *W, int S, int I, int O, int exact) {
#if defined(__AVX512F__)
    if (!exact) {
        #pragma omp parallel for schedule(static) if(O >= 512)
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
    /* exact double-accumulate reference (the oracle path) */
    #pragma omp parallel for schedule(static) if(O >= 512)
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
    if (!exact) {   /* weight bf16->f32 exact, activations f32 */
        #pragma omp parallel for schedule(static) if(O >= 512)
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
#endif
    #pragma omp parallel for schedule(static) if(O >= 512)
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
