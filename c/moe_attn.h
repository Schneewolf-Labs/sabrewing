/* moe_attn.h — the shared scaled-dot-product attention core.
 * Phase 1 finale of the MoE-runtime refactor (docs/moe-runtime-plan.md).
 *
 * Attention is where the engines diverge most (laguna: RoPE + per-head softplus
 * output gate; inkling: NO RoPE / learned relative-position bias + short-convs +
 * log-length scaling), so — as with the kernels and the MoE block — only the
 * genuinely identical CORE is shared, and the contract differences are parameters,
 * not forks in the caller. The divergent SETUP (projection, qk-norm, RoPE vs
 * rel-bias, sconv, the output gate) stays per-engine; each engine calls sdpa_head
 * for the actual attend-over-cache once its Q and cache are ready.
 *
 * Per (query position qpos, head): over the window [t0..qpos],
 *     score[t] = tau * ( scale * <q, K_t> + (bias && dist<bias_len ? bias[dist] : 0) )
 *   where dist = qpos - t; softmax(score); out[d] = sum_t score[t] * V_t[d].
 *
 * Contract knobs (each engine's own, bit-identical to its hand loop):
 *   scale        — 1/sqrt(hd) (laguna) or 1/hd (inkling, deliberate w/ qk-norm).
 *   tau          — 1 (laguna) or the log-length scaling factor (inkling global).
 *   bias/bias_len— NULL (laguna) or the per-distance relative-bias bank (inkling).
 *   qk_mode      — how <q,k> accumulates, which is a per-engine numeric contract:
 *                  MOE_QK_F32  scalar float  (inkling's contract — its oracle is
 *                              token-exact against this exact reduction order)
 *                  MOE_QK_DBL  scalar double (laguna's oracle contract)
 *                  MOE_QK_SIMD AVX-512 FMA + vectorized value accumulation, for
 *                              GENERATION. Profiling batched decode showed attention
 *                              at 35% of a 2k-context step, all of it scalar double
 *                              math; this is the same exact-vs-fast split the matmuls
 *                              already use (laguna passes DBL under g_exact so the
 *                              oracle keeps validating the exact path).
 *
 * Kbase/Vbase point at this head's position-0 K/V row; consecutive cached
 * positions are kv_stride floats apart (laguna: n_kv*hd interleaved; inkling:
 * hd, head-major). sc is caller scratch of length >= (qpos - t0 + 1).
 *
 *   ring — 0: the cache is indexed by absolute position (the whole context is
 *          stored). Otherwise a POWER-OF-TWO ring size: position t lives at
 *          t & (ring-1). A sliding-window layer only ever attends to the last
 *          `window` positions, so it only needs `window` slots — storing the full
 *          context for those layers is pure waste (laguna: 36 of 48 layers slide
 *          over 512, so full-length f32 KV costs 3.4x what it needs). Wrapping is
 *          invisible to the math: the same positions are read in the same order. */
#ifndef MOE_ATTN_H
#define MOE_ATTN_H
#include <stdint.h>
#include <math.h>       /* lrintf (int8 KV quantization) */
#include "moe_math.h"   /* softmax_row */
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

enum { MOE_QK_F32 = 0, MOE_QK_DBL = 1, MOE_QK_SIMD = 2, MOE_QK_ONLINE = 3 };

/* Online-softmax (flash-attention-style) SDPA for generation: ONE pass over the K/V
 * window instead of two, and no score buffer at all. Per position it keeps a running
 * max m and sum s, rescaling the accumulator when a new max appears:
 *     m' = max(m, x);  acc *= e^(m-m');  acc += e^(x-m')*V_t;  s = s*e^(m-m') + e^(x-m')
 * which is algebraically the same softmax the two-pass version computes, but reads K
 * and V together so the window is streamed once. That halves the memory traffic of the
 * phase our profile put at 41% of a 2k-context decode step, and removes the per-(row,
 * head) `sc` scratch.
 *
 * Rescaling makes the rounding differ from the two-pass form, so this is a separate
 * mode: the exact contracts (MOE_QK_F32 / MOE_QK_DBL) keep their reduction order and
 * their oracles, and kernel_check measures this path against MOE_QK_DBL. Callers with
 * a per-distance bias (inkling) can use it too — bias is folded into the score before
 * the max update. */
static void sdpa_head_online(const float *q, const float *Kbase, const float *Vbase, int kv_stride,
                             int hd, int t0, int qpos, float scale, float tau,
                             const float *bias, int bias_len, float *out, int ring) {
    int mask = ring - 1;
    float m = -INFINITY, s = 0.f;
    for (int d = 0; d < hd; d++) out[d] = 0.f;
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = ring ? (t & mask) : t;
        const float *k = Kbase + slot * kv_stride;
        const float *v = Vbase + slot * kv_stride;
        float dot;
#if defined(__AVX512F__)
        {
            __m512 acc = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= hd; d += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), _mm512_loadu_ps(k + d), acc);
            float sd = _mm512_reduce_add_ps(acc);
            for (; d < hd; d++) sd += q[d] * k[d];
            dot = sd;
        }
#else
        { float sd = 0; for (int d = 0; d < hd; d++) sd += q[d] * k[d]; dot = sd; }
#endif
        int dist = qpos - t;
        float b = (bias && dist < bias_len) ? bias[dist] : 0.f;
        float x = tau * (dot * scale + b);
        /* The running max only rises, so it stops changing after the first few
         * positions (O(log n) updates for unordered scores). Rescaling the whole
         * accumulator on EVERY position doubles the value-accumulation work — the
         * common branch must be a plain FMA, with the rescale paid only when the max
         * actually moves. Same arithmetic either way. */
        if (x > m) {
            float rescale = (m == -INFINITY) ? 0.f : expf(m - x);
            s = s * rescale + 1.f;                 /* w = exp(x-x) = 1 */
#if defined(__AVX512F__)
            {
                __m512 vr = _mm512_set1_ps(rescale);
                int d = 0;
                for (; d + 16 <= hd; d += 16)
                    _mm512_storeu_ps(out + d, _mm512_fmadd_ps(_mm512_loadu_ps(out + d), vr,
                                                              _mm512_loadu_ps(v + d)));
                for (; d < hd; d++) out[d] = out[d] * rescale + v[d];
            }
#else
            for (int d = 0; d < hd; d++) out[d] = out[d] * rescale + v[d];
#endif
            m = x;
        } else {
            float w = expf(x - m);
            s += w;
#if defined(__AVX512F__)
            {
                __m512 vw = _mm512_set1_ps(w);
                int d = 0;
                for (; d + 16 <= hd; d += 16)
                    _mm512_storeu_ps(out + d, _mm512_fmadd_ps(vw, _mm512_loadu_ps(v + d),
                                                              _mm512_loadu_ps(out + d)));
                for (; d < hd; d++) out[d] += w * v[d];
            }
#else
            for (int d = 0; d < hd; d++) out[d] += w * v[d];
#endif
        }
    }
    float inv = s > 0.f ? 1.f / s : 0.f;
    for (int d = 0; d < hd; d++) out[d] *= inv;
}

/* SDPA over an int8 K/V cache (per (position, kv-head) row: hd int8 values + one f32
 * scale). Attention is 41-45% of a 2k-context decode step and it is bytes-bound, so
 * storing the cache at 1.03 bytes/value instead of 4 is the lever that actually removes
 * work — unlike restructuring passes over data that was already cache-resident.
 *
 * The scales fold into the arithmetic for free: <q, K_t> = s_K * <q, K_t^int8>, and the
 * value accumulation folds s_V into the softmax weight. Both stay f32 FMA over
 * int8->f32 converted lanes. This is a LOSSY storage choice (like the int4 experts), so
 * it is opt-in and gated on perplexity, not on speed alone. Laguna's contract only
 * (no per-distance bias). */
static void sdpa_head_q8(const float *q,
                         const int8_t *Kq, const float *Ks,
                         const int8_t *Vq, const float *Vs,
                         int kv_stride, int sc_stride,
                         int hd, int t0, int qpos, float scale, float tau,
                         float *out, float *sc, int ring) {
    int n = qpos - t0 + 1, mask = ring - 1;
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = ring ? (t & mask) : t;
        const int8_t *k = Kq + slot * kv_stride;
        float dot;
#if defined(__AVX512F__)
        {
            __m512 acc = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= hd; d += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + d),
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(k + d)))), acc);
            float s = _mm512_reduce_add_ps(acc);
            for (; d < hd; d++) s += q[d] * (float)k[d];
            dot = s;
        }
#else
        { float s = 0; for (int d = 0; d < hd; d++) s += q[d] * (float)k[d]; dot = s; }
#endif
        sc[t - t0] = tau * (dot * Ks[slot * sc_stride] * scale);
    }
    softmax_row(sc, n);
    for (int d = 0; d < hd; d++) out[d] = 0.f;
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = ring ? (t & mask) : t;
        const int8_t *v = Vq + slot * kv_stride;
        float a = sc[t - t0] * Vs[slot * sc_stride];      /* value scale folds into the weight */
#if defined(__AVX512F__)
        {
            __m512 va = _mm512_set1_ps(a);
            int d = 0;
            for (; d + 16 <= hd; d += 16)
                _mm512_storeu_ps(out + d, _mm512_fmadd_ps(va,
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(v + d)))),
                        _mm512_loadu_ps(out + d)));
            for (; d < hd; d++) out[d] += a * (float)v[d];
        }
#else
        for (int d = 0; d < hd; d++) out[d] += a * (float)v[d];
#endif
    }
}

/* quantize one hd-long row to int8 + scale (symmetric per row, like the residents) */
static void kv_quant_row(const float *x, int hd, int8_t *q, float *scale) {
    float mx = 0;
    for (int d = 0; d < hd; d++) { float a = x[d] < 0 ? -x[d] : x[d]; if (a > mx) mx = a; }
    float s = mx / 127.f; if (s < 1e-12f) s = 1e-12f;
    *scale = s;
    float inv = 1.f / s;
    for (int d = 0; d < hd; d++) {
        int v = (int)lrintf(x[d] * inv);
        q[d] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
    }
}

static void sdpa_head(const float *q, const float *Kbase, const float *Vbase, int kv_stride,
                      int hd, int t0, int qpos, float scale, float tau,
                      const float *bias, int bias_len, int qk_mode, float *out, float *sc,
                      int ring) {
    if (qk_mode == MOE_QK_ONLINE) {   /* one pass, no score buffer (sc unused) */
        sdpa_head_online(q, Kbase, Vbase, kv_stride, hd, t0, qpos, scale, tau, bias, bias_len, out, ring);
        return;
    }
    int n = qpos - t0 + 1;
    int mask = ring - 1;                     /* ring is a power of two, or 0 (unused) */
    int fast = (qk_mode == MOE_QK_SIMD);
    for (int t = t0; t <= qpos; t++) {
        const float *k = Kbase + (int64_t)(ring ? (t & mask) : t) * kv_stride;
        float dot;
#if defined(__AVX512F__)
        if (fast) {
            __m512 acc = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= hd; d += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), _mm512_loadu_ps(k + d), acc);
            float s = _mm512_reduce_add_ps(acc);
            for (; d < hd; d++) s += q[d] * k[d];
            dot = s;
        } else
#endif
        if (qk_mode == MOE_QK_DBL) { double s = 0; for (int d = 0; d < hd; d++) s += (double)q[d] * k[d]; dot = (float)s; }
        else                       { float  s = 0; for (int d = 0; d < hd; d++) s += q[d] * k[d];          dot = s;        }
        int dist = qpos - t;
        float b = (bias && dist < bias_len) ? bias[dist] : 0.f;
        sc[t - t0] = tau * (dot * scale + b);
    }
    softmax_row(sc, n);
    for (int d = 0; d < hd; d++) out[d] = 0.f;
    for (int t = t0; t <= qpos; t++) {
        const float *v = Vbase + (int64_t)(ring ? (t & mask) : t) * kv_stride;
        float a = sc[t - t0];
        /* the value accumulation is vectorized over d, so each out[d] still sums over
         * t in the same order — but only on the fast path, so the exact-mode callers
         * keep the byte-for-byte scalar loop their oracles were validated against. */
#if defined(__AVX512F__)
        if (fast) {
            __m512 va = _mm512_set1_ps(a);
            int d = 0;
            for (; d + 16 <= hd; d += 16)
                _mm512_storeu_ps(out + d, _mm512_fmadd_ps(va, _mm512_loadu_ps(v + d), _mm512_loadu_ps(out + d)));
            for (; d < hd; d++) out[d] += a * v[d];
            continue;
        }
#endif
        for (int d = 0; d < hd; d++) out[d] += a * v[d];
    }
}

#endif
