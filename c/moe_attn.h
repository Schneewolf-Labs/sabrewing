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
#include "moe_math.h"   /* softmax_row */
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

enum { MOE_QK_F32 = 0, MOE_QK_DBL = 1, MOE_QK_SIMD = 2 };

static void sdpa_head(const float *q, const float *Kbase, const float *Vbase, int kv_stride,
                      int hd, int t0, int qpos, float scale, float tau,
                      const float *bias, int bias_len, int qk_mode, float *out, float *sc,
                      int ring) {
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
