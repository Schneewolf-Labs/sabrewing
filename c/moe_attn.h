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
 *   qk_accum_dbl — 1: <q,k> in double (laguna's oracle contract); 0: float (inkling).
 *
 * Kbase/Vbase point at this head's position-0 K/V row; consecutive cached
 * positions are kv_stride floats apart (laguna: n_kv*hd interleaved; inkling:
 * hd, head-major). sc is caller scratch of length >= (qpos - t0 + 1). */
#ifndef MOE_ATTN_H
#define MOE_ATTN_H
#include <stdint.h>
#include "moe_math.h"   /* softmax_row */

static void sdpa_head(const float *q, const float *Kbase, const float *Vbase, int kv_stride,
                      int hd, int t0, int qpos, float scale, float tau,
                      const float *bias, int bias_len, int qk_accum_dbl, float *out, float *sc) {
    int n = qpos - t0 + 1;
    for (int t = t0; t <= qpos; t++) {
        const float *k = Kbase + (int64_t)t * kv_stride;
        float dot;
        if (qk_accum_dbl) { double s = 0; for (int d = 0; d < hd; d++) s += (double)q[d] * k[d]; dot = (float)s; }
        else              { float  s = 0; for (int d = 0; d < hd; d++) s += q[d] * k[d];          dot = s;        }
        int dist = qpos - t;
        float b = (bias && dist < bias_len) ? bias[dist] : 0.f;
        sc[t - t0] = tau * (dot * scale + b);
    }
    softmax_row(sc, n);
    for (int d = 0; d < hd; d++) out[d] = 0.f;
    for (int t = t0; t <= qpos; t++) {
        const float *v = Vbase + (int64_t)t * kv_stride;
        float a = sc[t - t0];
        for (int d = 0; d < hd; d++) out[d] += a * v[d];
    }
}

#endif
