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

/* GQA-grouped attention: attend `nq` query heads that share one KV head in a SINGLE pass
 * over the K/V window.
 *
 * Laguna has 8 KV heads serving 48-72 query heads, so a per-head loop reads each KV head's
 * K and V 6-9 TIMES per layer — the same bytes, once per query head in the group. At 2k
 * context attention was 41-45% of a decode step, and this is where those bytes go.
 *
 * The online-softmax form is what makes one pass possible (no score buffer to keep per
 * head, running max/sum instead). For a single query row that form measured a wash — the
 * score row was already cache-resident, so there was nothing to save. With 6-9 rows sharing
 * the stream it is the whole point: K_t and V_t are loaded once and used nq times.
 *
 * Queries and outputs for a group are contiguous (head hh belongs to kv head hh/group), so
 * the caller passes the first head of the group and a stride of hd.
 *
 * Rescaling makes the arithmetic differ from the two-pass path, so this is generation-only;
 * exact callers keep sdpa_head and their oracles. */
#define SDPA_MAX_GROUP 16

static void sdpa_group(const float *q, int q_stride,
                       const float *Kbase, const float *Vbase, int kv_stride,
                       int hd, int t0, int qpos, float scale, float tau,
                       float *out, int out_stride, int nq, int ring) {
    float m[SDPA_MAX_GROUP], sum[SDPA_MAX_GROUP];
    int mask = ring - 1;
    for (int j = 0; j < nq; j++) {
        m[j] = -INFINITY;
        sum[j] = 0.f;
        float *o = out + (int64_t)j * out_stride;
        for (int d = 0; d < hd; d++) o[d] = 0.f;
    }

    for (int t = t0; t <= qpos; t++) {
        int64_t slot = ring ? (t & mask) : t;
        const float *k = Kbase + slot * kv_stride;
        const float *v = Vbase + slot * kv_stride;
        for (int j = 0; j < nq; j++) {
            const float *qj = q + (int64_t)j * q_stride;
            float dot;
#if defined(__AVX512F__)
            {
                __m512 acc = _mm512_setzero_ps();
                int d = 0;
                for (; d + 16 <= hd; d += 16)
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(qj + d), _mm512_loadu_ps(k + d), acc);
                float sd = _mm512_reduce_add_ps(acc);
                for (; d < hd; d++) sd += qj[d] * k[d];
                dot = sd;
            }
#else
            { float sd = 0; for (int d = 0; d < hd; d++) sd += qj[d] * k[d]; dot = sd; }
#endif
            float x = tau * (dot * scale);
            float *o = out + (int64_t)j * out_stride;
            if (x > m[j]) {          /* running max rises O(log n) times, not n */
                float rescale = (m[j] == -INFINITY) ? 0.f : expf(m[j] - x);
                sum[j] = sum[j] * rescale + 1.f;
#if defined(__AVX512F__)
                __m512 vr = _mm512_set1_ps(rescale);
                int d = 0;
                for (; d + 16 <= hd; d += 16)
                    _mm512_storeu_ps(o + d, _mm512_fmadd_ps(_mm512_loadu_ps(o + d), vr,
                                                            _mm512_loadu_ps(v + d)));
                for (; d < hd; d++) o[d] = o[d] * rescale + v[d];
#else
                for (int d = 0; d < hd; d++) o[d] = o[d] * rescale + v[d];
#endif
                m[j] = x;
            } else {
                float w = expf(x - m[j]);
                sum[j] += w;
#if defined(__AVX512F__)
                __m512 vw = _mm512_set1_ps(w);
                int d = 0;
                for (; d + 16 <= hd; d += 16)
                    _mm512_storeu_ps(o + d, _mm512_fmadd_ps(vw, _mm512_loadu_ps(v + d),
                                                            _mm512_loadu_ps(o + d)));
                for (; d < hd; d++) o[d] += w * v[d];
#else
                for (int d = 0; d < hd; d++) o[d] += w * v[d];
#endif
            }
        }
    }

    for (int j = 0; j < nq; j++) {
        float inv = sum[j] > 0.f ? 1.f / sum[j] : 0.f;
        float *o = out + (int64_t)j * out_stride;
        for (int d = 0; d < hd; d++) o[d] *= inv;
    }
}

/* SDPA over an int8 K/V cache.
 *
 * Scales are per BLOCK of KV_Q8_BLOCK elements, not per row. One scale per 128-element row
 * measured 706/1024 teacher-forced prediction agreement on the real model — a third of
 * next-token predictions changed, which is not a cache, it is a different model. The reason
 * is asymmetric: K passes through qk-RMSNorm before it is cached and is well-conditioned,
 * while V is stored raw from v_proj, and LLM activations carry outlier channels. One outlier
 * sets the scale for all 128 values and collapses the rest into a couple of levels.
 * llama.cpp's q8_0 KV uses 32-element blocks for exactly this reason; the cost is 4 bytes
 * per 32 values (1.125 B/value vs 1.03), still ~3.5x smaller than f32.
 *
 * Scales fold into the arithmetic for free: <q,K_t> = sum_b s_b * <q_b, K_b^int8>, and the
 * value accumulation folds each block's s_b into the softmax weight. Lossy, so opt-in and
 * gated on measured prediction agreement, not on perplexity (which moved the wrong way and
 * hid the problem entirely). */
#define KV_Q8_BLOCK 32

/* quantize one hd-long row to int8 with a scale per KV_Q8_BLOCK elements */
static void kv_quant_row(const float *x, int hd, int8_t *q, float *scales) {
    for (int b0 = 0; b0 < hd; b0 += KV_Q8_BLOCK) {
        int n = hd - b0 < KV_Q8_BLOCK ? hd - b0 : KV_Q8_BLOCK;
        float mx = 0;
        for (int d = 0; d < n; d++) { float a = x[b0 + d] < 0 ? -x[b0 + d] : x[b0 + d]; if (a > mx) mx = a; }
        float s = mx / 127.f; if (s < 1e-12f) s = 1e-12f;
        scales[b0 / KV_Q8_BLOCK] = s;
        float inv = 1.f / s;
        for (int d = 0; d < n; d++) {
            int v = (int)lrintf(x[b0 + d] * inv);
            q[b0 + d] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
        }
    }
}

/* scales per quantized row */
static inline int kv_q8_blocks(int hd) { return (hd + KV_Q8_BLOCK - 1) / KV_Q8_BLOCK; }

/* Kq/Vq: int8 rows, stride kv_stride elements. Ks/Vs: scales, stride sc_stride floats
 * (== kv_q8_blocks(hd) per row times the number of kv heads). */
static void sdpa_head_q8(const float *q,
                         const int8_t *Kq, const float *Ks,
                         const int8_t *Vq, const float *Vs,
                         int kv_stride, int sc_stride,
                         int hd, int t0, int qpos, float scale, float tau,
                         float *out, float *sc, int ring) {
    int n = qpos - t0 + 1, mask = ring - 1, nb = kv_q8_blocks(hd);
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = ring ? (t & mask) : t;
        const int8_t *k = Kq + slot * kv_stride;
        const float *ks = Ks + slot * sc_stride;
        float dot = 0.f;
        for (int b = 0; b < nb; b++) {
            int b0 = b * KV_Q8_BLOCK, len = hd - b0 < KV_Q8_BLOCK ? hd - b0 : KV_Q8_BLOCK;
            float part = 0.f;
#if defined(__AVX512F__)
            int d = 0;
            __m512 acc = _mm512_setzero_ps();
            for (; d + 16 <= len; d += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + b0 + d),
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(k + b0 + d)))), acc);
            part = _mm512_reduce_add_ps(acc);
            for (; d < len; d++) part += q[b0 + d] * (float)k[b0 + d];
#else
            for (int d = 0; d < len; d++) part += q[b0 + d] * (float)k[b0 + d];
#endif
            dot += part * ks[b];
        }
        sc[t - t0] = tau * (dot * scale);
    }
    softmax_row(sc, n);
    for (int d = 0; d < hd; d++) out[d] = 0.f;
    for (int t = t0; t <= qpos; t++) {
        int64_t slot = ring ? (t & mask) : t;
        const int8_t *v = Vq + slot * kv_stride;
        const float *vs = Vs + slot * sc_stride;
        float a = sc[t - t0];
        for (int b = 0; b < nb; b++) {
            int b0 = b * KV_Q8_BLOCK, len = hd - b0 < KV_Q8_BLOCK ? hd - b0 : KV_Q8_BLOCK;
            float aw = a * vs[b];               /* block scale folds into the weight */
#if defined(__AVX512F__)
            int d = 0;
            __m512 va = _mm512_set1_ps(aw);
            for (; d + 16 <= len; d += 16)
                _mm512_storeu_ps(out + b0 + d, _mm512_fmadd_ps(va,
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(v + b0 + d)))),
                        _mm512_loadu_ps(out + b0 + d)));
            for (; d < len; d++) out[b0 + d] += aw * (float)v[b0 + d];
#else
            for (int d = 0; d < len; d++) out[b0 + d] += aw * (float)v[b0 + d];
#endif
        }
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
