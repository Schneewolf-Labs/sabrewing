/* moe_linattn.h — gated delta net (linear attention), the Qwen3.5-MoE token mixer.
 *
 * 30 of Qwen3.5-35B-A3B's 40 layers use this instead of softmax attention, and that is the
 * interesting part of the architecture for a serving engine: the per-sequence state is
 * O(1) in context length — one [k_dim x v_dim] matrix per head — rather than growing with
 * every token. A 32k-context conversation keeps ~2 MB per linear layer regardless of how
 * long it gets, where a full-attention layer at that context needs ~134 MB. That turns the
 * KV-capacity problem (see docs/kv-cache-design.md) from the dominant constraint into a
 * rounding error on 3/4 of the layers.
 *
 * The recurrence, per head, per token (matching transformers'
 * torch_recurrent_gated_delta_rule, which is the reference this is validated against):
 *
 *     q, k  <- l2norm(q), l2norm(k)          (eps 1e-6)
 *     q     <- q / sqrt(k_dim)
 *     S     <- S * exp(g)                    decay, g < 0
 *     mem   <- k^T S                         [v_dim]
 *     delta <- (v - mem) * beta
 *     S     <- S + k (x) delta               rank-1 update
 *     out   <- q^T S                         [v_dim]
 *
 * It is cheap: ~3 * k_dim * v_dim MACs per head per token, so all 30 layers cost ~47 MMAC
 * per token against ~3 GMAC for the MoE. The naive sequential form is therefore fine — no
 * chunked/WY-transform kernel is needed to be fast, only to be parallel across a prefill.
 *
 * Ordering matters and is not obvious: the decay multiplies the state BEFORE the memory
 * read, and the state update uses the post-decay state. Getting either wrong still produces
 * plausible-looking activations, which is why this header ships with a fixture test
 * (tools/linattn_check.c against tools/gdn_ref.py) rather than trusting inspection. */
#ifndef MOE_LINATTN_H
#define MOE_LINATTN_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Qwen3.5-35B-A3B uses k_dim = v_dim = 128; the caps only size step/scan-local scratch,
 * and anything larger falls back to malloc rather than misbehaving. */
#define LINATTN_MAX_KDIM 256
#define LINATTN_MAX_VDIM 256

/* Which recurrence kernel linattn_scan uses. NAIVE follows the reference literally (3 sweeps
 * over the state); FUSED is the 2-sweep algebraic collapse and is the default. Same pattern
 * as the MOE_QK_* modes in moe_attn.h: keep the literal form so the fast one has something
 * to be wrong against. */
enum { LINATTN_FUSED = 0, LINATTN_NAIVE = 1 };

/* in-place L2 normalize, matching transformers' l2norm(x, eps=1e-6):
 * x * rsqrt(sum(x^2) + eps) — note eps is added to the SUM, not to the mean. */
static void linattn_l2norm(float *x, int n, float eps) {
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    float inv = 1.f / sqrtf(s + eps);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* softplus(x) = log1p(exp(x)), guarded for large x where exp overflows */
static inline float linattn_softplus(float x) {
    return x > 20.f ? x : log1pf(expf(x));
}

/* One gated-delta-rule step for one head.
 *   state: [k_dim][v_dim], updated in place
 *   q, k:  [k_dim] (already l2-normalized and q pre-scaled by the caller)
 *   v:     [v_dim]
 *   out:   [v_dim]
 * decay = exp(g), beta as computed from the projections. */
static void linattn_step(float *state, const float *q, const float *k, const float *v,
                         float decay, float beta, float *out, int k_dim, int v_dim) {
    /* S *= decay, and read mem = k^T S in the same sweep (S is walked once) */
    for (int i = 0; i < v_dim; i++) out[i] = 0.f;
    for (int a = 0; a < k_dim; a++) {
        float *row = state + (size_t)a * v_dim;
        float ka = k[a];
        for (int i = 0; i < v_dim; i++) {
            row[i] *= decay;
            out[i] += row[i] * ka;          /* out temporarily holds mem */
        }
    }
    /* delta = (v - mem) * beta; S += k (x) delta */
    float delta[512];
    int use_stack = v_dim <= 512;
    float *d = use_stack ? delta : (float *)malloc((size_t)v_dim * sizeof(float));
    for (int i = 0; i < v_dim; i++) d[i] = (v[i] - out[i]) * beta;
    for (int a = 0; a < k_dim; a++) {
        float *row = state + (size_t)a * v_dim;
        float ka = k[a];
        for (int i = 0; i < v_dim; i++) row[i] += ka * d[i];
    }
    /* out = q^T S (recomputed from the updated state, not accumulated above) */
    for (int i = 0; i < v_dim; i++) out[i] = 0.f;
    for (int a = 0; a < k_dim; a++) {
        const float *row = state + (size_t)a * v_dim;
        float qa = q[a];
        for (int i = 0; i < v_dim; i++) out[i] += row[i] * qa;
    }
    if (!use_stack) free(d);
}

/* Depthwise causal conv1d over a rolling window, then SiLU — the pre-recurrence mixer.
 *   win:  [conv_dim][kernel] ring of past inputs, oldest first; updated in place
 *   x:    [conv_dim] this token's pre-conv values
 *   out:  [conv_dim] silu(conv(x))
 * Each channel is independent (groups == conv_dim), so this is a per-channel dot with the
 * channel's kernel over the last `kernel` timesteps. */
static void linattn_conv_step(float *win, const float *x, const float *w, const float *bias,
                              float *out, int conv_dim, int kernel) {
    for (int c = 0; c < conv_dim; c++) {
        float *wc = win + (size_t)c * kernel;
        for (int t = 0; t < kernel - 1; t++) wc[t] = wc[t + 1];   /* shift left */
        wc[kernel - 1] = x[c];
        const float *kw = w + (size_t)c * kernel;
        float s = bias ? bias[c] : 0.f;
        for (int t = 0; t < kernel; t++) s += wc[t] * kw[t];
        out[c] = s / (1.f + expf(-s));                            /* SiLU */
    }
}

/* Fused gated-delta-rule step: same math, two sweeps over the state instead of three.
 *
 * The naive form above walks S three times (decay+read, update, output read) and writes it
 * twice. Two identities collapse that:
 *
 *   1. decay is a *scalar*, so it factors out of the memory read entirely — S_dec = decay*S
 *      means k^T S_dec == decay * (k^T S). No decay pass over the state is needed at all;
 *      fold the decay into the same read-modify-write sweep that applies the rank-1 update.
 *   2. the output reads the *updated* state, but that expands:
 *        q^T (S_dec + k (x) delta) == decay * (q^T S) + (q.k) * delta
 *      so the output can be accumulated in the same read-only sweep as the memory read, from
 *      the OLD state, and corrected afterwards with one dot product.
 *
 * Net: 1 read-only sweep + 1 read-modify-write sweep (2R + 1W) against the naive 3R + 2W —
 * ~40% less state traffic, which is what this kernel is actually bound by. At the 35B's
 * 128x128 heads, arithmetic intensity is ~3 flops per float touched; the FLOPs are free and
 * the bytes are everything.
 *
 * Not bit-identical to the naive form (different summation order for `out`), which is why
 * linattn_check.c validates both against the transformers reference separately rather than
 * against each other. */
static void linattn_step_fused(float *state, const float *q, const float *k, const float *v,
                              float decay, float beta, float *out, int k_dim, int v_dim) {
    float mem_s[LINATTN_MAX_VDIM];
    float *mem = mem_s;
    float *heap = NULL;
    if (v_dim > LINATTN_MAX_VDIM) { heap = (float *)malloc((size_t)v_dim * sizeof(float)); mem = heap; }

    /* sweep 1, read-only: mem = k^T S, out = q^T S (both against the pre-decay state) */
    for (int i = 0; i < v_dim; i++) { mem[i] = 0.f; out[i] = 0.f; }
    float qk = 0.f;
    for (int a = 0; a < k_dim; a++) {
        const float *row = state + (size_t)a * v_dim;
        float ka = k[a], qa = q[a];
        qk += qa * ka;
        for (int i = 0; i < v_dim; i++) {
            float s = row[i];
            mem[i] += s * ka;
            out[i] += s * qa;
        }
    }
    /* delta from the decayed memory read; out corrected for the rank-1 term */
    float dbuf_s[LINATTN_MAX_VDIM];
    float *d = dbuf_s;
    float *dheap = NULL;
    if (v_dim > LINATTN_MAX_VDIM) { dheap = (float *)malloc((size_t)v_dim * sizeof(float)); d = dheap; }
    for (int i = 0; i < v_dim; i++) {
        d[i] = (v[i] - decay * mem[i]) * beta;
        out[i] = decay * out[i] + qk * d[i];
    }
    /* sweep 2, read-modify-write: S = decay*S + k (x) delta */
    for (int a = 0; a < k_dim; a++) {
        float *row = state + (size_t)a * v_dim;
        float ka = k[a];
        for (int i = 0; i < v_dim; i++) row[i] = row[i] * decay + ka * d[i];
    }
    free(heap); free(dheap);
}

/* ---- span forms: what a chunked prefill actually calls ----
 *
 * Our prefill runs in spans (forward_span, bounded by kv_span()), so both stateful pieces —
 * the recurrent state and the conv window — have to carry across span boundaries and give
 * the same answer as feeding tokens one at a time. That equality is not free: it is the same
 * class of bug as the sliding-KV ring off-by-N, where a pass that writes positions before
 * reading them broke continuation silently. linattn_check.c tests it directly by running the
 * fixture as 1-token steps and as uneven spans and requiring a match. */
typedef struct {
    float *state;   /* [n_v_heads][k_dim][v_dim] — the O(1)-in-context part */
    float *conv;    /* [conv_dim][kernel] rolling window, oldest first */
} LinAttnState;

/* Bytes of per-sequence state. The whole reason this architecture matters for serving:
 * independent of context length. */
static size_t linattn_state_bytes(int n_v_heads, int k_dim, int v_dim, int conv_dim, int kernel) {
    return ((size_t)n_v_heads * k_dim * v_dim + (size_t)conv_dim * kernel) * sizeof(float);
}

/* Depthwise causal conv + SiLU over S tokens, carrying `win` across calls.
 *   in:  [S][conv_dim] row-major, out: [S][conv_dim] (may alias in) */
static void linattn_conv_span(float *win, const float *in, const float *w, const float *bias,
                              float *out, int S, int conv_dim, int kernel) {
    for (int c = 0; c < conv_dim; c++) {
        float *wc = win + (size_t)c * kernel;
        const float *kw = w + (size_t)c * kernel;
        float b = bias ? bias[c] : 0.f;
        for (int t = 0; t < S; t++) {
            for (int j = 0; j < kernel - 1; j++) wc[j] = wc[j + 1];
            wc[kernel - 1] = in[(size_t)t * conv_dim + c];
            float s = b;
            for (int j = 0; j < kernel; j++) s += wc[j] * kw[j];
            out[(size_t)t * conv_dim + c] = s / (1.f + expf(-s));
        }
    }
}

/* Gated delta rule over S tokens, all heads, carrying st->state across calls.
 *   q, k: [S][n_k_heads * k_dim]  (NOT pre-expanded to value heads — this reads head h/rep)
 *   v:    [S][n_v_heads * v_dim]
 *   g, beta: [S][n_v_heads]
 *   out:  [S][n_v_heads * v_dim]
 *
 * Head-outer / token-inner: one head's state (k_dim*v_dim floats, 64 KB at the 35B's
 * 128x128) stays resident for the whole span instead of being re-streamed per token, and the
 * heads are independent so this is the axis to thread on. The recurrence is sequential in t
 * and cannot be batched into a GEMM without the chunked/WY form — which is not worth writing
 * yet, because all 30 linear layers together are ~1.5% of the model's per-token MACs. */
static void linattn_scan_mode(LinAttnState *st, const float *q, const float *k, const float *v,
                              const float *g, const float *beta, float *out,
                              int S, int n_k_heads, int n_v_heads, int k_dim, int v_dim,
                              int mode) {
    int rep = n_v_heads / n_k_heads;
    int qk_stride = n_k_heads * k_dim, v_stride = n_v_heads * v_dim;
    float scale = 1.f / sqrtf((float)k_dim);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < n_v_heads; h++) {
        float *S_h = st->state + (size_t)h * k_dim * v_dim;
        int kh = h / rep;                                  /* value heads share a key head */
        float qn[LINATTN_MAX_KDIM], kn[LINATTN_MAX_KDIM];
        for (int t = 0; t < S; t++) {
            const float *qs = q + (size_t)t * qk_stride + (size_t)kh * k_dim;
            const float *ks = k + (size_t)t * qk_stride + (size_t)kh * k_dim;
            memcpy(qn, qs, (size_t)k_dim * sizeof(float));
            memcpy(kn, ks, (size_t)k_dim * sizeof(float));
            linattn_l2norm(qn, k_dim, 1e-6f);
            linattn_l2norm(kn, k_dim, 1e-6f);
            for (int d = 0; d < k_dim; d++) qn[d] *= scale;
            const float *vt = v + (size_t)t * v_stride + (size_t)h * v_dim;
            float *ot = out + (size_t)t * v_stride + (size_t)h * v_dim;
            float decay = expf(g[(size_t)t * n_v_heads + h]);
            float bt = beta[(size_t)t * n_v_heads + h];
            if (mode == LINATTN_NAIVE)
                linattn_step(S_h, qn, kn, vt, decay, bt, ot, k_dim, v_dim);
            else
                linattn_step_fused(S_h, qn, kn, vt, decay, bt, ot, k_dim, v_dim);
        }
    }
}

static void linattn_scan(LinAttnState *st, const float *q, const float *k, const float *v,
                         const float *g, const float *beta, float *out,
                         int S, int n_k_heads, int n_v_heads, int k_dim, int v_dim) {
    linattn_scan_mode(st, q, k, v, g, beta, out, S, n_k_heads, n_v_heads, k_dim, v_dim,
                      LINATTN_FUSED);
}

/* RMSNorm over v_dim then multiply by silu(gate) — transformers' RMSNormGated.
 * Norm BEFORE gate, and the variance is a mean with eps added after. */
static void linattn_gated_norm(float *x, const float *gate, const float *weight,
                               int v_dim, float eps) {
    float var = 0;
    for (int i = 0; i < v_dim; i++) var += x[i] * x[i];
    var /= (float)v_dim;
    float inv = 1.f / sqrtf(var + eps);
    for (int i = 0; i < v_dim; i++) {
        float g = gate[i];
        x[i] = (x[i] * inv) * weight[i] * (g / (1.f + expf(-g)));
    }
}

#endif
