/* deepseek.c — DeepSeek-V4-Flash engine (284B total / 13B active, 1M context).
 *
 * Stage A: a readable f32 CPU forward pass, validated token-exact against
 * transformers' `deepseek_v4` reference. Quantized containers and a CUDA tier come
 * after the oracle is green — the same order laguna and qwen35 landed in.
 *
 * The architecture, and why almost nothing here is reusable from the other engines:
 *
 *   residual   x is NOT a vector. Every token carries `hc_mult` (4) parallel residual
 *              streams, mixed at each of the two sublayer sites by Manifold-Constrained
 *              Hyper-Connections (mHC): a per-token learned (pre, post, comb) triple where
 *              `comb` is projected onto the doubly-stochastic manifold by Sinkhorn-Knopp
 *              before it mixes the streams. `pre` collapses 4 streams -> 1 for the sublayer,
 *              `post` places the sublayer output back across the 4.
 *   attention  shared-KV MQA (ONE kv head broadcast to 64 query heads, and K IS V — the same
 *              tensor is read as both), head_dim 512, partial interleaved RoPE on the
 *              trailing 64 channels, per-head learnable attention sink, and a *grouped*
 *              low-rank output projection (8 groups -> 1024 each -> mixed to hidden).
 *   context    three layer types. Every layer has a 128-token sliding window; CSA layers add
 *              a 4:1 compressed KV series selected per query by a Lightning Indexer, HCA
 *              layers add a 128:1 compressed series visible in full. The compressed entries
 *              are concatenated onto the sliding keys and masked per query.
 *   MoE        256 experts, top-6, sqrt(softplus(x)) router scores, clamped SwiGLU. The first
 *              layers route by a FROZEN tid2eid[token_id] hash table, not by the router —
 *              the router still supplies the combine weights.
 *
 * Traps this code exists to get right, each read off the reference rather than guessed
 * (transformers/models/deepseek_v4/modeling_deepseek_v4.py):
 *   - RoPE is INTERLEAVED (pairs consecutive channels, GPT-J style), not the half-split
 *     rotation every other engine here uses, and it applies to the TRAILING rope_dim
 *     channels of each head with the leading channels passed through.
 *   - K == V, so the value carries RoPE. The attention output is therefore un-rotated by
 *     the CONJUGATE rotation (-sin) at the query position before the output projection.
 *     Skip that and you get plausible-looking garbage.
 *   - the CSA compressor emits TWO series per token (Ca | Cb) in one 2*head_dim tensor.
 *     Compressed entry w is a softmax-gated mix of window w-1's Ca with window w's Cb —
 *     width 2m, stride m. Window 0 of a forward call needs the PREVIOUS call's last-window
 *     Ca, which is the only cross-call state besides the running entry list.
 *   - the compressor's window softmax is PER CHANNEL over the window slots, not per row.
 *   - the gate carries `position_bias` added BEFORE the softmax, and the saved overlap
 *     slice is the already-biased gate.
 *   - router weights come from the UNBIASED score; e_score_correction_bias only picks
 *     which experts win. Then renormalize, then scale by routed_scaling_factor.
 *   - the swiglu clamp is asymmetric: gate.clamp(max=limit), up.clamp(-limit, limit).
 *   - mHC's `comb` is consumed TRANSPOSED (sum over the first stream axis). Sinkhorn
 *     produces a doubly-stochastic but NON-symmetric matrix, so the direction is load-bearing.
 *
 * Validation: SNAP=/tmp/tds ./deepseek /tmp/tds/ref_deepseek.json must be token-exact
 * against transformers (tools/make_tiny_deepseek.py builds both).
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "compat.h"
#include "json.h"
#include "moe_math.h"
#include "moe_matmul.h"
#include "moe_quant.h"         /* matmul_fp4_k: e2m1 + ue8m0 block scales (V4's native experts) */
#include "moe_sample.h"
#include "st.h"
#include "tok.h"

#define MAXL   128
#define MAXHC  8
#define MAXROT 512

static int g_exact = 0;         /* double-accumulate GEMM: the oracle contract */

/* ---------- config ---------- */
typedef enum { ATT_SLIDING = 0, ATT_CSA = 1, ATT_HCA = 2 } AttType;

typedef struct {
    int n_layers, D, V;
    int n_heads, head_dim, q_lora;
    int rdim;                       /* qk_rope_head_dim: rotated channels per head */
    int E, K, moe_I, shared_I;
    float route_scale, swiglu_limit;
    int norm_topk;
    int win;                        /* sliding_window */
    int o_groups, o_rank;
    int idx_heads, idx_dim, idx_topk;
    int hc_mult, hc_iters;
    float hc_eps, eps;
    int att[MAXL];                  /* AttType per layer */
    int hash[MAXL];                 /* mlp_layer_types[i] == "hash_moe" */
    int m_csa, m_hca;               /* compress rates */
    float inv_main[MAXROT], inv_comp[MAXROT];   /* rdim/2 entries each */
    float ms_main, ms_comp;         /* YaRN attention_factor: scales cos AND sin */
    int ctx_max;
    int tie_embed;                  /* tie_word_embeddings: lm_head IS the embedding */
    int eos[8], n_eos;
} Cfg;

/* A projection matrix as it is STORED. The oracle fixtures ship f32; the real
 * container ships bf16, and keeping it bf16 in RAM instead of expanding on load
 * halves the resident tier — ~25 GB -> ~12.5 GB across V4-Flash's 43 layers, which
 * is the difference between fitting a 187 GB box and not. It is also lossless for
 * this checkpoint: the dense tensors are fp8 e4m3 (3 mantissa bits) times a
 * power-of-two block scale, which bf16's 8 mantissa bits represent exactly.
 * Activations stay f32 (round_x=0), so this is weight-only. */
typedef struct { const float *f; const uint16_t *b; } Wt;

typedef struct {
    float *ln1, *ln2;
    /* attention */
    Wt q_a, q_b, kv_w, o_a, o_b;
    float *q_an, *kv_n, *sinks;
    /* compressor (CSA / HCA) */
    Wt c_kv, c_gate;
    float *c_bias, *c_norm;
    /* lightning indexer (CSA only) */
    Wt i_kv, i_gate, i_qb, i_w;
    float *i_bias, *i_norm;
    /* mHC: attention site and ffn site */
    float *a_fn, *a_base, *a_scale;
    float *f_fn, *f_base, *f_scale;
    /* MoE */
    Wt router;
    float *corr;
    int64_t *tid2eid;
    float *e_gu, *e_dn;             /* f32 experts: [E, 2I, D] and [E, D, I] */
    uint8_t *q_gu, *q_dn;           /* fp4 container: e2m1 nibbles [E*2I, D/2], [E*D, I/2] */
    uint8_t *s_gu, *s_dn;           /* ...and their ue8m0 block scales [E*2I, D/32], [E*D, I/32] */
    Wt sg, su, sd;                  /* shared expert */
} Layer;

typedef struct {
    shards S;
    Cfg c;
    Layer L[MAXL];
    Wt embed, lm_head;
    float *norm;
    float *h_fn, *h_base, *h_scale; /* hc_head: final stream collapse */
    int xq;                         /* 1 = fp4 routed-expert container */
} Model;

/* Per-layer sequence state. Two "entries" per layer share the compressor machinery:
 * slot 0 is the outer compressor, slot 1 the indexer's own scaled-down compressor. */
typedef struct {
    float *kv;        /* sliding ring, [win-1][head_dim], oldest first */
    int   *kpos;
    int    nkv;
    float *buf_kv[2], *buf_gate[2];   /* partial window carried across calls, [m][width] */
    int    nbuf[2];
    float *comp[2];                   /* running compressed entries, [ncomp][dim] */
    int    ncomp[2], cap[2], ecount[2];
    float *ov_kv[2], *ov_gate[2];     /* CSA overlap: previous window's Ca slice, [m][dim] */
    int    has_ov[2];
} LState;

typedef struct { LState *l; int len; } State;

static float *falloc(int64_t n) {
    float *p = malloc((size_t)n * sizeof(float));
    if (!p) { fprintf(stderr, "OOM %lld floats\n", (long long)n); exit(1); }
    return p;
}
static float *fzalloc(int64_t n) {
    float *p = calloc((size_t)n, sizeof(float));
    if (!p) { fprintf(stderr, "OOM %lld floats\n", (long long)n); exit(1); }
    return p;
}

/* ---------- config loading ---------- */
static double jnum(jval *o, const char *k, double dflt) {
    jval *v = json_get(o, k);
    return (v && v->t == J_NUM) ? v->num : dflt;
}

/* YaRN inverse frequencies, matching transformers' _compute_yarn_parameters.
 *
 * YaRN also returns an `attention_factor` (mscale) that the rotary embedding multiplies
 * into BOTH cos and sin. It is NOT 1.0 here: with no explicit override in the config it
 * is 0.1*ln(factor)+1, which is 1.2773 for V4-Flash's compress rope (factor 16). An
 * earlier note in this file claimed the reference pinned it to 1.0; it does not, and
 * dropping it left every CSA/HCA layer subtly wrong while sliding layers stayed exact
 * (the main rope is not YaRN, so its factor really is 1.0). */
static void yarn_inv(float *inv, int dim, double base, double factor,
                     double beta_fast, double beta_slow, int orig_max) {
    double lo = dim * log((double)orig_max / (beta_fast * 2 * M_PI)) / (2 * log(base));
    double hi = dim * log((double)orig_max / (beta_slow * 2 * M_PI)) / (2 * log(base));
    lo = floor(lo); hi = ceil(hi);
    if (lo < 0) lo = 0;
    if (hi > dim - 1) hi = dim - 1;
    if (hi == lo) hi = lo + 0.001;
    for (int j = 0; j < dim / 2; j++) {
        double pos_freq = pow(base, (double)(2 * j) / dim);
        double extrap = 1.0 / pos_freq, interp = 1.0 / (factor * pos_freq);
        double ramp = ((double)j - lo) / (hi - lo);
        if (ramp < 0) ramp = 0;
        if (ramp > 1) ramp = 1;
        double ext_factor = 1.0 - ramp;              /* 1 - linear_ramp_factor */
        inv[j] = (float)(interp * (1 - ext_factor) + extrap * ext_factor);
    }
}

/* 0.1*mscale*ln(scale)+1, transformers' get_mscale */
static double get_mscale(double scale, double mscale) {
    return scale <= 1.0 ? 1.0 : 0.1 * mscale * log(scale) + 1.0;
}

/* Fills `inv` and returns the attention_factor that scales cos/sin. */
static float rope_table(float *inv, int dim, jval *rp, double dflt_theta, int ctx_max) {
    double theta = rp ? jnum(rp, "rope_theta", dflt_theta) : dflt_theta;
    jval *rt = rp ? json_get(rp, "rope_type") : NULL;
    if (!rt) rt = rp ? json_get(rp, "type") : NULL;
    if (rt && rt->t == J_STR && !strcmp(rt->str, "yarn")) {
        int om = (int)jnum(rp, "original_max_position_embeddings", ctx_max);
        double factor = jnum(rp, "factor", 0.0);
        if (factor <= 0) factor = om ? (double)ctx_max / om : 1.0;   /* DeepSeek-V3 style */
        double bf = jnum(rp, "beta_fast", 32.0), bs = jnum(rp, "beta_slow", 1.0);
        yarn_inv(inv, dim, theta, factor, bf, bs, om);
        jval *af = json_get(rp, "attention_factor");
        if (af && af->t == J_NUM) return (float)af->num;
        jval *ms = json_get(rp, "mscale"), *msa = json_get(rp, "mscale_all_dim");
        if (ms && ms->t == J_NUM && msa && msa->t == J_NUM && ms->num && msa->num)
            return (float)(get_mscale(factor, ms->num) / get_mscale(factor, msa->num));
        return (float)get_mscale(factor, 1.0);
    }
    for (int j = 0; j < dim / 2; j++) inv[j] = (float)(1.0 / pow(theta, (double)(2 * j) / dim));
    return 1.f;
}

static void cfg_load(Cfg *c, const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read config\n"); exit(1); }
    buf[n] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    jval *o = json_get(root, "text_config");
    if (!o) o = root;

    c->n_layers = (int)jnum(o, "num_hidden_layers", 0);
    if (c->n_layers <= 0 || c->n_layers > MAXL) {
        fprintf(stderr, "unsupported num_hidden_layers %d\n", c->n_layers); exit(1); }
    c->D = (int)jnum(o, "hidden_size", 0);
    c->V = (int)jnum(o, "vocab_size", 0);
    c->n_heads = (int)jnum(o, "num_attention_heads", 0);
    c->head_dim = (int)jnum(o, "head_dim", 0);
    c->q_lora = (int)jnum(o, "q_lora_rank", 0);
    c->E = (int)jnum(o, "n_routed_experts", 0);
    c->K = (int)jnum(o, "num_experts_per_tok", 0);
    c->moe_I = (int)jnum(o, "moe_intermediate_size", 0);
    c->route_scale = (float)jnum(o, "routed_scaling_factor", 1.0);
    c->norm_topk = (int)jnum(o, "norm_topk_prob", 1);
    c->swiglu_limit = (float)jnum(o, "swiglu_limit", 10.0);
    c->win = (int)jnum(o, "sliding_window", 128);
    c->o_groups = (int)jnum(o, "o_groups", 1);
    c->o_rank = (int)jnum(o, "o_lora_rank", 0);
    c->idx_heads = (int)jnum(o, "index_n_heads", 0);
    c->idx_dim = (int)jnum(o, "index_head_dim", 0);
    c->idx_topk = (int)jnum(o, "index_topk", 0);
    c->hc_mult = (int)jnum(o, "hc_mult", 4);
    c->hc_iters = (int)jnum(o, "hc_sinkhorn_iters", 20);
    c->hc_eps = (float)jnum(o, "hc_eps", 1e-6);
    c->eps = (float)jnum(o, "rms_norm_eps", 1e-6);
    c->ctx_max = (int)jnum(o, "max_position_embeddings", 4096);
    { jval *t = json_get(o, "tie_word_embeddings");
      c->tie_embed = t ? (t->t == J_BOOL ? t->boolean : (t->t == J_NUM ? (int)t->num : 0)) : 0; }
    if (c->hc_mult < 1 || c->hc_mult > MAXHC) {
        fprintf(stderr, "hc_mult %d out of range (max %d)\n", c->hc_mult, MAXHC); exit(1); }
    if (c->o_groups < 1 || (c->n_heads * c->head_dim) % c->o_groups) {
        fprintf(stderr, "o_groups %d does not divide n_heads*head_dim\n", c->o_groups); exit(1); }

    /* compress_rates is a dict keyed by attention type; legacy configs ship scalars. */
    jval *cr = json_get(o, "compress_rates");
    c->m_csa = (int)jnum(o, "compress_rate_csa", 4);
    c->m_hca = (int)jnum(o, "compress_rate_hca", 128);
    if (cr && cr->t == J_OBJ) {
        c->m_csa = (int)jnum(cr, "compressed_sparse_attention", c->m_csa);
        c->m_hca = (int)jnum(cr, "heavily_compressed_attention", c->m_hca);
    }
    if (c->m_csa < 1 || c->m_hca < 1) { fprintf(stderr, "bad compress rates\n"); exit(1); }

    /* layer_types: explicit list, else legacy compress_ratios (0/4/128), else the
     * config-class default (2x HCA bootstrap then HCA/CSA interleave). */
    jval *lt = json_get(o, "layer_types");
    jval *cro = json_get(o, "compress_ratios");
    if (lt && lt->t == J_ARR) {
        if (lt->len != c->n_layers) {
            fprintf(stderr, "layer_types has %d entries, expected %d\n", lt->len, c->n_layers); exit(1); }
        for (int i = 0; i < c->n_layers; i++) {
            const char *s = lt->kids[i]->str;
            if (!strcmp(s, "sliding_attention")) c->att[i] = ATT_SLIDING;
            else if (!strcmp(s, "compressed_sparse_attention")) c->att[i] = ATT_CSA;
            else if (!strcmp(s, "heavily_compressed_attention")) c->att[i] = ATT_HCA;
            else { fprintf(stderr, "unknown layer type '%s'\n", s); exit(1); }
        }
    } else if (cro && cro->t == J_ARR && cro->len >= c->n_layers) {
        /* >=, not ==: the shipped V4-Flash-0731 config carries 46 ratios for 43 layers —
         * the trailing entries describe the DSpark speculative-decoding module, which is
         * a separate `mtp.*` stack this engine does not run. Demanding an exact length
         * silently fell through to the default interleave, which is NOT this checkpoint's
         * layout (it opens with two SLIDING layers, not two HCA). */
        for (int i = 0; i < c->n_layers; i++) {
            int r = (int)cro->kids[i]->num;
            c->att[i] = r == 0 ? ATT_SLIDING : (r == c->m_hca ? ATT_HCA : ATT_CSA);
        }
    } else {
        for (int i = 0; i < c->n_layers; i++)
            c->att[i] = i < 2 ? ATT_HCA : ((i - 2) % 2 ? ATT_CSA : ATT_HCA);
    }

    /* mlp_layer_types: leading hash_moe layers route by the frozen tid2eid table. */
    jval *mt = json_get(o, "mlp_layer_types");
    if (mt && mt->t == J_ARR && mt->len == c->n_layers) {
        for (int i = 0; i < c->n_layers; i++) c->hash[i] = !strcmp(mt->kids[i]->str, "hash_moe");
    } else {
        int nh = (int)jnum(o, "num_hash_layers", 3);
        for (int i = 0; i < c->n_layers; i++) c->hash[i] = i < nh;
    }

    /* Partial interleaved rotary. rope_parameters is keyed by rope TYPE (main / compress),
     * not by layer type: sliding layers use main (theta 10000, no scaling), CSA/HCA layers
     * and every compressor use compress (theta 160000, usually YaRN). */
    jval *rp = json_get(o, "rope_parameters");
    jval *rmain = rp ? json_get(rp, "main") : NULL;
    jval *rcomp = rp ? json_get(rp, "compress") : NULL;
    double prf = rmain ? jnum(rmain, "partial_rotary_factor", 0) : 0;
    if (prf <= 0) prf = jnum(o, "partial_rotary_factor", 0);
    if (prf <= 0) {
        double qk = jnum(o, "qk_rope_head_dim", 0);
        prf = qk > 0 ? qk / c->head_dim : 64.0 / 512.0;
    }
    c->rdim = (int)(c->head_dim * prf) & ~1;
    if (c->rdim < 2 || c->rdim / 2 > MAXROT) {
        fprintf(stderr, "rotary dim %d unsupported\n", c->rdim); exit(1); }
    c->ms_main = rope_table(c->inv_main, c->rdim, rmain, jnum(o, "rope_theta", 10000.0), c->ctx_max);
    c->ms_comp = rope_table(c->inv_comp, c->rdim, rcomp, jnum(o, "compress_rope_theta", 160000.0), c->ctx_max);

    c->n_eos = 0;
    for (int pass = 0; pass < 2 && !c->n_eos; pass++) {
        jval *src = NULL; char *gbuf = NULL; char *garena = NULL;
        if (pass == 0) {
            char gp[1024]; snprintf(gp, sizeof(gp), "%s/generation_config.json", dir);
            FILE *gf = fopen(gp, "rb");
            if (gf) {
                fseek(gf, 0, SEEK_END); long gn = ftell(gf); fseek(gf, 0, SEEK_SET);
                gbuf = malloc(gn + 1);
                if (fread(gbuf, 1, gn, gf) == (size_t)gn) { gbuf[gn] = 0; src = json_parse(gbuf, &garena); }
                fclose(gf);
            }
        } else src = o;
        jval *e = src ? json_get(src, "eos_token_id") : NULL;
        if (e && e->t == J_NUM) c->eos[c->n_eos++] = (int)e->num;
        else if (e && e->t == J_ARR)
            for (int i = 0; i < e->len && c->n_eos < 8; i++)
                if (e->kids[i]->t == J_NUM) c->eos[c->n_eos++] = (int)e->kids[i]->num;
        free(gbuf); free(garena);
    }
    free(buf); free(arena);
}

/* ---------- weight loading ---------- */
static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    float *p = falloc(n); st_read_f32(&m->S, name, p, 0); return p;
}
static float *load_opt(Model *m, const char *name) {
    return st_numel(&m->S, name) < 0 ? NULL : load_t(m, name);
}
/* A projection: kept bf16 when the file stores bf16, expanded to f32 otherwise.
 * st.h dtype 0 == BF16. */
static Wt load_w(Model *m, const char *name) {
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    Wt w = {NULL, NULL};
    if (t->dtype == 0) {
        uint16_t *p = malloc((size_t)t->numel * sizeof(uint16_t));
        if (!p) { fprintf(stderr, "OOM %lld bf16 for %s\n", (long long)t->numel, name); exit(1); }
        st_read_raw(&m->S, name, p, 1);
        w.b = p;
    } else {
        w.f = load_t(m, name);
    }
    return w;
}
static Wt load_w_opt(Model *m, const char *name) {
    Wt z = {NULL, NULL};
    return st_numel(&m->S, name) < 0 ? z : load_w(m, name);
}
static int wt_null(Wt w) { return !w.f && !w.b; }

/* raw bytes of an already-quantized tensor (fp4 nibbles, ue8m0 scales) */
static uint8_t *load_u8(Model *m, const char *name) {
    int64_t nb = st_nbytes(&m->S, name);
    if (nb < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    uint8_t *p = malloc((size_t)nb);
    if (!p) { fprintf(stderr, "OOM %lld bytes for %s\n", (long long)nb, name); exit(1); }
    st_read_raw(&m->S, name, p, 1);
    return p;
}

/* Routed experts ship in three shapes; take whichever the snapshot has.
 *
 *  1. fp4 container (a `.es` block-scale sidecar next to the packed tensor) —
 *     what tools/convert_deepseek_fp4.py emits, and the only one that fits the
 *     real 284B checkpoint. Stored verbatim from the checkpoint's own e2m1 bytes.
 *  2. fused f32 `gate_up_proj` [E,2I,D] / `down_proj` [E,D,I] — what the pure-python
 *     harness (tools/dsv4_ref.py) writes, and what transformers <= 5.7 saved.
 *  3. per-expert f32 `experts.{e}.w1/w2/w3` — what transformers saves TODAY, and
 *     the layout the real checkpoint uses. Fused here at load time.
 *
 * gate_up is block-concatenated per expert (all I gate rows, then all I up rows),
 * matching F.linear(x, gate_up_proj[e]).chunk(2). */
static void load_experts(Model *m, Layer *L, int i) {
    Cfg *c = &m->c;
    char n[256], es[256];
    snprintf(n, sizeof(n), "model.layers.%d.mlp.experts.gate_up_proj", i);
    snprintf(es, sizeof(es), "model.layers.%d.mlp.experts.gate_up_proj.es", i);

    if (st_numel(&m->S, es) >= 0) {                       /* 1. fp4 container */
        m->xq = 1;
        if (c->moe_I <= 0) {
            fprintf(stderr, "fp4 container needs moe_intermediate_size in config.json\n"); exit(1); }
        int64_t want = (int64_t)c->E * 2 * c->moe_I * (c->D / 2);
        if (st_nbytes(&m->S, n) != want) {
            fprintf(stderr, "layer %d gate_up_proj: %lld bytes, expected %lld (E=%d I=%d D=%d)\n",
                    i, (long long)st_nbytes(&m->S, n), (long long)want, c->E, c->moe_I, c->D); exit(1); }
        L->q_gu = load_u8(m, n);
        L->s_gu = load_u8(m, es);
        snprintf(n, sizeof(n), "model.layers.%d.mlp.experts.down_proj", i);
        snprintf(es, sizeof(es), "model.layers.%d.mlp.experts.down_proj.es", i);
        L->q_dn = load_u8(m, n);
        L->s_dn = load_u8(m, es);
        return;
    }
    if (st_numel(&m->S, n) >= 0) {                        /* 2. fused f32 */
        L->e_gu = load_t(m, n);
        snprintf(n, sizeof(n), "model.layers.%d.mlp.experts.down_proj", i);
        L->e_dn = load_t(m, n);
        return;
    }
    /* 3. per-expert f32 w1/w2/w3 -> fuse */
    snprintf(n, sizeof(n), "model.layers.%d.mlp.experts.0.w1.weight", i);
    int64_t w1n = st_numel(&m->S, n);
    if (w1n < 0) { fprintf(stderr, "layer %d: no routed experts in any known layout\n", i); exit(1); }
    int I = (int)(w1n / c->D);
    if (c->moe_I <= 0) c->moe_I = I;
    if (I != c->moe_I) {
        fprintf(stderr, "layer %d expert width %d disagrees with moe_intermediate_size %d\n",
                i, I, c->moe_I); exit(1); }
    L->e_gu = falloc((int64_t)c->E * 2 * I * c->D);
    L->e_dn = falloc((int64_t)c->E * c->D * I);
    for (int e = 0; e < c->E; e++) {
        snprintf(n, sizeof(n), "model.layers.%d.mlp.experts.%d.w1.weight", i, e);   /* gate */
        st_read_f32_cap(&m->S, n, L->e_gu + (int64_t)e * 2 * I * c->D, (int64_t)I * c->D, 0);
        snprintf(n, sizeof(n), "model.layers.%d.mlp.experts.%d.w3.weight", i, e);   /* up */
        st_read_f32_cap(&m->S, n, L->e_gu + ((int64_t)e * 2 * I + I) * c->D, (int64_t)I * c->D, 0);
        snprintf(n, sizeof(n), "model.layers.%d.mlp.experts.%d.w2.weight", i, e);   /* down */
        st_read_f32_cap(&m->S, n, L->e_dn + (int64_t)e * c->D * I, (int64_t)c->D * I, 0);
    }
}

static void model_load(Model *m, const char *snap) {
    st_init(&m->S, snap);
    cfg_load(&m->c, snap);
    Cfg *c = &m->c;
    char n[256];

    m->embed = load_w(m, "model.embed_tokens.weight");
    m->norm = load_t(m, "model.norm.weight");
    /* Output projection. V4 calls it `head.weight` — both in the shipped checkpoint and
     * in what transformers' save_pretrained writes for this arch — while other engines
     * here see `lm_head.weight`. Take either. Falling back to the embedding is ONLY
     * correct when the config actually ties them; doing it silently on a missing tensor
     * turned an untied model into a tied one and cost a full oracle failure (the whole
     * stack matched transformers layer-for-layer, and only the logits were wrong). */
    m->lm_head = load_w_opt(m, "lm_head.weight");
    if (wt_null(m->lm_head)) m->lm_head = load_w_opt(m, "head.weight");
    if (wt_null(m->lm_head)) {
        if (!c->tie_embed) {
            fprintf(stderr, "no output projection (lm_head.weight / head.weight) and "
                            "tie_word_embeddings is false\n"); exit(1); }
        m->lm_head = m->embed;
    }
    m->h_fn = load_t(m, "model.hc_head.hc_fn");
    m->h_base = load_t(m, "model.hc_head.hc_base");
    m->h_scale = load_t(m, "model.hc_head.hc_scale");

    for (int i = 0; i < c->n_layers; i++) {
        Layer *L = &m->L[i];
#define LT(dst, fmt) do { snprintf(n, sizeof(n), fmt, i); L->dst = load_t(m, n); } while (0)
#define LO(dst, fmt) do { snprintf(n, sizeof(n), fmt, i); L->dst = load_opt(m, n); } while (0)
#define LW(dst, fmt) do { snprintf(n, sizeof(n), fmt, i); L->dst = load_w(m, n); } while (0)
        LT(ln1, "model.layers.%d.input_layernorm.weight");
        LT(ln2, "model.layers.%d.post_attention_layernorm.weight");
        LW(q_a, "model.layers.%d.self_attn.q_a_proj.weight");
        LT(q_an, "model.layers.%d.self_attn.q_a_norm.weight");
        LW(q_b, "model.layers.%d.self_attn.q_b_proj.weight");
        LW(kv_w, "model.layers.%d.self_attn.kv_proj.weight");
        LT(kv_n, "model.layers.%d.self_attn.kv_norm.weight");
        LW(o_a, "model.layers.%d.self_attn.o_a_proj.weight");
        LW(o_b, "model.layers.%d.self_attn.o_b_proj.weight");
        LT(sinks, "model.layers.%d.self_attn.sinks");
        LT(a_fn, "model.layers.%d.attn_hc.fn");
        LT(a_base, "model.layers.%d.attn_hc.base");
        LT(a_scale, "model.layers.%d.attn_hc.scale");
        LT(f_fn, "model.layers.%d.ffn_hc.fn");
        LT(f_base, "model.layers.%d.ffn_hc.base");
        LT(f_scale, "model.layers.%d.ffn_hc.scale");
        LW(router, "model.layers.%d.mlp.gate.weight");
        LO(corr, "model.layers.%d.mlp.gate.e_score_correction_bias");
        load_experts(m, L, i);
        LW(sg, "model.layers.%d.mlp.shared_experts.gate_proj.weight");
        LW(su, "model.layers.%d.mlp.shared_experts.up_proj.weight");
        LW(sd, "model.layers.%d.mlp.shared_experts.down_proj.weight");

        if (c->att[i] != ATT_SLIDING) {
            LW(c_kv, "model.layers.%d.self_attn.compressor.kv_proj.weight");
            LW(c_gate, "model.layers.%d.self_attn.compressor.gate_proj.weight");
            LT(c_bias, "model.layers.%d.self_attn.compressor.position_bias");
            LT(c_norm, "model.layers.%d.self_attn.compressor.kv_norm.weight");
        }
        if (c->att[i] == ATT_CSA) {
            LW(i_kv, "model.layers.%d.self_attn.compressor.indexer.kv_proj.weight");
            LW(i_gate, "model.layers.%d.self_attn.compressor.indexer.gate_proj.weight");
            LT(i_bias, "model.layers.%d.self_attn.compressor.indexer.position_bias");
            LT(i_norm, "model.layers.%d.self_attn.compressor.indexer.kv_norm.weight");
            LW(i_qb, "model.layers.%d.self_attn.compressor.indexer.q_b_proj.weight");
            LW(i_w, "model.layers.%d.self_attn.compressor.indexer.scorer.weights_proj.weight");
        }
        if (c->hash[i]) {
            snprintf(n, sizeof(n), "model.layers.%d.mlp.gate.tid2eid", i);
            int64_t cnt = st_numel(&m->S, n);
            if (cnt < (int64_t)c->V * c->K) {
                fprintf(stderr, "hash layer %d: tid2eid has %lld entries, expected %lld\n",
                        i, (long long)cnt, (long long)c->V * c->K); exit(1); }
            L->tid2eid = malloc((size_t)cnt * sizeof(int64_t));
            if (!L->tid2eid) { fprintf(stderr, "OOM tid2eid\n"); exit(1); }
            st_read_raw(&m->S, n, L->tid2eid, 0);
        }
#undef LT
#undef LO
#undef LW
    }
    /* Shared-expert width is whatever the checkpoint ships (the reference reads it from
     * `intermediate_size`, which attribute-maps onto moe_intermediate_size). Derive it
     * rather than assume, so a checkpoint with a wider shared expert still loads. */
    c->shared_I = (int)(st_numel(&m->S, "model.layers.0.mlp.shared_experts.gate_proj.weight") / c->D);
    /* load_experts() already cross-checked the routed width against the config for
     * the fp4 and per-expert layouts; the fused f32 tensor is the remaining case. */
    int64_t egu = st_numel(&m->S, "model.layers.0.mlp.experts.gate_up_proj");
    if (!m->xq && egu >= 0) {
        int derived = (int)(egu / ((int64_t)c->E * 2 * c->D));
        if (c->moe_I <= 0) c->moe_I = derived;
        if (derived != c->moe_I) {
            fprintf(stderr, "expert width %d disagrees with moe_intermediate_size %d\n", derived, c->moe_I);
            exit(1); }
    }
    if (c->moe_I <= 0) { fprintf(stderr, "could not determine routed expert width\n"); exit(1); }
}

/* ---------- per-sequence state ---------- */
static void state_init(Model *m, State *s) {
    Cfg *c = &m->c;
    s->l = calloc(c->n_layers, sizeof(LState));
    s->len = 0;
    for (int i = 0; i < c->n_layers; i++) {
        LState *L = &s->l[i];
        L->kv = falloc((int64_t)(c->win > 1 ? c->win - 1 : 1) * c->head_dim);
        L->kpos = malloc(sizeof(int) * (c->win > 1 ? c->win - 1 : 1));
        if (c->att[i] == ATT_SLIDING) continue;
        int m_r = c->att[i] == ATT_CSA ? c->m_csa : c->m_hca;
        int wid = c->att[i] == ATT_CSA ? 2 * c->head_dim : c->head_dim;
        L->buf_kv[0] = falloc((int64_t)m_r * wid);
        L->buf_gate[0] = falloc((int64_t)m_r * wid);
        if (c->att[i] == ATT_CSA) {
            L->ov_kv[0] = falloc((int64_t)m_r * c->head_dim);
            L->ov_gate[0] = falloc((int64_t)m_r * c->head_dim);
            L->buf_kv[1] = falloc((int64_t)m_r * 2 * c->idx_dim);
            L->buf_gate[1] = falloc((int64_t)m_r * 2 * c->idx_dim);
            L->ov_kv[1] = falloc((int64_t)m_r * c->idx_dim);
            L->ov_gate[1] = falloc((int64_t)m_r * c->idx_dim);
        }
    }
}

static void state_free(Model *m, State *s) {
    for (int i = 0; i < m->c.n_layers; i++) {
        LState *L = &s->l[i];
        free(L->kv); free(L->kpos);
        for (int k = 0; k < 2; k++) {
            free(L->buf_kv[k]); free(L->buf_gate[k]); free(L->comp[k]);
            free(L->ov_kv[k]); free(L->ov_gate[k]);
        }
    }
    free(s->l);
}

/* ---------- small kernels ---------- */
static void mm(float *y, const float *x, const float *W, int S, int I, int O) {
    matmul_f32(y, x, W, S, I, O, g_exact);
}

static void mmw(float *y, const float *x, Wt w, int S, int I, int O) {
    if (w.b) matmul_bf16_k(y, x, w.b, S, I, O, 0, g_exact);
    else     matmul_f32(y, x, w.f, S, I, O, g_exact);
}
static inline const float *wt_row_f32(Wt w, int64_t off, float *tmp, int n) {
    if (!w.b) return w.f + off;
    for (int i = 0; i < n; i++) {                    /* bf16 -> f32 for a raw row read */
        uint32_t u = (uint32_t)w.b[off + i] << 16;
        memcpy(&tmp[i], &u, 4);
    }
    return tmp;
}

/* RMSNorm with no learned weight (DeepseekV4UnweightedRMSNorm). */
static void rmsnorm_plain(float *out, const float *x, int D, float eps) {
    double ss = 0; for (int i = 0; i < D; i++) ss += (double)x[i] * x[i];
    float inv = 1.f / sqrtf((float)(ss / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * inv;
}

/* Interleaved partial RoPE on the TRAILING rdim channels of one head. sgn = -1 applies
 * the conjugate rotation (used to un-rotate the attention output, since K == V). */
static void rope_head(float *x, int hd, int rdim, const float *inv, int pos, float sgn, float ms) {
    float *r = x + hd - rdim;
    for (int j = 0; j < rdim / 2; j++) {
        float ang = (float)pos * inv[j];
        float cs = cosf(ang) * ms, sn = sgn * sinf(ang) * ms;
        float a = r[2 * j], b = r[2 * j + 1];
        r[2 * j]     = a * cs - b * sn;
        r[2 * j + 1] = b * cs + a * sn;
    }
}

/* ---------- Manifold-Constrained Hyper-Connections ----------
 * One site (attention or ffn). Reads the hc_mult residual streams for one token, returns
 * `post` (where the sublayer output lands), `comb` (the doubly-stochastic stream mixer)
 * and `collapsed` (the single vector the sublayer actually consumes). */
static void hc_site(const Cfg *c, const float *fn, const float *base, const float *scale,
                    const float *streams, float *post, float *comb, float *collapsed,
                    float *flat, float *mix) {
    int hc = c->hc_mult, D = c->D, mixn = (2 + hc) * hc;
    rmsnorm_plain(flat, streams, hc * D, c->eps);
    mm(mix, flat, fn, 1, hc * D, mixn);

    float eps = c->hc_eps, pre[MAXHC];
    for (int k = 0; k < hc; k++)
        pre[k] = 1.f / (1.f + expf(-(mix[k] * scale[0] + base[k]))) + eps;
    for (int k = 0; k < hc; k++)
        post[k] = 2.f / (1.f + expf(-(mix[hc + k] * scale[1] + base[hc + k])));

    /* comb: softmax rows, then Sinkhorn-Knopp onto the doubly-stochastic manifold.
     * The first normalization is over COLUMNS (dim=-2), then iters-1 x (rows, columns). */
    for (int j = 0; j < hc; j++) {
        float *row = comb + j * hc;
        for (int k = 0; k < hc; k++)
            row[k] = mix[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
        softmax_row(row, hc);
        for (int k = 0; k < hc; k++) row[k] += eps;
    }
    for (int k = 0; k < hc; k++) {
        float s = 0; for (int j = 0; j < hc; j++) s += comb[j * hc + k];
        s += eps; for (int j = 0; j < hc; j++) comb[j * hc + k] /= s;
    }
    for (int it = 0; it < c->hc_iters - 1; it++) {
        for (int j = 0; j < hc; j++) {
            float s = 0; for (int k = 0; k < hc; k++) s += comb[j * hc + k];
            s += eps; for (int k = 0; k < hc; k++) comb[j * hc + k] /= s;
        }
        for (int k = 0; k < hc; k++) {
            float s = 0; for (int j = 0; j < hc; j++) s += comb[j * hc + k];
            s += eps; for (int j = 0; j < hc; j++) comb[j * hc + k] /= s;
        }
    }
    for (int i = 0; i < D; i++) {
        float a = 0; for (int k = 0; k < hc; k++) a += pre[k] * streams[k * D + i];
        collapsed[i] = a;
    }
}

/* streams[k] = post[k]*y + sum_j comb[j][k]*streams[j]   (comb consumed TRANSPOSED) */
static void hc_merge(const Cfg *c, float *streams, const float *post, const float *comb,
                     const float *y, float *tmp) {
    int hc = c->hc_mult, D = c->D;
    memcpy(tmp, streams, (size_t)hc * D * sizeof(float));
    for (int k = 0; k < hc; k++) {
        float *dst = streams + (int64_t)k * D;
        for (int i = 0; i < D; i++) {
            float a = post[k] * y[i];
            for (int j = 0; j < hc; j++) a += comb[j * hc + k] * tmp[j * D + i];
            dst[i] = a;
        }
    }
}

/* ---------- MoE ---------- */
/* One routed expert: clamped SwiGLU over the fused gate_up, then down. Reads either
 * the f32 tensors or the fp4 container — same arithmetic, different weight decode. */
static void swiglu_expert(const Model *m, const Layer *L, int e, const float *x,
                          float *out, float *h, int I) {
    const Cfg *c = &m->c;
    int D = c->D;
    float lim = c->swiglu_limit;
    float *g = h, *u = h + I;
    if (m->xq)                                          /* rows [0,I) gate, [I,2I) up */
        matmul_fp4_k(g, x, L->q_gu + (int64_t)e * 2 * I * (D / 2),
                     L->s_gu + (int64_t)e * 2 * I * (D / 32), D, 2 * I, g_exact);
    else
        mm(g, x, L->e_gu + (int64_t)e * 2 * I * D, 1, D, 2 * I);
    for (int i = 0; i < I; i++) {
        float gi = g[i] > lim ? lim : g[i];
        float ui = u[i] > lim ? lim : (u[i] < -lim ? -lim : u[i]);
        g[i] = siluf(gi) * ui;
    }
    if (m->xq)
        matmul_fp4_k(out, g, L->q_dn + (int64_t)e * D * (I / 2),
                     L->s_dn + (int64_t)e * D * (I / 32), I, D, g_exact);
    else
        mm(out, g, L->e_dn + (int64_t)e * D * I, 1, I, D);
}

static void moe_row(Model *m, Layer *L, int li, const float *x, int tok, float *out,
                    float *scratch) {
    Cfg *c = &m->c;
    int E = c->E, K = c->K, D = c->D, I = c->moe_I;
    float *logits = scratch, *score = logits + E, *w = score + E;
    float *acc = w + K, *tmp = acc + D, *h = tmp + D;    /* h needs 2*max(I, shared_I) */
    int sel[64];
    if (K > 64) { fprintf(stderr, "top-k %d exceeds 64\n", K); exit(1); }

    mmw(logits, x, L->router, 1, D, E);
    for (int e = 0; e < E; e++) score[e] = sqrtf(softplusf(logits[e]));

    if (c->hash[li]) {
        const int64_t *row = L->tid2eid + (int64_t)tok * K;
        for (int a = 0; a < K; a++) {
            int64_t e = row[a];
            if (e < 0 || e >= E) { fprintf(stderr, "tid2eid[%d][%d] = %lld out of range\n",
                                           tok, a, (long long)e); exit(1); }
            sel[a] = (int)e;
        }
    } else {
        /* top-k on (score + correction bias); the COMBINE weight is the unbiased score */
        for (int a = 0; a < K; a++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int used = 0; for (int b = 0; b < a; b++) if (sel[b] == e) { used = 1; break; }
                if (used) continue;
                float s = score[e] + (L->corr ? L->corr[e] : 0.f);
                if (s > bv) { bv = s; best = e; }
            }
            if (best < 0) { for (int e = 0; e < E && best < 0; e++) {
                                int used = 0; for (int b = 0; b < a; b++) if (sel[b] == e) { used = 1; break; }
                                if (!used) best = e; }
                            if (best < 0) best = 0; }
            sel[a] = best;
        }
    }
    for (int a = 0; a < K; a++) w[a] = score[sel[a]];
    if (c->norm_topk) {
        float sm = 0; for (int a = 0; a < K; a++) sm += w[a];
        sm += 1e-20f; for (int a = 0; a < K; a++) w[a] /= sm;
    }
    for (int a = 0; a < K; a++) w[a] *= c->route_scale;

    /* Accumulate in ascending expert id: the reference's Experts.forward walks the hit
     * experts in ascending order and index_add_s, so this keeps the summation order. */
    for (int a = 0; a < K; a++)
        for (int b = a + 1; b < K; b++)
            if (sel[b] < sel[a]) { int t = sel[a]; sel[a] = sel[b]; sel[b] = t;
                                   float tw = w[a]; w[a] = w[b]; w[b] = tw; }

    for (int i = 0; i < D; i++) acc[i] = 0;
    for (int a = 0; a < K; a++) {
        swiglu_expert(m, L, sel[a], x, tmp, h, I);
        for (int i = 0; i < D; i++) acc[i] += w[a] * tmp[i];
    }

    /* shared expert: separate gate/up/down tensors, same clamped SwiGLU */
    int SI = c->shared_I; float lim = c->swiglu_limit;
    float *g = h, *u = h + SI;
    mmw(g, x, L->sg, 1, D, SI);
    mmw(u, x, L->su, 1, D, SI);
    for (int i = 0; i < SI; i++) {
        float gi = g[i] > lim ? lim : g[i];
        float ui = u[i] > lim ? lim : (u[i] < -lim ? -lim : u[i]);
        g[i] = siluf(gi) * ui;
    }
    mmw(tmp, g, L->sd, 1, SI, D);
    for (int i = 0; i < D; i++) out[i] = acc[i] + tmp[i];
}

/* ---------- compressors ----------
 * Shared by the HCA compressor (no overlap, width = dim), the CSA compressor and the
 * indexer's own compressor (both overlap, width = 2*dim). Consumes `n` freshly projected
 * (kv, gate) rows, closes every window that is now complete, and appends one compressed
 * entry per closed window to the running list. Returns the number of entries appended. */
static int compress_push(const Cfg *c, LState *ls, int slot, const float *kv, const float *gate,
                         int n, int m_r, int dim, int overlap, const float *pos_bias,
                         const float *norm_w) {
    int wid = overlap ? 2 * dim : dim;
    int nb = ls->nbuf[slot], total = nb + n;
    int usable = (total / m_r) * m_r, nwin = usable / m_r;
    int first_pos = ls->ecount[slot] * m_r;

    /* buffered leftover ++ new rows, so window boundaries survive across calls */
    float *ckv = falloc((int64_t)total * wid), *cgt = falloc((int64_t)total * wid);
    if (nb) {
        memcpy(ckv, ls->buf_kv[slot], (size_t)nb * wid * sizeof(float));
        memcpy(cgt, ls->buf_gate[slot], (size_t)nb * wid * sizeof(float));
    }
    memcpy(ckv + (int64_t)nb * wid, kv, (size_t)n * wid * sizeof(float));
    memcpy(cgt + (int64_t)nb * wid, gate, (size_t)n * wid * sizeof(float));
    ls->nbuf[slot] = total - usable;
    if (ls->nbuf[slot]) {
        memcpy(ls->buf_kv[slot], ckv + (int64_t)usable * wid, (size_t)ls->nbuf[slot] * wid * sizeof(float));
        memcpy(ls->buf_gate[slot], cgt + (int64_t)usable * wid, (size_t)ls->nbuf[slot] * wid * sizeof(float));
    }
    if (nwin == 0) { free(ckv); free(cgt); return 0; }

    if (ls->ncomp[slot] + nwin > ls->cap[slot]) {
        int cap = ls->cap[slot] ? ls->cap[slot] : 64;
        while (cap < ls->ncomp[slot] + nwin) cap *= 2;
        ls->comp[slot] = realloc(ls->comp[slot], (size_t)cap * dim * sizeof(float));
        if (!ls->comp[slot]) { fprintf(stderr, "OOM compressed KV\n"); exit(1); }
        ls->cap[slot] = cap;
    }

    int slots = overlap ? 2 * m_r : m_r;
    float *sk = falloc((int64_t)slots * dim), *sg = falloc((int64_t)slots * dim);
    float *col = falloc(slots);
    for (int wnd = 0; wnd < nwin; wnd++) {
        if (!overlap) {
            for (int j = 0; j < m_r; j++) {
                const float *k0 = ckv + (int64_t)(wnd * m_r + j) * wid;
                const float *g0 = cgt + (int64_t)(wnd * m_r + j) * wid;
                for (int d = 0; d < dim; d++) {
                    sk[j * dim + d] = k0[d];
                    sg[j * dim + d] = g0[d] + pos_bias[j * wid + d];
                }
            }
        } else {
            /* second half: this window's Cb slice; first half: previous window's Ca slice
             * (from this call when wnd > 0, else from the saved cross-call overlap). */
            for (int j = 0; j < m_r; j++) {
                const float *k0 = ckv + (int64_t)(wnd * m_r + j) * wid;
                const float *g0 = cgt + (int64_t)(wnd * m_r + j) * wid;
                for (int d = 0; d < dim; d++) {
                    sk[(m_r + j) * dim + d] = k0[dim + d];
                    sg[(m_r + j) * dim + d] = g0[dim + d] + pos_bias[j * wid + dim + d];
                }
            }
            for (int j = 0; j < m_r; j++) {
                if (wnd > 0) {
                    const float *k0 = ckv + (int64_t)((wnd - 1) * m_r + j) * wid;
                    const float *g0 = cgt + (int64_t)((wnd - 1) * m_r + j) * wid;
                    for (int d = 0; d < dim; d++) {
                        sk[j * dim + d] = k0[d];
                        sg[j * dim + d] = g0[d] + pos_bias[j * wid + d];
                    }
                } else if (ls->has_ov[slot]) {
                    memcpy(sk + (int64_t)j * dim, ls->ov_kv[slot] + (int64_t)j * dim, (size_t)dim * sizeof(float));
                    memcpy(sg + (int64_t)j * dim, ls->ov_gate[slot] + (int64_t)j * dim, (size_t)dim * sizeof(float));
                } else {
                    for (int d = 0; d < dim; d++) { sk[j * dim + d] = 0.f; sg[j * dim + d] = -INFINITY; }
                }
            }
        }
        /* per-CHANNEL softmax across the window slots, then the gated sum */
        float *dst = ls->comp[slot] + (int64_t)(ls->ncomp[slot] + wnd) * dim;
        for (int d = 0; d < dim; d++) {
            for (int j = 0; j < slots; j++) col[j] = sg[j * dim + d];
            softmax_row(col, slots);
            float a = 0; for (int j = 0; j < slots; j++) a += col[j] * sk[j * dim + d];
            dst[d] = a;
        }
        rmsnorm_row(dst, dst, norm_w, dim, c->eps);
        rope_head(dst, dim, c->rdim, c->inv_comp, wnd * m_r + first_pos, 1.f, c->ms_comp);
    }
    /* persist this call's last full window's Ca slice for the next call's window 0 */
    if (overlap) {
        for (int j = 0; j < m_r; j++) {
            const float *k0 = ckv + (int64_t)((nwin - 1) * m_r + j) * wid;
            const float *g0 = cgt + (int64_t)((nwin - 1) * m_r + j) * wid;
            for (int d = 0; d < dim; d++) {
                ls->ov_kv[slot][j * dim + d] = k0[d];
                ls->ov_gate[slot][j * dim + d] = g0[d] + pos_bias[j * wid + d];
            }
        }
        ls->has_ov[slot] = 1;
    }
    ls->ncomp[slot] += nwin;
    ls->ecount[slot] += nwin;
    free(sk); free(sg); free(col); free(ckv); free(cgt);
    return nwin;
}

/* ---------- attention ----------
 * xn[n][D] is the already-normalized block input; writes out[n][D]. */
static void attn_forward(Model *m, Layer *L, State *s, int li, const float *xn, int n,
                         int pos0, float *out) {
    Cfg *c = &m->c;
    LState *ls = &s->l[li];
    int D = c->D, H = c->n_heads, hd = c->head_dim, R = c->rdim;
    const float *inv = c->att[li] == ATT_SLIDING ? c->inv_main : c->inv_comp;
    float ms = c->att[li] == ATT_SLIDING ? c->ms_main : c->ms_comp;

    /* queries: q_a -> q_a_norm -> q_b -> per-head unweighted RMSNorm -> rope */
    float *qres = falloc((int64_t)n * c->q_lora);
    float *qtmp = falloc((int64_t)n * c->q_lora);
    mmw(qtmp, xn, L->q_a, n, D, c->q_lora);
    for (int t = 0; t < n; t++)
        rmsnorm_row(qres + (int64_t)t * c->q_lora, qtmp + (int64_t)t * c->q_lora, L->q_an, c->q_lora, c->eps);
    free(qtmp);
    float *q = falloc((int64_t)n * H * hd);
    mmw(q, qres, L->q_b, n, c->q_lora, H * hd);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < H; h++) {
            float *qh = q + ((int64_t)t * H + h) * hd;
            rmsnorm_plain(qh, qh, hd, c->eps);
            rope_head(qh, hd, R, inv, pos0 + t, 1.f, ms);
        }

    /* the single shared KV head: kv_proj -> kv_norm -> rope. K IS V. */
    float *knew = falloc((int64_t)n * hd);
    { float *raw = falloc((int64_t)n * hd);
      mmw(raw, xn, L->kv_w, n, D, hd);
      for (int t = 0; t < n; t++) {
          rmsnorm_row(knew + (int64_t)t * hd, raw + (int64_t)t * hd, L->kv_n, hd, c->eps);
          rope_head(knew + (int64_t)t * hd, hd, R, inv, pos0 + t, 1.f, ms);
      }
      free(raw); }

    /* sliding keys visible this call = cached tail ++ new rows (the reference's
     * DynamicSlidingWindowLayer.update returns exactly that concatenation) */
    int nc = ls->nkv, nk = nc + n;
    float *kall = falloc((int64_t)nk * hd);
    int *kpos = malloc(sizeof(int) * nk);
    if (nc) {
        memcpy(kall, ls->kv, (size_t)nc * hd * sizeof(float));
        memcpy(kpos, ls->kpos, sizeof(int) * nc);
    }
    memcpy(kall + (int64_t)nc * hd, knew, (size_t)n * hd * sizeof(float));
    for (int t = 0; t < n; t++) kpos[nc + t] = pos0 + t;

    /* compressed long-range entries, plus per-query visibility */
    int ncomp = 0, m_r = 0;
    const float *comp = NULL;
    char *vis = NULL;                      /* [n][ncomp], 1 = attend */
    if (c->att[li] != ATT_SLIDING) {
        m_r = c->att[li] == ATT_CSA ? c->m_csa : c->m_hca;
        int wid = c->att[li] == ATT_CSA ? 2 * hd : hd;
        float *ckv = falloc((int64_t)n * wid), *cgt = falloc((int64_t)n * wid);
        mmw(ckv, xn, L->c_kv, n, D, wid);
        mmw(cgt, xn, L->c_gate, n, D, wid);
        compress_push(c, ls, 0, ckv, cgt, n, m_r, hd, c->att[li] == ATT_CSA, L->c_bias, L->c_norm);
        free(ckv); free(cgt);
        ncomp = ls->ncomp[0];
        comp = ls->comp[0];
        vis = calloc((size_t)n * (ncomp > 0 ? ncomp : 1), 1);

        if (c->att[li] == ATT_HCA) {
            /* every entry whose window closed strictly before this query's position */
            for (int t = 0; t < n; t++) {
                int thr = (pos0 + t + 1) / m_r;
                for (int e = 0; e < ncomp && e < thr; e++) vis[(int64_t)t * ncomp + e] = 1;
            }
        } else {
            /* Lightning Indexer. Its own compressor must consume every token even when no
             * window has closed yet — its buffer is what keeps the two window schedules
             * aligned across calls, so this cannot be skipped when ncomp == 0. */
            int idim = c->idx_dim, IH = c->idx_heads;
            float *ikv = falloc((int64_t)n * 2 * idim), *igt = falloc((int64_t)n * 2 * idim);
            mmw(ikv, xn, L->i_kv, n, D, 2 * idim);
            mmw(igt, xn, L->i_gate, n, D, 2 * idim);
            compress_push(c, ls, 1, ikv, igt, n, m_r, idim, 1, L->i_bias, L->i_norm);
            free(ikv); free(igt);
            int nic = ls->ncomp[1];
            const float *ic = ls->comp[1];
            if (ncomp > 0) {
            float *iq = falloc((int64_t)n * IH * idim);
            mmw(iq, qres, L->i_qb, n, c->q_lora, IH * idim);
            float *iw = falloc((int64_t)n * IH);
            mmw(iw, xn, L->i_w, n, D, IH);
            float wsc = 1.f / sqrtf((float)IH), ssc = 1.f / sqrtf((float)idim);

            int cand = nic < ncomp ? nic : ncomp;
            float *sc = falloc(cand > 0 ? cand : 1);
            for (int t = 0; t < n; t++) {
                for (int h = 0; h < IH; h++)
                    rope_head(iq + ((int64_t)t * IH + h) * idim, idim, R, c->inv_comp, pos0 + t, 1.f, c->ms_comp);
                int thr = (pos0 + t + 1) / m_r;
                for (int e = 0; e < cand; e++) {
                    float a = 0;
                    for (int h = 0; h < IH; h++) {
                        const float *qh = iq + ((int64_t)t * IH + h) * idim;
                        float dot = 0;
                        for (int d = 0; d < idim; d++) dot += qh[d] * ic[(int64_t)e * idim + d];
                        if (dot > 0) a += (iw[(int64_t)t * IH + h] * wsc) * (dot * ssc);
                    }
                    sc[e] = e < thr ? a : -INFINITY;
                }
                /* top-k by score. Ties go to the LOWEST entry index (strict >, stable
                 * scan). torch.topk defines no tie-break — it was observed returning the
                 * highest tied index — so an exactly-tied row is unmatchable by
                 * construction, not a bug on either side. It only arises when relu()
                 * zeroes the score across EVERY indexer head at once: a 2^-64 event at
                 * V4-Flash's 64 heads, but near-certain in a 2-head toy, which is why
                 * make_tiny_deepseek.py uses 8. */
                int kk = c->idx_topk < cand ? c->idx_topk : cand;
                for (int a = 0; a < kk; a++) {
                    int best = -1; float bv = -INFINITY;
                    for (int e = 0; e < cand; e++) {
                        if (vis[(int64_t)t * ncomp + e]) continue;
                        if (best < 0 || sc[e] > bv) { bv = sc[e]; best = e; }
                    }
                    if (best < 0 || best >= thr) break;      /* -1 sentinel: nothing valid left */
                    vis[(int64_t)t * ncomp + best] = 1;
                }
            }
            free(sc); free(iq); free(iw);
            }
        }
    }

    /* per-query softmax attention over [sliding window ++ selected compressed], with the
     * per-head learnable sink as an extra logit that is dropped after normalization */
    float scale = 1.f / sqrtf((float)hd);
    int gin = H * hd / c->o_groups;
    float *acc = falloc((int64_t)H * hd);
    float *lg = falloc(nk + (ncomp > 0 ? ncomp : 1));
    int *src = malloc(sizeof(int) * (nk + (ncomp > 0 ? ncomp : 1)));
    float *grp = falloc((int64_t)c->o_groups * c->o_rank);
    for (int t = 0; t < n; t++) {
        int p = pos0 + t;
        for (int h = 0; h < H; h++) {
            const float *qh = q + ((int64_t)t * H + h) * hd;
            int nl = 0;
            float mx = L->sinks[h];
            for (int j = 0; j < nk; j++) {
                if (kpos[j] > p || p - kpos[j] >= c->win) continue;
                float d = 0;
                for (int i = 0; i < hd; i++) d += qh[i] * kall[(int64_t)j * hd + i];
                d *= scale;
                lg[nl] = d; if (d > mx) mx = d;
                src[nl] = j;
                nl++;
            }
            int nsl = nl;
            for (int e = 0; e < ncomp; e++) {
                if (!vis[(int64_t)t * ncomp + e]) continue;
                float d = 0;
                for (int i = 0; i < hd; i++) d += qh[i] * comp[(int64_t)e * hd + i];
                d *= scale;
                lg[nl] = d; if (d > mx) mx = d;
                src[nl] = e;
                nl++;
            }
            double den = exp((double)L->sinks[h] - mx);
            for (int j = 0; j < nl; j++) { lg[j] = expf(lg[j] - mx); den += lg[j]; }
            float rd = (float)(1.0 / den);
            float *ah = acc + (int64_t)h * hd;
            for (int i = 0; i < hd; i++) ah[i] = 0;
            for (int j = 0; j < nl; j++) {
                float w = lg[j] * rd;
                const float *v = j < nsl ? kall + (int64_t)src[j] * hd
                                         : comp + (int64_t)src[j] * hd;
                for (int i = 0; i < hd; i++) ah[i] += w * v[i];
            }
            /* K == V, so the value carried RoPE: undo it at the QUERY position */
            rope_head(ah, hd, R, inv, p, -1.f, ms);
        }
        /* grouped low-rank output projection: g blocks of gin -> o_rank, then mixed to D */
        for (int g = 0; g < c->o_groups; g++)
        {   Wt oa = L->o_a;
            int64_t off = (int64_t)g * c->o_rank * gin;
            Wt g_oa = { oa.f ? oa.f + off : NULL, oa.b ? oa.b + off : NULL };
            mmw(grp + (int64_t)g * c->o_rank, acc + (int64_t)g * gin, g_oa, 1, gin, c->o_rank); }
        mmw(out + (int64_t)t * D, grp, L->o_b, 1, c->o_groups * c->o_rank, D);
    }

    /* keep the last win-1 sliding entries */
    int keep = c->win - 1;
    if (keep > nk) keep = nk;
    if (keep > 0) {
        memmove(ls->kv, kall + (int64_t)(nk - keep) * hd, (size_t)keep * hd * sizeof(float));
        memmove(ls->kpos, kpos + (nk - keep), sizeof(int) * keep);
    }
    ls->nkv = keep > 0 ? keep : 0;

    free(qres); free(q); free(knew); free(kall); free(kpos);
    free(acc); free(lg); free(src); free(grp); free(vis);
}

/* ---------- forward ---------- */
static void forward(Model *m, State *s, const int *ids, int n, int pos0, float *logits,
                    int logits_all) {
    Cfg *c = &m->c;
    int D = c->D, V = c->V, hc = c->hc_mult;

    float *hs = falloc((int64_t)n * hc * D);            /* the hc parallel residual streams */
    float *erow = falloc(D);
    for (int t = 0; t < n; t++) {
        const float *e = wt_row_f32(m->embed, (int64_t)ids[t] * D, erow, D);
        for (int k = 0; k < hc; k++)
            memcpy(hs + ((int64_t)t * hc + k) * D, e, (size_t)D * sizeof(float));
    }
    free(erow);

    int mixn = (2 + hc) * hc;
    float *coll = falloc((int64_t)n * D), *xn = falloc((int64_t)n * D);
    float *sub = falloc((int64_t)n * D);
    float *post = falloc((int64_t)n * hc), *comb = falloc((int64_t)n * hc * hc);
    float *flat = falloc((int64_t)hc * D), *mix = falloc(mixn);
    float *tmp = falloc((int64_t)hc * D);
    int64_t scr = (int64_t)2 * c->E + c->K + 2 * D +
                  2 * (c->moe_I > c->shared_I ? c->moe_I : c->shared_I) + 16;
    float *scratch = falloc(scr);

    for (int li = 0; li < c->n_layers; li++) {
        Layer *L = &m->L[li];
        /* --- attention site --- */
        for (int t = 0; t < n; t++)
            hc_site(c, L->a_fn, L->a_base, L->a_scale, hs + (int64_t)t * hc * D,
                    post + (int64_t)t * hc, comb + (int64_t)t * hc * hc,
                    coll + (int64_t)t * D, flat, mix);
        for (int t = 0; t < n; t++)
            rmsnorm_row(xn + (int64_t)t * D, coll + (int64_t)t * D, L->ln1, D, c->eps);
        attn_forward(m, L, s, li, xn, n, pos0, sub);
        for (int t = 0; t < n; t++)
            hc_merge(c, hs + (int64_t)t * hc * D, post + (int64_t)t * hc,
                     comb + (int64_t)t * hc * hc, sub + (int64_t)t * D, tmp);

        /* --- ffn site --- */
        for (int t = 0; t < n; t++)
            hc_site(c, L->f_fn, L->f_base, L->f_scale, hs + (int64_t)t * hc * D,
                    post + (int64_t)t * hc, comb + (int64_t)t * hc * hc,
                    coll + (int64_t)t * D, flat, mix);
        for (int t = 0; t < n; t++)
            rmsnorm_row(xn + (int64_t)t * D, coll + (int64_t)t * D, L->ln2, D, c->eps);
        for (int t = 0; t < n; t++)
            moe_row(m, L, li, xn + (int64_t)t * D, ids[t], sub + (int64_t)t * D, scratch);
        for (int t = 0; t < n; t++)
            hc_merge(c, hs + (int64_t)t * hc * D, post + (int64_t)t * hc,
                     comb + (int64_t)t * hc * hc, sub + (int64_t)t * D, tmp);
    }
    s->len = pos0 + n;

    /* hc_head: final collapse of the hc streams, then the shared norm and lm_head */
    int t0 = logits_all ? 0 : n - 1;
    float *hv = falloc(D);
    for (int t = t0; t < n; t++) {
        const float *st = hs + (int64_t)t * hc * D;
        rmsnorm_plain(flat, st, hc * D, c->eps);
        mm(mix, flat, m->h_fn, 1, hc * D, hc);
        for (int i = 0; i < D; i++) hv[i] = 0;
        for (int k = 0; k < hc; k++) {
            float pre = 1.f / (1.f + expf(-(mix[k] * m->h_scale[0] + m->h_base[k]))) + c->hc_eps;
            for (int i = 0; i < D; i++) hv[i] += pre * st[(int64_t)k * D + i];
        }
        rmsnorm_row(hv, hv, m->norm, D, c->eps);
        mmw(logits + (int64_t)(logits_all ? t : 0) * V, hv, m->lm_head, 1, D, V);
    }
    free(hs); free(coll); free(xn); free(sub); free(post); free(comb);
    free(flat); free(mix); free(tmp); free(scratch); free(hv);
}

/* ---------- oracle ---------- */
static char *read_file(const char *p, long *len) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n + 1);
    if (fread(b, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p); exit(1); }
    b[n] = 0; fclose(f); *len = n; return b;
}

static int run_oracle(Model *m, const char *ref_path) {
    long rn;
    char *rbuf = read_file(ref_path, &rn);
    char *arena = NULL;
    jval *r = json_parse(rbuf, &arena);
    jval *pj = json_get(r, "prompt_ids"), *fj = json_get(r, "full_ids"), *tj = json_get(r, "tf_pred");
    if (!pj || !fj || !tj) { fprintf(stderr, "ref json missing keys\n"); return 1; }
    int np = pj->len, nf = fj->len;
    int *prompt = calloc(np > 0 ? np : 1, sizeof(int)), *full = calloc(nf > 0 ? nf : 1, sizeof(int));
    int *tf = calloc(tj->len > 0 ? tj->len : 1, sizeof(int));
    for (int i = 0; i < np; i++) prompt[i] = (int)pj->kids[i]->num;
    for (int i = 0; i < nf; i++) full[i] = (int)fj->kids[i]->num;
    for (int i = 0; i < tj->len; i++) tf[i] = (int)tj->kids[i]->num;
    double ppl_ref = jnum(r, "ppl_ref", 0);
    int V = m->c.V;

    /* 1. teacher forcing in one pass: argmax everywhere, plus perplexity */
    State s; state_init(m, &s);
    float *lg = falloc((int64_t)nf * V);
    forward(m, &s, full, nf, 0, lg, 1);
    int tf_ok = 0; double nll = 0;
    for (int t = 0; t < nf; t++) {
        const float *l = lg + (int64_t)t * V;
        int arg = 0;
        for (int i = 1; i < V; i++) if (l[i] > l[arg]) arg = i;
        if (t < tj->len && arg == tf[t]) tf_ok++;
        else if (getenv("DS_TF") && t < tj->len)
            fprintf(stderr, "  tf mismatch at pos %d: ref %d got %d\n", t, tf[t], arg);
        if (t + 1 < nf) {
            float mx = l[0];
            for (int i = 1; i < V; i++) if (l[i] > mx) mx = l[i];
            double sum = 0;
            for (int i = 0; i < V; i++) sum += exp((double)l[i] - mx);
            nll -= ((double)l[full[t + 1]] - mx) - log(sum);
        }
    }
    double ppl = exp(nll / (nf - 1));

    /* Optional numeric gate. The transformers oracle ships argmaxes only; the pure-Python
     * reference (tools/dsv4_ref.py) also ships the final logit row, which catches drift
     * that argmax rounds away. */
    int num_ok = 1; double maxdiff = 0;
    jval *lj = json_get(r, "logits_last");
    if (lj && lj->t == J_ARR && lj->len == V) {
        const float *l = lg + (int64_t)(nf - 1) * V;
        double scale = 0;
        for (int i = 0; i < V; i++) { double a = fabs(lj->kids[i]->num); if (a > scale) scale = a; }
        if (scale < 1e-6) scale = 1e-6;
        for (int i = 0; i < V; i++) {
            double d = fabs((double)l[i] - lj->kids[i]->num);
            if (d > maxdiff) maxdiff = d;
        }
        num_ok = maxdiff / scale < 1e-3;
        printf("logits: max|diff| %.3e over %d (rel %.2e vs ref scale %.3f) — %s\n",
               maxdiff, V, maxdiff / scale, scale, num_ok ? "ok" : "TOO LARGE");
    }
    state_free(m, &s); free(lg);

    /* 2. incremental greedy generation: the compressor buffers, the sliding ring and the
     * indexer state all have to agree with the one-shot pass */
    State s2; state_init(m, &s2);
    float *l1 = falloc(V);
    int gen_ok = 0, ngen = nf - np;
    forward(m, &s2, prompt, np, 0, l1, 0);
    int *got = malloc(sizeof(int) * (ngen > 0 ? ngen : 1));
    for (int i = 0; i < ngen; i++) {
        int arg = 0;
        for (int v = 1; v < V; v++) if (l1[v] > l1[arg]) arg = v;
        got[i] = arg;
        if (arg == full[np + i]) gen_ok++;
        if (i + 1 < ngen) forward(m, &s2, &arg, 1, np + i, l1, 0);
    }
    state_free(m, &s2);

    /* 3. split prefill: the same prompt in two chunks must land on the same logits as one
     * shot. Nothing upstream covers this — it is the only check on the compressor buffers
     * and the CSA overlap slice being carried correctly across a call boundary with n > 1,
     * which is exactly the state the window scheme is easiest to get wrong in. */
    int split_ok = 1; double split_diff = 0;
    if (np >= 4) {
        int cut = np / 2;
        State a, b;
        state_init(m, &a); state_init(m, &b);
        float *la = falloc(V), *lb = falloc(V);
        forward(m, &a, prompt, np, 0, la, 0);
        forward(m, &b, prompt, cut, 0, lb, 0);
        forward(m, &b, prompt + cut, np - cut, cut, lb, 0);
        double sc = 0;
        for (int i = 0; i < V; i++) { double v = fabs(la[i]); if (v > sc) sc = v; }
        if (sc < 1e-6) sc = 1e-6;
        for (int i = 0; i < V; i++) {
            double d = fabs((double)la[i] - lb[i]);
            if (d > split_diff) split_diff = d;
        }
        split_ok = split_diff / sc < 1e-4;
        printf("split prefill (%d+%d): max|diff| %.3e — %s\n",
               cut, np - cut, split_diff, split_ok ? "ok" : "DIVERGED");
        state_free(m, &a); state_free(m, &b); free(la); free(lb);
    }

    printf("teacher forcing: %d/%d   greedy: %d/%d   ppl %.4f (ref %.4f, %+.2f%%)\n",
           tf_ok, (int)tj->len, gen_ok, ngen, ppl, ppl_ref,
           ppl_ref > 0 ? (ppl - ppl_ref) / ppl_ref * 100 : 0);
    if (gen_ok != ngen) {
        printf("  ref: "); for (int i = 0; i < ngen; i++) printf("%d ", full[np + i]);
        printf("\n  got: "); for (int i = 0; i < ngen; i++) printf("%d ", got[i]);
        printf("\n");
    }
    int ok = (tf_ok == (int)tj->len) && (gen_ok == ngen) && num_ok && split_ok;
    printf("%s\n", ok ? "ORACLE PASS" : "ORACLE FAIL");
    free(rbuf); free(arena); free(prompt); free(full); free(tf); free(l1); free(got);
    return !ok;
}

/* ---------- generation ---------- */
static int argmax_logits(const float *l, int n) {
    int a = 0; for (int i = 1; i < n; i++) if (l[i] > l[a]) a = i; return a;
}

static void generate(Model *m, Tok *T, const char *text, int n_gen, float temp, float top_p) {
    Cfg *c = &m->c;
    int cap = (int)strlen(text) + 16;
    int *ids = malloc(sizeof(int) * cap);
    int np = tok_encode(T, text, (int)strlen(text), ids, cap);
    if (np <= 0) { fprintf(stderr, "empty prompt\n"); free(ids); return; }

    State s; state_init(m, &s);
    float *lg = falloc(c->V);
    double t0 = now_s();
    forward(m, &s, ids, np, 0, lg, 0);
    double pre = now_s() - t0;
    fprintf(stderr, "[deepseek] prefill %d tok in %.2fs (%.1f tok/s)\n", np, pre, np / pre);

    double td = now_s();
    int ngen = 0;
    for (int i = 0; i < n_gen; i++) {
        int tk = temp > 0 ? sample_logits(lg, c->V, temp, top_p) : argmax_logits(lg, c->V);
        int stop = 0;
        for (int e = 0; e < c->n_eos; e++) if (tk == c->eos[e]) stop = 1;
        if (stop) break;
        char buf[512];
        int bl = tok_decode(T, &tk, 1, buf, sizeof(buf) - 1);
        if (bl > 0) { fwrite(buf, 1, bl, stdout); fflush(stdout); }
        ngen++;
        forward(m, &s, &tk, 1, np + i, lg, 0);
    }
    double dec = now_s() - td;
    printf("\n");
    fprintf(stderr, "[deepseek] %d tok in %.2fs (%.2f tok/s), rss %.1f GB\n",
            ngen, dec, ngen > 0 ? ngen / dec : 0.0, rss_gb());
    state_free(m, &s); free(lg); free(ids);
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    if (getenv("EXACT")) g_exact = 1;

    const char *ref = NULL, *pfile = NULL, *prompt = NULL;
    int n_gen = 64;
    float temp = 0.f, top_p = 0.95f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-f") && i + 1 < argc) pfile = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (strstr(argv[i], ".json")) ref = argv[i];
    }

    Model m;
    memset(&m, 0, sizeof(m));
    double t0 = now_s();
    model_load(&m, snap);
    fprintf(stderr, "[deepseek] %d layers, %d experts (top-%d), hc_mult %d, "
            "loaded in %.1fs (rss %.1f GB)\n",
            m.c.n_layers, m.c.E, m.c.K, m.c.hc_mult, now_s() - t0, rss_gb());

    if (ref) { g_exact = 1; return run_oracle(&m, ref); }

    long tn = 0;
    char *txt = pfile ? read_file(pfile, &tn) : (prompt ? strdup(prompt) : NULL);
    if (!txt) { fprintf(stderr, "usage: SNAP=dir %s [-f prompt.txt | -p text] [-n tokens] "
                                "[-t temp] | <ref.json>\n", argv[0]); return 1; }
    char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
    Tok T; tok_load(&T, tkp);
    generate(&m, &T, txt, n_gen, temp, top_p);
    free(txt);
    return 0;
}
