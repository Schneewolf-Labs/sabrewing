/* glm53.c — GLM-5.3-Flash engine (zai-org/GLM-5.3-Flash, transformers `glm5_next`).
 *
 * 321B total / ~18B active, 45 layers, 288 routed experts top-8 + 1 shared. Validated
 * against transformers' Glm5NextForConditionalGeneration (tools/make_tiny_glm53.py).
 * The name says GLM, but almost nothing is shared with GLM-5.2 (colibri.c):
 *
 *   residual   mHC hyper-connections, as in DeepSeek-V4 (moe_hc.h): 4 streams per token,
 *              collapsed/expanded/mixed at both sublayer sites. The final collapse is a
 *              plain MEAN of the streams (V4 learns one).
 *   KDA        34 of 45 layers are Kimi Delta Attention, a linear attention with a
 *              per-CHANNEL decay: state S [k_dim x v_dim] per head, O(1) in context.
 *                q,k,v = SiLU(depthwise causal conv4(W x)); q,k L2-normalized (eps INSIDE
 *                        the sqrt), q *= k_dim^-0.5
 *                g     = lb * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias))  (lb = -5)
 *                beta  = sigmoid(b_proj x)
 *                S     = diag(exp(g)) S;  S += k (x) ((v - k^T S) * beta);  o = q^T S
 *                out   = o_proj( RMSNorm_head(o) * sigmoid(g_b(g_a(x))) )
 *   DSA        the other 11 are NoPE MLA (qk_rope 0: position lives in KDA only) behind a
 *              pooled Lightning Indexer. Keys are grouped into pools of index_kpool; a
 *              pool's key is a per-channel softmax mixture of its tokens' index keys
 *              (gate = compress_gate . x + ape[slot]). Each query scores the complete,
 *              causally visible pools, keeps the top index_topk/index_kpool, and attends
 *              to those pools' tokens plus the incomplete visible tail. indexer_types
 *              "shared" reuses the previous layer's selection.
 *   MoE        sigmoid router + selection-only correction bias, renormalized top-8 x 2.5,
 *              clamped SwiGLU (gate <= 10, |up| <= 10) on routed, shared AND dense MLPs.
 *
 * MLA is ABSORBED: the cache holds the kv_lora latent (512 floats/token) instead of
 * expanded keys and values. q is pulled into latent space through kv_b's key rows,
 * the attention-weighted latent is pushed out through its value rows. An identity, not
 * an approximation — only the order of the products changes.
 *
 * Validation:   SNAP=tiny_glm53 ./glm53 tiny_glm53/ref_glm53.json
 * Real model:   tools/convert_glm53_int4.py -> bf16 residents + int4 routed experts.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "compat.h"
#include "json.h"
#include "moe_math.h"
#include "moe_matmul.h"
#include "moe_quant.h"         /* matmul_q4_kb: int4 routed experts, per-row scales */
#include "moe_sample.h"
#include "moe_util.h"          /* falloc, jnum, jstr_at, moe_omp_autotune */
#include "moe_hc.h"            /* mHC, shared with deepseek.c */
#include "moe_linattn.h"       /* linattn_conv_span, linattn_l2norm */
#include "moe_serve.h"         /* SUBMIT/CANCEL protocol for the gateway */
#ifdef COLI_CUDA
#include "backend_cuda_laguna.h"   /* lag_cuda_matmul_bf16: arch-generic resident GEMM */
#endif
#include "st.h"
#include "tok.h"

#define MAXL 128

static int g_exact = 0;        /* double-accumulate GEMMs: the oracle contract */
static int g_prof = 0;
static double g_t_kda, g_t_dsa, g_t_moe, g_t_hc;

/* ---------- config ---------- */
typedef struct {
    int n_layers, D, V;
    /* MLA (DSA layers) */
    int n_heads, nope, vdim, q_lora, kv_lora;
    /* pooled lightning indexer */
    int idx_heads, idx_dim, idx_topk, kpool, tail;
    /* KDA (linear layers) */
    int la_heads, la_dim, conv_k, la_has_lb;
    float la_lb;
    /* FFN */
    int E, K, moe_I, shared_I, dense_I, n_group, topk_group, norm_topk;
    float route_scale, swiglu_limit;
    /* mHC */
    int hc_mult, hc_iters;
    float hc_eps, eps;
    int is_la[MAXL];               /* 1 = KDA, 0 = DSA */
    int is_moe[MAXL];              /* 1 = routed MoE, 0 = dense MLP */
    int idx_shared[MAXL];          /* 1 = reuse the previous layer's top-k */
    int ctx_max, tie_embed;
    int eos[8], n_eos;
} Cfg;

/* A projection as stored: f32 (oracle fixtures) or bf16 (the converted container,
 * kept bf16 in RAM — the dense set is ~18 GB bf16 vs ~36 GB f32). Activations stay f32. */
typedef struct { const float *f; const uint16_t *b; void *d; } Wt;

typedef struct {
    float *ln1, *ln2;
    float *a_fn, *a_base, *a_scale, *f_fn, *f_base, *f_scale;
    /* KDA */
    Wt q, k, v, o, fa, fb, bp, ga, gb;
    float *conv, *dt, *alog, *onorm;          /* conv [3*qkv][K] (q|k|v channel order) */
    /* DSA */
    Wt qa, qb, kva, kvb, ao;
    float *qa_n, *kva_n;
    Wt iqb, iwk, iw, igate;
    float *ik_w, *ik_b, *iape;                /* k LayerNorm (weight + bias), ape [kpool][idx_dim] */
    /* FFN */
    Wt mg, mu, md;                            /* dense MLP */
    Wt router;
    float *corr;
    float *e_gu, *e_dn;                       /* f32 experts [E,2I,D] (gate rows, then up), [E,D,I] */
    uint8_t *q_gu, *q_dn;                     /* int4 container [E*2I, D/2], [E*D, I/2] */
    float *s_gu, *s_dn;                       /* per-row scales [E*2I], [E*D] */
    Wt sg, su, sd;                            /* shared expert */
} Layer;

typedef struct {
    shards S;
    Cfg c;
    Layer L[MAXL];
    Wt embed, lm_head;
    float *norm;
    int xq;                                   /* 1 = int4 routed-expert container */
    char P[64];                               /* tensor-name prefix */
} Model;

/* Per-layer sequence state. KDA: the conv window and the recurrent state, constant size.
 * DSA: the absorbed-MLA latent, the indexer's keys and gate scores, and the keys of every
 * COMPLETE pool (a pool's key never changes once its last token has arrived). */
typedef struct {
    float *win, *S;
    float *lat, *ik, *ig, *pool;
    int npool;
} LState;
typedef struct { LState *l; int len, cap; } State;

static double get_num(jval *o, const char *k, double d) { return jnum(o, k, d); }

static void cfg_load(Cfg *c, const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    buf[n] = 0; fclose(f);
    jval *root = json_parse_checked(buf);
    if (!root || root->t != J_OBJ) { fprintf(stderr, "%s: not a JSON object\n", path); exit(1); }
    jval *o = json_get(root, "text_config");          /* the multimodal wrapper nests it */
    if (!o || o->t != J_OBJ) o = root;

    memset(c, 0, sizeof(*c));
    c->n_layers = (int)get_num(o, "num_hidden_layers", 0);
    c->D = (int)get_num(o, "hidden_size", 0);
    c->V = (int)get_num(o, "vocab_size", 0);
    c->n_heads = (int)get_num(o, "num_attention_heads", 0);
    c->nope = (int)get_num(o, "qk_nope_head_dim", 0);
    c->vdim = (int)get_num(o, "v_head_dim", 0);
    c->q_lora = (int)get_num(o, "q_lora_rank", 0);
    c->kv_lora = (int)get_num(o, "kv_lora_rank", 0);
    if ((int)get_num(o, "qk_rope_head_dim", 0) != 0) {
        fprintf(stderr, "qk_rope_head_dim != 0: this engine implements GLM-5.3's NoPE MLA\n"); exit(1); }
    c->idx_heads = (int)get_num(o, "index_n_heads", 32);
    c->idx_dim = (int)get_num(o, "index_head_dim", 128);
    c->idx_topk = (int)get_num(o, "index_topk", 2048);
    c->kpool = (int)get_num(o, "index_kpool", 16);
    jval *tl = json_get(o, "index_kpool_always_select_tail");
    c->tail = !(tl && tl->t == J_BOOL && !tl->boolean);
    if (c->kpool < 1 || c->idx_topk % c->kpool) {
        fprintf(stderr, "index_topk %d must be a positive multiple of index_kpool %d\n", c->idx_topk, c->kpool); exit(1); }

    /* KDA: top-level keys, overridden by linear_attn_config like the HF config does */
    c->la_heads = (int)get_num(o, "linear_num_heads", 64);
    c->la_dim = (int)get_num(o, "linear_head_dim", 128);
    c->conv_k = (int)get_num(o, "linear_conv_kernel_dim", 4);
    jval *lb = json_get(o, "linear_lower_bound");
    c->la_has_lb = !(lb && lb->t == J_NULL);
    c->la_lb = (lb && lb->t == J_NUM) ? (float)lb->num : -5.f;
    jval *la = json_get(o, "linear_attn_config");
    if (la && la->t == J_OBJ) {
        c->la_dim = (int)get_num(la, "head_dim", c->la_dim);
        c->la_heads = (int)get_num(la, "num_heads", c->la_heads);
        c->conv_k = (int)get_num(la, "short_conv_kernel_size", c->conv_k);
        jval *g = json_get(la, "gate_lower_bound");
        if (g && g->t == J_NUM) { c->la_lb = (float)g->num; c->la_has_lb = 1; }
        jval *sg = json_get(la, "safe_gate");
        if (!c->la_has_lb && !(sg && sg->t == J_BOOL && !sg->boolean)) { c->la_lb = -5.f; c->la_has_lb = 1; }
    }

    c->E = (int)get_num(o, "n_routed_experts", 0);
    c->K = (int)get_num(o, "num_experts_per_tok", 0);
    c->moe_I = (int)get_num(o, "moe_intermediate_size", 0);
    c->shared_I = c->moe_I * (int)get_num(o, "n_shared_experts", 1);
    c->dense_I = (int)get_num(o, "intermediate_size", 0);
    c->n_group = (int)get_num(o, "n_group", 1);
    c->topk_group = (int)get_num(o, "topk_group", 1);
    jval *nt = json_get(o, "norm_topk_prob");
    c->norm_topk = !(nt && nt->t == J_BOOL && !nt->boolean);
    c->route_scale = (float)get_num(o, "routed_scaling_factor", 1.0);
    c->swiglu_limit = (float)get_num(o, "swiglu_limit", 10.0);
    c->hc_mult = (int)get_num(o, "hc_mult", 4);
    c->hc_iters = (int)get_num(o, "hc_sinkhorn_iters", 20);
    c->hc_eps = (float)get_num(o, "hc_eps", 1e-6);
    c->eps = (float)get_num(o, "rms_norm_eps", 1e-5);
    jval *te = json_get(o, "tie_word_embeddings");
    if (!te) te = json_get(root, "tie_word_embeddings");
    c->tie_embed = te && te->t == J_BOOL && te->boolean;
    c->ctx_max = getenv("CTX_MAX") ? atoi(getenv("CTX_MAX")) : 8192;

    if (c->n_layers < 1 || c->n_layers > MAXL || c->D < 1 || c->V < 1 || c->n_heads < 1 ||
        c->E < 1 || c->K < 1 || c->K > c->E || c->K > 64 || c->hc_mult < 1 || c->hc_mult > MOE_HC_MAX ||
        c->la_heads < 1 || c->la_dim < 1 || c->la_dim > 512 || c->conv_k < 1 || c->kv_lora < 1 ||
        c->n_group < 1 || c->n_group > 64 || c->E % c->n_group || c->topk_group < 1 ||
        c->topk_group > c->n_group || c->E > 1024 || c->kpool > 256) {
        fprintf(stderr, "config.json: unsupported or missing dimensions\n"); exit(1); }

    /* layer schedule, mirroring Glm5NextTextConfig.__post_init__ */
    jval *lt = json_get(o, "layer_types");
    for (int i = 0; i < c->n_layers; i++) {
        if (lt && lt->t == J_ARR) {
            if (lt->len != c->n_layers) { fprintf(stderr, "layer_types has %d entries, want %d\n", lt->len, c->n_layers); exit(1); }
            c->is_la[i] = !strcmp(jstr_at(lt, i), "linear_attention");
        } else c->is_la[i] = (i % 4) != 3;
    }
    jval *mt = json_get(o, "mlp_layer_types");
    for (int i = 0; i < c->n_layers; i++) {
        if (mt && mt->t == J_ARR) {
            if (mt->len != c->n_layers) { fprintf(stderr, "mlp_layer_types length mismatch\n"); exit(1); }
            c->is_moe[i] = !strcmp(jstr_at(mt, i), "sparse");
        } else c->is_moe[i] = i >= 3;
    }
    jval *it = json_get(o, "indexer_types"), *pat = json_get(o, "index_topk_pattern");
    int freq = (int)get_num(o, "index_topk_freq", 1), off = (int)get_num(o, "index_skip_topk_offset", 2);
    if (freq < 1) freq = 1;
    for (int i = 0; i < c->n_layers; i++) {
        if (it && it->t == J_ARR && it->len == c->n_layers) c->idx_shared[i] = !strcmp(jstr_at(it, i), "shared");
        else if (pat && pat->t == J_STR && (int)strlen(pat->str) == c->n_layers) c->idx_shared[i] = pat->str[i] == 'S';
        else c->idx_shared[i] = ((i - off + 1 > 0 ? i - off + 1 : 0) % freq) != 0;
    }

    /* eos: generation_config.json first, then the config */
    c->n_eos = 0;
    for (int pass = 0; pass < 2 && !c->n_eos; pass++) {
        jval *src = pass ? o : NULL, *groot = NULL;
        char *gbuf = NULL;
        if (!pass) {
            char gp[1024]; snprintf(gp, sizeof(gp), "%s/generation_config.json", dir);
            FILE *gf = fopen(gp, "rb");
            if (gf) {
                fseek(gf, 0, SEEK_END); long gn = ftell(gf); fseek(gf, 0, SEEK_SET);
                gbuf = malloc(gn + 1);
                if (fread(gbuf, 1, gn, gf) == (size_t)gn) { gbuf[gn] = 0; src = groot = json_parse_checked(gbuf); }
                fclose(gf);
            }
        }
        jval *e = src ? json_get(src, "eos_token_id") : NULL;
        if (e && e->t == J_NUM) c->eos[c->n_eos++] = (int)e->num;
        else if (e && e->t == J_ARR)
            for (int i = 0; i < e->len && c->n_eos < 8; i++)
                if (e->kids[i]->t == J_NUM) c->eos[c->n_eos++] = (int)e->kids[i]->num;
        json_free(groot); free(gbuf);
    }
    json_free(root); free(buf);
}

/* ---------- weight loading ---------- */
static float *fzalloc(int64_t n) {
    float *p = calloc((size_t)(n > 0 ? n : 1), sizeof(float));
    if (!p) { fprintf(stderr, "OOM %lld floats\n", (long long)n); exit(1); }
    return p;
}

/* The checkpoint nests the text model under the multimodal wrapper
 * (`model.language_model.`); a text-only export does not. Probed once at load. */
static const char *tn(Model *m, char *buf, size_t cap, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int k = snprintf(buf, cap, "%s", m->P);
    vsnprintf(buf + k, cap - k, fmt, ap);
    va_end(ap);
    return buf;
}
static int has(Model *m, const char *name) { return st_find(&m->S, name) != NULL; }

static float *load_f(Model *m, const char *name, int64_t want) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    if (want >= 0 && n != want) {
        fprintf(stderr, "tensor %s has %lld elements, expected %lld\n", name, (long long)n, (long long)want); exit(1); }
    float *p = falloc(n); st_read_f32(&m->S, name, p, 0); return p;
}

#ifdef COLI_CUDA
/* bf16 residents in VRAM, front to back until the budget (GLM53_CUDA_GB, default free
 * minus 2 GB) runs out. Host copies are kept, so a full card degrades to the CPU path. */
static size_t g_cuda_left, g_cuda_used;
static int g_cuda_on;
static void cuda_budget_init(void) {
    if (getenv("GLM53_NOCUDA") || lag_cuda_init(0) != 0) return;
    size_t freeb = lag_cuda_free_bytes();
    double gb = getenv("GLM53_CUDA_GB") ? atof(getenv("GLM53_CUDA_GB")) : 0.0;
    g_cuda_left = gb > 0 ? (size_t)(gb * 1e9) : (freeb > (size_t)2e9 ? freeb - (size_t)2e9 : 0);
    g_cuda_on = g_cuda_left > 0;
}
#endif

/* A projection [O, I]: bf16 stays bf16, anything else expands to f32. */
static Wt load_w(Model *m, const char *name, int64_t O, int64_t I) {
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    if (t->numel != O * I) {
        fprintf(stderr, "tensor %s has %lld elements, expected %lld x %lld\n",
                name, (long long)t->numel, (long long)O, (long long)I); exit(1); }
    Wt w = {NULL, NULL, NULL};
    if (t->dtype == 0) {
        uint16_t *p = malloc((size_t)t->numel * sizeof(uint16_t));
        if (!p) { fprintf(stderr, "OOM for %s\n", name); exit(1); }
        st_read_raw(&m->S, name, p, 1);
        w.b = p;
#ifdef COLI_CUDA
        size_t nb = (size_t)t->numel * sizeof(uint16_t);
        if (g_cuda_on && nb <= g_cuda_left && (w.d = lag_cuda_upload(p, nb))) { g_cuda_left -= nb; g_cuda_used += nb; }
#endif
    } else w.f = load_f(m, name, -1);
    return w;
}

/* y[S,O] = x[S,I] @ W^T for a resident projection */
static void mmw(float *y, const float *x, Wt w, int S, int I, int O) {
#ifdef COLI_CUDA
    if (w.d && !g_exact && lag_cuda_matmul_bf16(y, x, w.d, S, I, O) == 0) return;
#endif
    if (w.b) matmul_bf16_k(y, x, w.b, S, I, O, 0, g_exact);
    else     matmul_f32(y, x, w.f, S, I, O, g_exact);
}
static inline const float *wt_row(Wt w, int64_t off, float *tmp, int n) {
    if (w.f) return w.f + off;
    for (int i = 0; i < n; i++) tmp[i] = bf16_f32(w.b[off + i]);
    return tmp;
}

/* Per-layer name lookup: the original checkpoint layout first, then the names
 * transformers uses in memory (an alias, for exports written that way). */
static const char *pick(Model *m, char *buf, size_t cap, int i, const char *a, const char *b) {
    tn(m, buf, cap, "layers.%d.%s", i, a);
    if (has(m, buf) || !b) return buf;
    return tn(m, buf, cap, "layers.%d.%s", i, b);
}

static void load_experts(Model *m, Layer *L, int i) {
    Cfg *c = &m->c;
    int E = c->E, I = c->moe_I, D = c->D;
    char nm[256];
    tn(m, nm, sizeof(nm), "layers.%d.mlp.experts.gate_up_proj", i);
    st_tensor *t = st_find(&m->S, nm);
    if (t && t->dtype == 3) {                       /* int4 container (U8) */
        m->xq = 1;
        if (t->nbytes != (int64_t)E * 2 * I * (D / 2)) { fprintf(stderr, "%s: wrong size\n", nm); exit(1); }
        L->q_gu = malloc(t->nbytes); st_read_raw(&m->S, nm, L->q_gu, 1);
        tn(m, nm, sizeof(nm), "layers.%d.mlp.experts.gate_up_proj.qs", i); L->s_gu = load_f(m, nm, (int64_t)E * 2 * I);
        tn(m, nm, sizeof(nm), "layers.%d.mlp.experts.down_proj", i);
        st_tensor *td = st_find(&m->S, nm);
        if (!td || td->dtype != 3 || td->nbytes != (int64_t)E * D * (I / 2)) { fprintf(stderr, "%s: missing or wrong size\n", nm); exit(1); }
        L->q_dn = malloc(td->nbytes); st_read_raw(&m->S, nm, L->q_dn, 1);
        tn(m, nm, sizeof(nm), "layers.%d.mlp.experts.down_proj.qs", i); L->s_dn = load_f(m, nm, (int64_t)E * D);
        return;
    }
    if (t) {                                        /* fused f32/bf16 [E,2I,D] + [E,D,I] */
        L->e_gu = load_f(m, nm, (int64_t)E * 2 * I * D);
        tn(m, nm, sizeof(nm), "layers.%d.mlp.experts.down_proj", i);
        L->e_dn = load_f(m, nm, (int64_t)E * D * I);
        return;
    }
    /* per-expert originals: experts.{e}.{gate,up,down}_proj.weight */
    L->e_gu = falloc((int64_t)E * 2 * I * D);
    L->e_dn = falloc((int64_t)E * D * I);
    for (int e = 0; e < E; e++) {
        const char *parts[2] = {"gate_proj", "up_proj"};
        for (int p = 0; p < 2; p++) {
            tn(m, nm, sizeof(nm), "layers.%d.mlp.experts.%d.%s.weight", i, e, parts[p]);
            float *w = load_f(m, nm, (int64_t)I * D);
            memcpy(L->e_gu + ((int64_t)e * 2 * I + (int64_t)p * I) * D, w, (size_t)I * D * sizeof(float));
            free(w);
        }
        tn(m, nm, sizeof(nm), "layers.%d.mlp.experts.%d.down_proj.weight", i, e);
        float *w = load_f(m, nm, (int64_t)D * I);
        memcpy(L->e_dn + (int64_t)e * D * I, w, (size_t)D * I * sizeof(float));
        free(w);
    }
}

static void model_load(Model *m, const char *snap) {
    memset(m, 0, sizeof(*m));
    cfg_load(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    snprintf(m->P, sizeof(m->P), "model.language_model.");
    char nm[256];
    if (!has(m, tn(m, nm, sizeof(nm), "embed_tokens.weight"))) snprintf(m->P, sizeof(m->P), "model.");
    if (!has(m, tn(m, nm, sizeof(nm), "embed_tokens.weight"))) snprintf(m->P, sizeof(m->P), "%s", "");
#ifdef COLI_CUDA
    cuda_budget_init();
#endif
    int D = c->D, hc = c->hc_mult, mixn = (2 + hc) * hc;
    int qkv = c->la_heads * c->la_dim, H = c->n_heads, Id = c->idx_dim;
    m->embed = load_w(m, tn(m, nm, sizeof(nm), "embed_tokens.weight"), c->V, D);
    m->norm = load_f(m, tn(m, nm, sizeof(nm), "norm.weight"), D);
    if (has(m, "lm_head.weight")) m->lm_head = load_w(m, "lm_head.weight", c->V, D);
    else if (c->tie_embed) m->lm_head = m->embed;
    else { fprintf(stderr, "lm_head.weight missing on an untied model\n"); exit(1); }

    for (int i = 0; i < c->n_layers; i++) {
        Layer *L = &m->L[i];
        L->ln1 = load_f(m, tn(m, nm, sizeof(nm), "layers.%d.input_layernorm.weight", i), D);
        L->ln2 = load_f(m, tn(m, nm, sizeof(nm), "layers.%d.post_attention_layernorm.weight", i), D);
        L->a_fn = load_f(m, pick(m, nm, sizeof(nm), i, "hc_attn_fn", "attn_hc.fn"), (int64_t)mixn * hc * D);
        L->a_base = load_f(m, pick(m, nm, sizeof(nm), i, "hc_attn_base", "attn_hc.base"), mixn);
        L->a_scale = load_f(m, pick(m, nm, sizeof(nm), i, "hc_attn_scale", "attn_hc.scale"), 3);
        L->f_fn = load_f(m, pick(m, nm, sizeof(nm), i, "hc_ffn_fn", "ffn_hc.fn"), (int64_t)mixn * hc * D);
        L->f_base = load_f(m, pick(m, nm, sizeof(nm), i, "hc_ffn_base", "ffn_hc.base"), mixn);
        L->f_scale = load_f(m, pick(m, nm, sizeof(nm), i, "hc_ffn_scale", "ffn_hc.scale"), 3);
        if (c->is_la[i]) {
            int hd = c->la_dim, K = c->conv_k;
            L->q = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.q_proj.weight", NULL), qkv, D);
            L->k = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.k_proj.weight", NULL), qkv, D);
            L->v = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.v_proj.weight", NULL), qkv, D);
            L->o = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.o_proj.weight", NULL), D, qkv);
            L->fa = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.f_a_proj.weight", "self_attn.forget_gate.f_a_proj.weight"), hd, D);
            L->fb = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.f_b_proj.weight", "self_attn.forget_gate.f_b_proj.weight"), qkv, hd);
            L->dt = load_f(m, pick(m, nm, sizeof(nm), i, "self_attn.dt_bias", "self_attn.forget_gate.dt_bias"), qkv);
            L->alog = load_f(m, pick(m, nm, sizeof(nm), i, "self_attn.A_log", "self_attn.forget_gate.A_log"), c->la_heads);
            L->bp = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.b_proj.weight", NULL), c->la_heads, D);
            L->ga = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.g_a_proj.weight", NULL), hd, D);
            L->gb = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.g_b_proj.weight", NULL), qkv, hd);
            L->onorm = load_f(m, pick(m, nm, sizeof(nm), i, "self_attn.o_norm.weight", NULL), hd);
            /* the checkpoint keeps q/k/v convs apart; the recurrence wants them in qkv order */
            if (has(m, pick(m, nm, sizeof(nm), i, "self_attn.conv1d.weight", NULL)))
                L->conv = load_f(m, nm, (int64_t)3 * qkv * K);
            else {
                L->conv = falloc((int64_t)3 * qkv * K);
                const char *parts[3] = {"self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"};
                for (int p = 0; p < 3; p++) {
                    float *w = load_f(m, pick(m, nm, sizeof(nm), i, parts[p], NULL), (int64_t)qkv * K);
                    memcpy(L->conv + (int64_t)p * qkv * K, w, (size_t)qkv * K * sizeof(float));
                    free(w);
                }
            }
        } else {
            int ql = c->q_lora, kl = c->kv_lora;
            L->qa = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.q_a_proj.weight", NULL), ql, D);
            L->qa_n = load_f(m, pick(m, nm, sizeof(nm), i, "self_attn.q_a_layernorm.weight", NULL), ql);
            L->qb = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.q_b_proj.weight", NULL), (int64_t)H * c->nope, ql);
            L->kva = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.kv_a_proj_with_mqa.weight", NULL), kl, D);
            L->kva_n = load_f(m, pick(m, nm, sizeof(nm), i, "self_attn.kv_a_layernorm.weight", NULL), kl);
            L->kvb = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.kv_b_proj.weight", NULL), (int64_t)H * (c->nope + c->vdim), kl);
            L->ao = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.o_proj.weight", NULL), D, (int64_t)H * c->vdim);
            if (!c->idx_shared[i]) {
                L->iqb = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.indexer.wq_b.weight", NULL), (int64_t)c->idx_heads * Id, ql);
                L->iwk = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.indexer.wk.weight", NULL), Id, D);
                L->ik_w = load_f(m, pick(m, nm, sizeof(nm), i, "self_attn.indexer.k_norm.weight", NULL), Id);
                L->ik_b = load_f(m, pick(m, nm, sizeof(nm), i, "self_attn.indexer.k_norm.bias", NULL), Id);
                L->iw = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.indexer.weights_proj.weight", NULL), c->idx_heads, D);
                L->iape = load_f(m, pick(m, nm, sizeof(nm), i, "self_attn.indexer.index_kpool_compress_ape", NULL), (int64_t)c->kpool * Id);
                L->igate = load_w(m, pick(m, nm, sizeof(nm), i, "self_attn.indexer.index_kpool_compress_gate", NULL), Id, D);
            }
        }
        if (c->is_moe[i]) {
            L->router = load_w(m, pick(m, nm, sizeof(nm), i, "mlp.gate.weight", NULL), c->E, D);
            pick(m, nm, sizeof(nm), i, "mlp.gate.e_score_correction_bias", NULL);
            L->corr = has(m, nm) ? load_f(m, nm, c->E) : NULL;
            L->sg = load_w(m, pick(m, nm, sizeof(nm), i, "mlp.shared_experts.gate_proj.weight", NULL), c->shared_I, D);
            L->su = load_w(m, pick(m, nm, sizeof(nm), i, "mlp.shared_experts.up_proj.weight", NULL), c->shared_I, D);
            L->sd = load_w(m, pick(m, nm, sizeof(nm), i, "mlp.shared_experts.down_proj.weight", NULL), D, c->shared_I);
            load_experts(m, L, i);
        } else {
            L->mg = load_w(m, pick(m, nm, sizeof(nm), i, "mlp.gate_proj.weight", NULL), c->dense_I, D);
            L->mu = load_w(m, pick(m, nm, sizeof(nm), i, "mlp.up_proj.weight", NULL), c->dense_I, D);
            L->md = load_w(m, pick(m, nm, sizeof(nm), i, "mlp.down_proj.weight", NULL), D, c->dense_I);
        }
    }
    if (m->xq && (c->D % 32 || c->moe_I % 32)) {
        fprintf(stderr, "int4 container needs hidden and moe_intermediate multiples of 32\n"); exit(1); }
#ifdef COLI_CUDA
    if (g_cuda_on) fprintf(stderr, "[glm53] cuda: %.1f GB of bf16 residents in VRAM\n", g_cuda_used / 1e9);
#endif
}

/* ---------- state ---------- */
static void state_init(Model *m, State *s) {
    Cfg *c = &m->c;
    s->len = 0; s->cap = 0;
    s->l = calloc(c->n_layers, sizeof(LState));
    int qkv = c->la_heads * c->la_dim;
    for (int i = 0; i < c->n_layers; i++)
        if (c->is_la[i]) {
            s->l[i].win = fzalloc((int64_t)3 * qkv * c->conv_k);
            s->l[i].S = fzalloc((int64_t)c->la_heads * c->la_dim * c->la_dim);
        }
}
static void state_free(Model *m, State *s) {
    for (int i = 0; i < m->c.n_layers; i++) {
        LState *l = &s->l[i];
        free(l->win); free(l->S); free(l->lat); free(l->ik); free(l->ig); free(l->pool);
    }
    free(s->l);
}
/* DSA caches grow with the sequence (doubling); KDA state never does. */
static void state_reserve(Model *m, State *s, int need) {
    if (need <= s->cap) return;
    Cfg *c = &m->c;
    int cap = s->cap ? s->cap : 64;
    while (cap < need) cap *= 2;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_la[i]) continue;
        LState *l = &s->l[i];
        l->lat = realloc(l->lat, (size_t)cap * c->kv_lora * sizeof(float));
        if (!c->idx_shared[i]) {
            l->ik = realloc(l->ik, (size_t)cap * c->idx_dim * sizeof(float));
            l->ig = realloc(l->ig, (size_t)cap * c->idx_dim * sizeof(float));
            l->pool = realloc(l->pool, (size_t)(cap / c->kpool + 1) * c->idx_dim * sizeof(float));
        }
        if (!l->lat || (!c->idx_shared[i] && (!l->ik || !l->ig || !l->pool))) { fprintf(stderr, "OOM growing DSA cache\n"); exit(1); }
    }
    s->cap = cap;
}

/* ---------- small math ---------- */
static void rmsnorm_w(float *out, const float *x, const float *w, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * inv);
}
static inline float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
/* clamped SwiGLU: gate <= lim, |up| <= lim, silu(gate) * up */
static inline float swiglu_c(float g, float u, float lim) {
    g = g > lim ? lim : g;
    u = u > lim ? lim : (u < -lim ? -lim : u);
    return siluf(g) * u;
}

/* ---------- KDA ---------- */
/* One Kimi-Delta-Attention step for one head. S: [dk][dv]; q, k already l2-normed and q
 * pre-scaled; e[a] = exp(g[a]) per key channel. Reference order: decay, read, update,
 * read — the literal form (it is ~1% of the per-token work, so clarity wins). */
static void kda_step(float *S, const float *q, const float *k, const float *v, const float *e,
                     float beta, float *out, int dk, int dv) {
    float mem[512];
    for (int j = 0; j < dv; j++) mem[j] = 0.f;
    for (int a = 0; a < dk; a++) {
        float *row = S + (size_t)a * dv, ea = e[a], ka = k[a];
        for (int j = 0; j < dv; j++) { row[j] *= ea; mem[j] += row[j] * ka; }
    }
    for (int j = 0; j < dv; j++) mem[j] = (v[j] - mem[j]) * beta;     /* delta */
    for (int j = 0; j < dv; j++) out[j] = 0.f;
    for (int a = 0; a < dk; a++) {
        float *row = S + (size_t)a * dv, ka = k[a], qa = q[a];
        for (int j = 0; j < dv; j++) { row[j] += ka * mem[j]; out[j] += row[j] * qa; }
    }
}

static void kda_forward(Model *m, Layer *L, LState *ls, const float *x, int n, float *out) {
    Cfg *c = &m->c;
    int D = c->D, H = c->la_heads, hd = c->la_dim, qkv = H * hd;
    float *mix = falloc((int64_t)n * 3 * qkv), *g = falloc((int64_t)n * qkv);
    float *fa = falloc((int64_t)n * hd), *beta = falloc((int64_t)n * H);
    float *gate = falloc((int64_t)n * qkv), *o = falloc((int64_t)n * qkv), *tmp = falloc((int64_t)n * qkv);

    mmw(tmp, x, L->q, n, D, qkv);
    for (int t = 0; t < n; t++) memcpy(mix + (int64_t)t * 3 * qkv, tmp + (int64_t)t * qkv, qkv * sizeof(float));
    mmw(tmp, x, L->k, n, D, qkv);
    for (int t = 0; t < n; t++) memcpy(mix + (int64_t)t * 3 * qkv + qkv, tmp + (int64_t)t * qkv, qkv * sizeof(float));
    mmw(tmp, x, L->v, n, D, qkv);
    for (int t = 0; t < n; t++) memcpy(mix + (int64_t)t * 3 * qkv + 2 * qkv, tmp + (int64_t)t * qkv, qkv * sizeof(float));
    linattn_conv_span(ls->win, mix, L->conv, NULL, mix, n, 3 * qkv, c->conv_k);   /* SiLU(conv) */

    /* forget gate (per channel) and beta (per head) */
    mmw(fa, x, L->fa, n, D, hd);
    mmw(g, fa, L->fb, n, hd, qkv);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < H; h++) {
            float A = expf(L->alog[h]);
            for (int d = 0; d < hd; d++) {
                int ch = h * hd + d;
                float z = g[(int64_t)t * qkv + ch] + L->dt[ch];
                g[(int64_t)t * qkv + ch] = c->la_has_lb ? c->la_lb * sigmoidf(A * z)
                                                        : -A * (z > 20.f ? z : log1pf(expf(z)));
            }
        }
    mmw(beta, x, L->bp, n, D, H);
    for (int64_t i = 0; i < (int64_t)n * H; i++) beta[i] = sigmoidf(beta[i]);
    mmw(fa, x, L->ga, n, D, hd);
    mmw(gate, fa, L->gb, n, hd, qkv);

    /* the recurrence: sequential in t, independent per head */
    float scale = 1.f / sqrtf((float)hd);
    #pragma omp parallel for schedule(static) if(H > 1)
    for (int h = 0; h < H; h++) {
        float q[512], k[512], e[512];
        for (int t = 0; t < n; t++) {
            const float *row = mix + (int64_t)t * 3 * qkv;
            memcpy(q, row + h * hd, hd * sizeof(float));
            memcpy(k, row + qkv + h * hd, hd * sizeof(float));
            linattn_l2norm(q, hd, 1e-6f);
            linattn_l2norm(k, hd, 1e-6f);
            for (int d = 0; d < hd; d++) { q[d] *= scale; e[d] = expf(g[(int64_t)t * qkv + h * hd + d]); }
            float *oh = o + (int64_t)t * qkv + h * hd;
            kda_step(ls->S + (size_t)h * hd * hd, q, k, row + 2 * qkv + h * hd, e,
                     beta[(int64_t)t * H + h], oh, hd, hd);
            /* RMSNormGated: weight * norm(o) * sigmoid(gate), per head */
            const float *gt = gate + (int64_t)t * qkv + h * hd;
            rmsnorm_w(oh, oh, L->onorm, hd, c->eps);
            for (int d = 0; d < hd; d++) oh[d] *= sigmoidf(gt[d]);
        }
    }
    mmw(out, o, L->o, n, qkv, D);
    free(mix); free(g); free(fa); free(beta); free(gate); free(o); free(tmp);
}

/* ---------- DSA: pooled lightning indexer + absorbed NoPE MLA ---------- */
/* LayerNorm with weight and bias (the indexer's k_norm; eps 1e-6) */
static void layernorm_wb(float *x, const float *w, const float *b, int n) {
    double mu = 0; for (int i = 0; i < n; i++) mu += x[i];
    mu /= n;
    double var = 0; for (int i = 0; i < n; i++) { double d = x[i] - mu; var += d * d; }
    float inv = 1.f / sqrtf((float)(var / n) + 1e-6f);
    for (int i = 0; i < n; i++) x[i] = (float)(x[i] - mu) * inv * w[i] + b[i];
}

/* A selection: for each query, the attended positions in ascending order. */
typedef struct { int *pos; int *cnt; int width; } Sel;

static void sel_alloc(Sel *s, const Cfg *c, int n) {
    s->width = c->idx_topk + c->kpool;
    s->pos = malloc((size_t)n * s->width * sizeof(int));
    s->cnt = calloc(n, sizeof(int));
}
static void sel_free(Sel *s) { free(s->pos); free(s->cnt); s->pos = s->cnt = NULL; }

/* Run the indexer for n queries at positions pos0.. and write their selections. Pool p
 * covers tokens [p*kp, (p+1)*kp); its key is a per-channel softmax over the pool's slots
 * of (gate . x_token + ape[slot]), mixing the tokens' index keys. A query at position P
 * can select pools that END at or before P, keeps the top index_topk/kp of them by
 * sum_h w_h relu(q_h . key * Id^-0.5), and always attends the incomplete visible tail. */
static void indexer(Model *m, Layer *L, LState *ls, const float *x, const float *qres,
                    int n, int pos0, Sel *sel) {
    Cfg *c = &m->c;
    int D = c->D, Id = c->idx_dim, IH = c->idx_heads, kp = c->kpool, len = pos0 + n;
    float *q = falloc((int64_t)n * IH * Id), *w = falloc((int64_t)n * IH);
    mmw(q, qres, L->iqb, n, c->q_lora, IH * Id);
    mmw(w, x, L->iw, n, D, IH);
    mmw(ls->ik + (int64_t)pos0 * Id, x, L->iwk, n, D, Id);
    mmw(ls->ig + (int64_t)pos0 * Id, x, L->igate, n, D, Id);
    for (int t = 0; t < n; t++) layernorm_wb(ls->ik + (int64_t)(pos0 + t) * Id, L->ik_w, L->ik_b, Id);

    /* close every pool this call completed */
    for (int p = ls->npool; (p + 1) * kp <= len; p++) {
        float *pk = ls->pool + (int64_t)p * Id;
        for (int d = 0; d < Id; d++) {
            float lg[256], mx = -INFINITY;
            for (int s = 0; s < kp; s++) {
                lg[s] = ls->ig[(int64_t)(p * kp + s) * Id + d] + L->iape[s * Id + d];
                if (lg[s] > mx) mx = lg[s];
            }
            float sum = 0;
            for (int s = 0; s < kp; s++) { lg[s] = expf(lg[s] - mx); sum += lg[s]; }
            float a = 0;
            for (int s = 0; s < kp; s++) a += (lg[s] / sum) * ls->ik[(int64_t)(p * kp + s) * Id + d];
            pk[d] = a;
        }
        ls->npool = p + 1;
    }

    int keep = c->idx_topk / kp;
    float sc = 1.f / sqrtf((float)Id), hs = 1.f / sqrtf((float)IH);
    #pragma omp parallel for schedule(static) if(n > 1)
    for (int t = 0; t < n; t++) {
        int P = pos0 + t, nvis = (P + 1) / kp;
        int *dst = sel->pos + (int64_t)t * sel->width, cnt = 0;
        char *take = calloc(nvis > 0 ? nvis : 1, 1);
        if (nvis <= keep) memset(take, 1, nvis);
        else {
            float *score = malloc((size_t)nvis * sizeof(float));
            const float *qt = q + (int64_t)t * IH * Id, *wt = w + (int64_t)t * IH;
            for (int p = 0; p < nvis; p++) {
                const float *pk = ls->pool + (int64_t)p * Id;
                float s = 0;
                for (int h = 0; h < IH; h++) {
                    float d = 0; for (int j = 0; j < Id; j++) d += qt[h * Id + j] * pk[j];
                    d *= sc;
                    s += (wt[h] * hs) * (d > 0 ? d : 0);
                }
                score[p] = s;
            }
            /* top `keep` by score; ties to the lower pool (torch.topk leaves it undefined —
             * the fixture uses 8 indexer heads so a tie is not something it measures) */
            for (int r = 0; r < keep; r++) {
                int best = -1;
                for (int p = 0; p < nvis; p++)
                    if (!take[p] && (best < 0 || score[p] > score[best])) best = p;
                take[best] = 1;
            }
            free(score);
        }
        for (int p = 0; p < nvis; p++)
            if (take[p]) for (int s = 0; s < kp; s++) dst[cnt++] = p * kp + s;
        if (c->tail) for (int j = nvis * kp; j <= P; j++) dst[cnt++] = j;
        sel->cnt[t] = cnt;
        free(take);
    }
    free(q); free(w);
}

static void dsa_forward(Model *m, Layer *L, LState *ls, const float *x, int n, int pos0,
                        Sel *sel, float *out) {
    Cfg *c = &m->c;
    int D = c->D, H = c->n_heads, nope = c->nope, vd = c->vdim, ql = c->q_lora, kl = c->kv_lora;
    float *qres = falloc((int64_t)n * ql), *q = falloc((int64_t)n * H * nope);
    float *att = falloc((int64_t)n * H * vd), *lat = falloc((int64_t)n * kl);
    mmw(qres, x, L->qa, n, D, ql);
    for (int t = 0; t < n; t++) rmsnorm_w(qres + (int64_t)t * ql, qres + (int64_t)t * ql, L->qa_n, ql, c->eps);
    mmw(q, qres, L->qb, n, ql, H * nope);
    mmw(lat, x, L->kva, n, D, kl);
    for (int t = 0; t < n; t++)
        rmsnorm_w(ls->lat + (int64_t)(pos0 + t) * kl, lat + (int64_t)t * kl, L->kva_n, kl, c->eps);
    if (L->iqb.f || L->iqb.b) indexer(m, L, ls, x, qres, n, pos0, sel);   /* else: shared, reuse */

    /* absorbed attention, one (query, head) per task:
     *   score_j = (Wk_h^T q_h) . lat_j / sqrt(nope)   out_h = Wv_h (sum_j a_j lat_j) */
    float scale = 1.f / sqrtf((float)nope);
    int rows = nope + vd;
    #pragma omp parallel for schedule(dynamic) if(n * H > 1)
    for (int th = 0; th < n * H; th++) {
        int t = th / H, h = th % H;
        float *ql_ = malloc((size_t)kl * sizeof(float)), *ol = calloc(kl, sizeof(float));
        float *wrow = malloc((size_t)kl * sizeof(float));
        const float *qh = q + (int64_t)t * H * nope + h * nope;
        for (int j = 0; j < kl; j++) ql_[j] = 0.f;
        for (int d = 0; d < nope; d++) {
            const float *wk = wt_row(L->kvb, ((int64_t)h * rows + d) * kl, wrow, kl);
            float qd = qh[d];
            for (int j = 0; j < kl; j++) ql_[j] += qd * wk[j];
        }
        const int *pos = sel->pos + (int64_t)t * sel->width;
        int cnt = sel->cnt[t];
        float *sc = malloc((size_t)(cnt > 0 ? cnt : 1) * sizeof(float)), mx = -INFINITY;
        for (int i = 0; i < cnt; i++) {
            const float *lj = ls->lat + (int64_t)pos[i] * kl;
            float s = 0; for (int j = 0; j < kl; j++) s += ql_[j] * lj[j];
            sc[i] = s * scale;
            if (sc[i] > mx) mx = sc[i];
        }
        float sum = 0;
        for (int i = 0; i < cnt; i++) { sc[i] = expf(sc[i] - mx); sum += sc[i]; }
        for (int i = 0; i < cnt; i++) {
            const float *lj = ls->lat + (int64_t)pos[i] * kl;
            float a = sc[i] / sum;
            for (int j = 0; j < kl; j++) ol[j] += a * lj[j];
        }
        float *oh = att + (int64_t)t * H * vd + h * vd;
        for (int v = 0; v < vd; v++) {
            const float *wv = wt_row(L->kvb, ((int64_t)h * rows + nope + v) * kl, wrow, kl);
            float s = 0; for (int j = 0; j < kl; j++) s += wv[j] * ol[j];
            oh[v] = s;
        }
        free(ql_); free(ol); free(wrow); free(sc);
    }
    mmw(out, att, L->ao, n, H * vd, D);
    free(qres); free(q); free(att); free(lat);
}

/* ---------- FFN ---------- */
static void dense_mlp(Model *m, Layer *L, const float *x, int n, float *out) {
    Cfg *c = &m->c;
    int D = c->D, I = c->dense_I;
    float *g = falloc((int64_t)n * I), *u = falloc((int64_t)n * I);
    mmw(g, x, L->mg, n, D, I);
    mmw(u, x, L->mu, n, D, I);
    for (int64_t i = 0; i < (int64_t)n * I; i++) g[i] = swiglu_c(g[i], u[i], c->swiglu_limit);
    mmw(out, g, L->md, n, I, D);
    free(g); free(u);
}

/* sigmoid scores; selection on score + bias inside the best topk_group groups (a group
 * scores by the sum of its top-2); combine weights from the UNBIASED score, renormalized,
 * times routed_scaling_factor. sel/w come back in ascending expert id: the reference
 * index_add_s experts in ascending order, so that is the summation order. */
static void route(const Cfg *c, const float *logits, const float *corr, int *sel, float *w) {
    int E = c->E, K = c->K, G = c->n_group, per = E / G;
    float score[1024], choice[1024];
    for (int e = 0; e < E; e++) {
        score[e] = sigmoidf(logits[e]);
        choice[e] = score[e] + (corr ? corr[e] : 0.f);
    }
    if (G > 1) {
        float gs[64]; char gk[64] = {0};
        for (int g = 0; g < G; g++) {
            float a = -INFINITY, b = -INFINITY;
            for (int e = g * per; e < (g + 1) * per; e++) {
                if (choice[e] > a) { b = a; a = choice[e]; } else if (choice[e] > b) b = choice[e];
            }
            gs[g] = a + (per > 1 ? b : 0.f);
        }
        for (int r = 0; r < c->topk_group; r++) {
            int best = -1;
            for (int g = 0; g < G; g++) if (!gk[g] && (best < 0 || gs[g] > gs[best])) best = g;
            gk[best] = 1;
        }
        for (int e = 0; e < E; e++) if (!gk[e / per]) choice[e] = -INFINITY;
    }
    char used[1024] = {0};
    for (int a = 0; a < K; a++) {
        int best = -1;
        for (int e = 0; e < E; e++)
            if (!used[e] && (best < 0 || choice[e] > choice[best])) best = e;
        used[best] = 1;
    }
    int k = 0;
    for (int e = 0; e < E; e++) if (used[e]) { sel[k] = e; w[k] = score[e]; k++; }
    if (c->norm_topk) {
        float s = 1e-20f; for (int a = 0; a < K; a++) s += w[a];
        for (int a = 0; a < K; a++) w[a] /= s;
    }
    for (int a = 0; a < K; a++) w[a] *= c->route_scale;
}

/* Routed experts run GROUPED: each selected expert's weights are read once for all the
 * tokens routed to it (matmul_q4_kb / S=n_e GEMM), bit-identical to per-token. */
static void moe(Model *m, Layer *L, const float *x, int n, float *out) {
    Cfg *c = &m->c;
    int E = c->E, K = c->K, D = c->D, I = c->moe_I, SI = c->shared_I;
    int HI = I > SI ? I : SI;
    float *logits = falloc((int64_t)n * E), *w = falloc((int64_t)n * K);
    float *xg = falloc((int64_t)n * D), *h = falloc((int64_t)n * 2 * HI), *yg = falloc((int64_t)n * D);
    int *sel = malloc((size_t)n * K * sizeof(int)), *rows = malloc((size_t)n * K * sizeof(int));
    int *cnt = calloc(E, sizeof(int)), *start = malloc((size_t)(E + 1) * sizeof(int));

    mmw(logits, x, L->router, n, D, E);
    for (int t = 0; t < n; t++) route(c, logits + (int64_t)t * E, L->corr, sel + (int64_t)t * K, w + (int64_t)t * K);
    memset(out, 0, (size_t)n * D * sizeof(float));
    for (int i = 0; i < n * K; i++) cnt[sel[i]]++;
    start[0] = 0;
    for (int e = 0; e < E; e++) start[e + 1] = start[e] + cnt[e];
    memset(cnt, 0, E * sizeof(int));
    for (int i = 0; i < n * K; i++) rows[start[sel[i]] + cnt[sel[i]]++] = i;
    for (int e = 0; e < E; e++) {
        int ne = cnt[e];
        if (!ne) continue;
        const int *r = rows + start[e];
        for (int j = 0; j < ne; j++)
            memcpy(xg + (int64_t)j * D, x + (int64_t)(r[j] / K) * D, (size_t)D * sizeof(float));
        if (m->xq)
            matmul_q4_kb(h, xg, L->q_gu + (int64_t)e * 2 * I * (D / 2), L->s_gu + (int64_t)e * 2 * I,
                         ne, D, 2 * I, MOE_Q4_F32, g_exact);
        else
            matmul_f32(h, xg, L->e_gu + (int64_t)e * 2 * I * D, ne, D, 2 * I, g_exact);
        for (int j = 0; j < ne; j++) {                 /* [gate | up] -> act, packed */
            float *g = h + (int64_t)j * 2 * I, *u = g + I, *a = h + (int64_t)j * I;
            for (int i = 0; i < I; i++) a[i] = swiglu_c(g[i], u[i], c->swiglu_limit);
        }
        if (m->xq)
            matmul_q4_kb(yg, h, L->q_dn + (int64_t)e * D * (I / 2), L->s_dn + (int64_t)e * D,
                         ne, I, D, MOE_Q4_F32, g_exact);
        else
            matmul_f32(yg, h, L->e_dn + (int64_t)e * D * I, ne, I, D, g_exact);
        for (int j = 0; j < ne; j++) {
            float *o = out + (int64_t)(r[j] / K) * D, wt = w[r[j]];
            const float *y = yg + (int64_t)j * D;
            for (int i = 0; i < D; i++) o[i] += wt * y[i];
        }
    }
    /* shared expert, added after the routed sum (the reference's order) */
    float *g = h, *u = h + (int64_t)n * SI;
    mmw(g, x, L->sg, n, D, SI);
    mmw(u, x, L->su, n, D, SI);
    for (int64_t i = 0; i < (int64_t)n * SI; i++) g[i] = swiglu_c(g[i], u[i], c->swiglu_limit);
    mmw(yg, g, L->sd, n, SI, D);
    for (int64_t i = 0; i < (int64_t)n * D; i++) out[i] += yg[i];
    free(logits); free(w); free(xg); free(h); free(yg); free(sel); free(rows); free(cnt); free(start);
}

/* ---------- forward ---------- */
/* n tokens at positions pos0..; logits [n, V] if logits_all, else the last row only */
static void forward(Model *m, State *s, const int *ids, int n, int pos0, float *logits, int logits_all) {
    Cfg *c = &m->c;
    int D = c->D, V = c->V, hc = c->hc_mult, mixn = (2 + hc) * hc;
    MoeHC HC = { hc, D, c->hc_iters, c->hc_eps, c->eps };
    state_reserve(m, s, pos0 + n);

    float *hs = falloc((int64_t)n * hc * D), *erow = falloc(D);
    for (int t = 0; t < n; t++) {
        const float *e = wt_row(m->embed, (int64_t)ids[t] * D, erow, D);
        for (int k = 0; k < hc; k++) memcpy(hs + ((int64_t)t * hc + k) * D, e, (size_t)D * sizeof(float));
    }
    float *coll = falloc((int64_t)n * D), *xn = falloc((int64_t)n * D), *sub = falloc((int64_t)n * D);
    float *post = falloc((int64_t)n * hc), *comb = falloc((int64_t)n * hc * hc);
    float *flat = falloc((int64_t)n * hc * D), *mix = falloc((int64_t)n * mixn), *tmp = falloc((int64_t)hc * D);
    Sel sel; sel_alloc(&sel, c, n);

    for (int li = 0; li < c->n_layers; li++) {
        Layer *L = &m->L[li];
        double t0 = g_prof ? now_s() : 0;
        moe_hc_sites(&HC, L->a_fn, L->a_base, L->a_scale, hs, n, post, comb, coll, flat, mix, g_exact);
        for (int t = 0; t < n; t++) rmsnorm_w(xn + (int64_t)t * D, coll + (int64_t)t * D, L->ln1, D, c->eps);
        double t1 = g_prof ? now_s() : 0;
        if (c->is_la[li]) kda_forward(m, L, &s->l[li], xn, n, sub);
        else {
            if (c->idx_shared[li] && (li == 0 || c->is_la[li - 1])) {
                fprintf(stderr, "layer %d: a shared indexer needs a DSA layer right before it\n", li); exit(1); }
            dsa_forward(m, L, &s->l[li], xn, n, pos0, &sel, sub);
        }
        double t2 = g_prof ? now_s() : 0;
        if (g_prof) { if (c->is_la[li]) g_t_kda += t2 - t1; else g_t_dsa += t2 - t1; g_t_hc += t1 - t0; }
        for (int t = 0; t < n; t++)
            moe_hc_merge(&HC, hs + (int64_t)t * hc * D, post + (int64_t)t * hc, comb + (int64_t)t * hc * hc,
                         sub + (int64_t)t * D, tmp);

        moe_hc_sites(&HC, L->f_fn, L->f_base, L->f_scale, hs, n, post, comb, coll, flat, mix, g_exact);
        for (int t = 0; t < n; t++) rmsnorm_w(xn + (int64_t)t * D, coll + (int64_t)t * D, L->ln2, D, c->eps);
        double t3 = g_prof ? now_s() : 0;
        if (c->is_moe[li]) moe(m, L, xn, n, sub); else dense_mlp(m, L, xn, n, sub);
        if (g_prof) g_t_moe += now_s() - t3;
        for (int t = 0; t < n; t++)
            moe_hc_merge(&HC, hs + (int64_t)t * hc * D, post + (int64_t)t * hc, comb + (int64_t)t * hc * hc,
                         sub + (int64_t)t * D, tmp);
    }
    s->len = pos0 + n;

    /* hc head: the MEAN of the streams, then the final norm and lm_head */
    int t0 = logits_all ? 0 : n - 1;
    for (int t = t0; t < n; t++) {
        const float *st = hs + (int64_t)t * hc * D;
        float *hv = xn + (int64_t)(t - t0) * D;
        for (int i = 0; i < D; i++) {
            float a = 0; for (int k = 0; k < hc; k++) a += st[(int64_t)k * D + i];
            hv[i] = a / hc;
        }
        rmsnorm_w(hv, hv, m->norm, D, c->eps);
    }
    mmw(logits, xn, m->lm_head, n - t0, D, V);
    sel_free(&sel);
    free(hs); free(erow); free(coll); free(xn); free(sub); free(post); free(comb); free(flat); free(mix); free(tmp);
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
static int argmax_v(const float *l, int n) { int a = 0; for (int i = 1; i < n; i++) if (l[i] > l[a]) a = i; return a; }

/* Three gates, all required: teacher-forced argmax + logits of the last row (HF runs this
 * prefill through the CHUNKED KDA form), incremental greedy (the RECURRENT form plus every
 * cache), and a split prefill that must land where one shot does. */
static int run_oracle(Model *m, const char *ref_path) {
    long rn; char *rbuf = read_file(ref_path, &rn);
    jval *r = json_parse_checked(rbuf);
    jval *pj = r ? json_get(r, "prompt_ids") : NULL, *fj = r ? json_get(r, "full_ids") : NULL;
    jval *tj = r ? json_get(r, "tf_pred") : NULL;
    if (!pj || !fj || !tj || pj->t != J_ARR || fj->t != J_ARR || tj->t != J_ARR) {
        fprintf(stderr, "ref json missing keys\n"); return 1; }
    int np = pj->len, nf = fj->len, V = m->c.V;
    int *prompt = calloc(np + 1, sizeof(int)), *full = calloc(nf + 1, sizeof(int)), *tf = calloc(tj->len + 1, sizeof(int));
    for (int i = 0; i < np; i++) prompt[i] = (int)pj->kids[i]->num;
    for (int i = 0; i < nf; i++) full[i] = (int)fj->kids[i]->num;
    for (int i = 0; i < tj->len; i++) tf[i] = (int)tj->kids[i]->num;
    double ppl_ref = jnum(r, "ppl_ref", 0);

    State s; state_init(m, &s);
    float *lg = falloc((int64_t)nf * V);
    forward(m, &s, full, nf, 0, lg, 1);
    int tf_ok = 0; double nll = 0;
    for (int t = 0; t < nf; t++) {
        const float *l = lg + (int64_t)t * V;
        int arg = argmax_v(l, V);
        if (t < tj->len && arg == tf[t]) tf_ok++;
        else if (getenv("GLM53_TF") && t < tj->len) fprintf(stderr, "  tf mismatch at %d: ref %d got %d\n", t, tf[t], arg);
        if (t + 1 < nf) {
            float mx = l[0]; for (int i = 1; i < V; i++) if (l[i] > mx) mx = l[i];
            double sum = 0; for (int i = 0; i < V; i++) sum += exp((double)l[i] - mx);
            nll -= ((double)l[full[t + 1]] - mx) - log(sum);
        }
    }
    double ppl = exp(nll / (nf - 1));
    int num_ok = 1;
    jval *lj = json_get(r, "logits_last");
    if (lj && lj->t == J_ARR && lj->len == V) {
        const float *l = lg + (int64_t)(nf - 1) * V;
        double scale = 1e-6, md = 0;
        for (int i = 0; i < V; i++) { double a = fabs(lj->kids[i]->num); if (a > scale) scale = a; }
        for (int i = 0; i < V; i++) { double d = fabs((double)l[i] - lj->kids[i]->num); if (d > md) md = d; }
        num_ok = md / scale < 1e-3;
        printf("logits: max|diff| %.3e over %d (rel %.2e vs ref scale %.3f) — %s\n",
               md, V, md / scale, scale, num_ok ? "ok" : "TOO LARGE");
    }
    state_free(m, &s); free(lg);

    State s2; state_init(m, &s2);
    float *l1 = falloc(V);
    int ngen = nf - np, gen_ok = 0, *got = calloc(ngen + 1, sizeof(int));
    forward(m, &s2, prompt, np, 0, l1, 0);
    for (int i = 0; i < ngen; i++) {
        int arg = argmax_v(l1, V);
        got[i] = arg;
        if (arg == full[np + i]) gen_ok++;
        if (i + 1 < ngen) forward(m, &s2, &arg, 1, np + i, l1, 0);
    }
    state_free(m, &s2);

    int split_ok = 1;
    if (np >= 4) {
        int cut = np / 2 + 1;                   /* odd cut: splits a pool and a conv window */
        State a, b; state_init(m, &a); state_init(m, &b);
        float *la = falloc(V), *lb = falloc(V);
        forward(m, &a, prompt, np, 0, la, 0);
        forward(m, &b, prompt, cut, 0, lb, 0);
        forward(m, &b, prompt + cut, np - cut, cut, lb, 0);
        double sc = 1e-6, md = 0;
        for (int i = 0; i < V; i++) { double v = fabs(la[i]); if (v > sc) sc = v; }
        for (int i = 0; i < V; i++) { double d = fabs((double)la[i] - lb[i]); if (d > md) md = d; }
        split_ok = md / sc < 1e-4;
        printf("split prefill (%d+%d): max|diff| %.3e — %s\n", cut, np - cut, md, split_ok ? "ok" : "DIVERGED");
        state_free(m, &a); state_free(m, &b); free(la); free(lb);
    }
    printf("teacher forcing: %d/%d   greedy: %d/%d   ppl %.4f (ref %.4f, %+.2f%%)\n",
           tf_ok, (int)tj->len, gen_ok, ngen, ppl, ppl_ref, ppl_ref > 0 ? (ppl - ppl_ref) / ppl_ref * 100 : 0);
    if (gen_ok != ngen) {
        printf("  ref: "); for (int i = 0; i < ngen; i++) printf("%d ", full[np + i]);
        printf("\n  got: "); for (int i = 0; i < ngen; i++) printf("%d ", got[i]);
        printf("\n");
    }
    int ok = tf_ok == (int)tj->len && gen_ok == ngen && num_ok && split_ok;
    printf("%s\n", ok ? "ORACLE PASS" : "ORACLE FAIL");
    json_free(r); free(rbuf); free(prompt); free(full); free(tf); free(l1); free(got);
    return !ok;
}

/* ---------- generation ---------- */
static int is_eos(const Cfg *c, int t) { for (int e = 0; e < c->n_eos; e++) if (t == c->eos[e]) return 1; return 0; }

static void generate(Model *m, Tok *T, const char *text, int n_gen, float temp, float top_p) {
    Cfg *c = &m->c;
    int cap = (int)strlen(text) + 16, *ids = malloc(sizeof(int) * cap);
    int np = tok_encode(T, text, (int)strlen(text), ids, cap);
    if (np <= 0) { fprintf(stderr, "empty prompt\n"); free(ids); return; }
    State s; state_init(m, &s);
    float *lg = falloc(c->V);
    double t0 = now_s();
    forward(m, &s, ids, np, 0, lg, 0);
    double pre = now_s() - t0;
    fprintf(stderr, "[glm53] prefill %d tok in %.2fs (%.1f tok/s)\n", np, pre, np / pre);
    double td = now_s();
    int ngen = 0;
    for (int i = 0; i < n_gen; i++) {
        int tk = temp > 0 ? sample_logits(lg, c->V, temp, top_p) : argmax_v(lg, c->V);
        if (is_eos(c, tk)) break;
        char buf[512];
        int bl = tok_decode(T, &tk, 1, buf, sizeof(buf) - 1);
        if (bl > 0) { fwrite(buf, 1, bl, stdout); fflush(stdout); }
        ngen++;
        forward(m, &s, &tk, 1, np + i, lg, 0);
    }
    double dec = now_s() - td;
    printf("\n");
    fprintf(stderr, "[glm53] %d tok in %.2fs (%.2f tok/s), rss %.1f GB\n", ngen, dec, ngen > 0 ? ngen / dec : 0.0, rss_gb());
    if (g_prof)
        fprintf(stderr, "[prof] mhc %.2fs kda %.2fs dsa %.2fs ffn %.2fs\n", g_t_hc, g_t_kda, g_t_dsa, g_t_moe);
    state_free(m, &s); free(lg); free(ids);
}

/* ---------- serve: the gateway's SUBMIT/CANCEL protocol, one request at a time ---------- */
static void serve_one(Model *m, Tok *T, SReq *q) {
    Cfg *c = &m->c;
    int cap = q->plen + 16, *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    const char *bad = np <= 0 ? "empty prompt" : serve_fit(np, &q->max_tok, c->ctx_max);
    if (bad) { printf("ERROR %s %s\n", q->id, bad); fflush(stdout); free(ids); return; }
    State s; state_init(m, &s);
    float *lg = falloc(c->V);
    double t0 = now_s();
    forward(m, &s, ids, np, 0, lg, 0);
    int gen = 0, limited = 1, cancelled = 0;
    for (int i = 0; i < q->max_tok && !cancelled; i++) {
        int tk = q->temp > 0 ? sample_logits(lg, c->V, q->temp, q->top_p) : argmax_v(lg, c->V);
        if (is_eos(c, tk)) { limited = 0; break; }
        char buf[512];
        int bl = tok_decode(T, &tk, 1, buf, sizeof(buf) - 1);
        printf("DATA %s %d\n", q->id, bl > 0 ? bl : 0);
        if (bl > 0) fwrite(buf, 1, bl, stdout);
        printf("\n"); fflush(stdout);
        gen++;
        while (stdin_readable()) { int r = serve_read_cmd(q->id); if (r != 0) { cancelled = 1; limited = 0; break; } }
        if (i + 1 < q->max_tok && !cancelled) forward(m, &s, &tk, 1, np + i, lg, 0);
    }
    double dt = now_s() - t0;
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen, dt > 0 ? gen / dt : 0.0, 100.0, rss_gb(), np, limited);
    fflush(stdout);
    state_free(m, &s); free(lg); free(ids);
}

static int serve(Model *m, Tok *T) {
    printf("READY\n"); fflush(stdout);
    for (;;) {
        if (!g_qn) { if (serve_read_cmd(NULL) < 0) break; continue; }
        SReq q = g_q[0];
        memmove(g_q, g_q + 1, (size_t)(--g_qn) * sizeof(SReq));
        if (serve_take_cancel(q.id)) {
            printf("DONE %s STAT 0 0.000 100.0 %.2f 0 0\n", q.id, rss_gb()); fflush(stdout);
        } else serve_one(m, T, &q);
        free(q.payload);
    }
    return 0;
}

int main(int argc, char **argv) {
    moe_omp_autotune();                  /* one OpenMP thread per PHYSICAL core */
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    if (getenv("EXACT")) g_exact = 1;
    if (getenv("GLM53_PROF")) g_prof = 1;
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
    if (getenv("SEED")) g_rng = strtoull(getenv("SEED"), NULL, 10);

    Model m;
    double t0 = now_s();
    model_load(&m, snap);
    int nla = 0; for (int i = 0; i < m.c.n_layers; i++) nla += m.c.is_la[i];
    fprintf(stderr, "[glm53] %d layers (%d KDA, %d DSA), %d experts (top-%d)%s, hc_mult %d, loaded in %.1fs (rss %.1f GB)\n",
            m.c.n_layers, nla, m.c.n_layers - nla, m.c.E, m.c.K, m.xq ? " int4" : "", m.c.hc_mult,
            now_s() - t0, rss_gb());
    if (ref) { g_exact = 1; return run_oracle(&m, ref); }

    char tkp[1024]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
    Tok T; tok_load(&T, tkp);
    if (getenv("SERVE")) return serve(&m, &T);
    char *txt = NULL;
    if (pfile) { long n; txt = read_file(pfile, &n); }
    else if (prompt) txt = strdup(prompt);
    else { fprintf(stderr, "usage: %s -p \"prompt\" | -f prompt.txt [-n N] [-t temp] | <ref.json> | SERVE=1\n", argv[0]); return 1; }
    generate(&m, &T, txt, n_gen, temp, top_p);
    free(txt);
    return 0;
}
