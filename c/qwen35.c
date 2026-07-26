/* qwen35.c — Qwen3.5-MoE engine (linear attention + full attention + 256-expert MoE).
 *
 * The architecture, and why it is here: 30 of 40 layers use a *gated delta net* recurrent
 * linear-attention mixer instead of softmax attention (see docs/linear-attention.md). Its
 * per-sequence state is O(1) in context length, and at int4 the whole 35B container is
 * 21.7 GB — so unlike Laguna (57 GB of experts against ~40 GB of spare VRAM, which forces a
 * CPU expert tier costing 70% of expert time) this model fits on one 48 GB card whole.
 *
 * Layer:  x += mixer(rmsnorm(x));  x += moe(rmsnorm(x))
 *   mixer = linear_attn (gated delta net) or self_attn (GQA + per-element output gate)
 *   moe   = softmax(all 256) -> top-8 -> renormalize, plus sigmoid-gated shared expert
 *
 * Traps this code exists to get right, each of which was read off the reference rather than
 * guessed (transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py):
 *   - q_proj emits head_dim*2 per head, viewed [n_heads, head_dim*2] and chunked on the LAST
 *     dim: query and gate are INTERLEAVED PER HEAD, not block-concatenated. Getting this
 *     wrong still produces plausible activations. (Laguna's M.1 gate-width trap, new shape.)
 *   - the output gate is per-ELEMENT (head_dim wide per head), applied as out*sigmoid(gate).
 *   - routing is softmax over ALL experts, THEN top-k, THEN renormalize by the top-k sum.
 *     Not top-k-then-softmax. No correction bias, no route scale.
 *   - rotary is PARTIAL: rdim = head_dim*partial_rotary_factor (64 of 256 in the 35B); the
 *     rest of each head passes through unrotated. mRoPE's 3 position grids collapse to plain
 *     RoPE for text-only input, which is all this engine does.
 *   - q_norm/k_norm are RMSNorm over head_dim, applied per head BEFORE rope.
 *
 * Validation: SNAP=/tmp/tq35 ./qwen35 /tmp/tq35/ref_qwen35.json  must be token-exact
 * against transformers (tools/make_tiny_qwen35.py builds both).
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "json.h"
#include "moe_math.h"
#include "moe_matmul.h"
#include "moe_quant.h"
#include "moe_attn.h"
#include "moe_linattn.h"
#include "moe_sample.h"
#include "st.h"
#include "tok.h"

#define MAXL 80
static int g_exact = 0;          /* double accumulation, for the oracle */

/* ---------- config ---------- */
typedef struct {
    int n_layers, D, V;
    int n_heads, n_kv, head_dim;               /* full attention */
    int E, K, moe_I, shared_I;                 /* MoE */
    int lin_kh, lin_vh, lin_kd, lin_vd, conv_k;   /* gated delta net */
    int key_dim, value_dim, conv_dim;
    int is_linear[MAXL];
    float eps;
    float inv[256];                            /* rotary inverse frequencies */
    int rdim;                                  /* rotated dims per head (<= head_dim) */
    int ctx_max;
} Cfg;

typedef struct {
    float *ln1, *ln2;
    /* full attention */
    void *wq, *wk, *wv, *wo;
    float *qn, *kn;
    /* linear attention */
    void *w_qkv, *w_z, *w_b, *w_a, *w_out;
    float *conv_w, *dt_bias, *A_log, *gnorm;
    /* MoE */
    float *router, *sgate;
    uint8_t *gu_q, *dn_q;                      /* int4 fused experts */
    float *gu_s, *dn_s;
    float **eg, **eu, **ed;                    /* f32 per-expert (tiny oracle) */
    void *sg, *su, *sd;                        /* shared expert */
} Layer;

typedef struct {
    shards S;
    Cfg c;
    Layer L[MAXL];
    void *embed, *lm_head;
    float *norm;
    int xq;                                    /* 1 = int4 fused experts */
    int res_dt;                                /* 0 f32, 1 bf16 */
} Model;

/* per-sequence state: full-attention layers own a KV cache, linear layers own recurrent
 * state + a conv window. Only 1/4 of layers pay the context-proportional cost. */
typedef struct {
    float **k, **v;                            /* [layer], [ctx][n_kv*head_dim] */
    LinAttnState *ls;                          /* [layer] */
    int len;
} State;

static float *falloc(int64_t n) {
    float *p = malloc((size_t)n * sizeof(float));
    if (!p) { fprintf(stderr, "OOM %lld floats\n", (long long)n); exit(1); }
    return p;
}

/* ---------- loading ---------- */
static double jnum(jval *o, const char *k, double dflt) {
    jval *v = json_get(o, k);
    return (v && v->t == J_NUM) ? v->num : dflt;
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
    /* multimodal checkpoints nest the text config; the tiny oracle does not */
    jval *o = json_get(root, "text_config");
    if (!o) o = root;

    c->n_layers = (int)jnum(o, "num_hidden_layers", 0);
    if (c->n_layers <= 0 || c->n_layers > MAXL) {
        fprintf(stderr, "unsupported num_hidden_layers %d\n", c->n_layers); exit(1); }
    c->D = (int)jnum(o, "hidden_size", 0);
    c->V = (int)jnum(o, "vocab_size", 0);
    c->n_heads = (int)jnum(o, "num_attention_heads", 0);
    c->n_kv = (int)jnum(o, "num_key_value_heads", 0);
    c->head_dim = (int)jnum(o, "head_dim", c->D / (c->n_heads ? c->n_heads : 1));
    c->E = (int)jnum(o, "num_experts", 0);
    c->K = (int)jnum(o, "num_experts_per_tok", 0);
    c->moe_I = (int)jnum(o, "moe_intermediate_size", 0);
    c->shared_I = (int)jnum(o, "shared_expert_intermediate_size", 0);
    c->lin_kh = (int)jnum(o, "linear_num_key_heads", 0);
    c->lin_vh = (int)jnum(o, "linear_num_value_heads", 0);
    c->lin_kd = (int)jnum(o, "linear_key_head_dim", 0);
    c->lin_vd = (int)jnum(o, "linear_value_head_dim", 0);
    c->conv_k = (int)jnum(o, "linear_conv_kernel_dim", 4);
    c->eps = (float)jnum(o, "rms_norm_eps", 1e-6);
    c->key_dim = c->lin_kh * c->lin_kd;
    c->value_dim = c->lin_vh * c->lin_vd;
    c->conv_dim = c->key_dim * 2 + c->value_dim;
    if (c->lin_kh && c->lin_vh % c->lin_kh) {
        fprintf(stderr, "value heads %d not a multiple of key heads %d\n", c->lin_vh, c->lin_kh);
        exit(1); }
    if (c->lin_kd > LINATTN_MAX_KDIM || c->lin_vd > LINATTN_MAX_VDIM) {
        fprintf(stderr, "linear head dim %d/%d exceeds LINATTN_MAX (%d/%d)\n",
                c->lin_kd, c->lin_vd, LINATTN_MAX_KDIM, LINATTN_MAX_VDIM); exit(1); }

    /* layer_types drives which mixer each layer uses; fall back to the interval. */
    jval *lt = json_get(o, "layer_types");
    if (lt && lt->t == J_ARR) {
        if (lt->len != c->n_layers) {
            fprintf(stderr, "layer_types has %d entries, expected %d\n", lt->len, c->n_layers);
            exit(1); }
        for (int i = 0; i < c->n_layers; i++)
            c->is_linear[i] = !strcmp(lt->kids[i]->str, "linear_attention");
    } else {
        int iv = (int)jnum(o, "full_attention_interval", 4);
        for (int i = 0; i < c->n_layers; i++) c->is_linear[i] = ((i + 1) % iv) != 0;
    }

    /* Rotary. Partial: only head_dim*partial_rotary_factor dims rotate. mRoPE's three
     * position grids (mrope_section) are identical for text-only input, so the interleave
     * is a no-op here and plain RoPE is exact -- image/video input would not be. */
    jval *rp = json_get(o, "rope_parameters");
    float theta = (float)(rp ? jnum(rp, "rope_theta", 10000.0) : 10000.0);
    float prf = (float)(rp ? jnum(rp, "partial_rotary_factor", 1.0) : 1.0);
    c->rdim = (int)(c->head_dim * prf) & ~1;
    if (c->rdim / 2 > (int)(sizeof(c->inv) / sizeof(c->inv[0]))) {
        fprintf(stderr, "rotary dim %d too large\n", c->rdim); exit(1); }
    for (int j = 0; j < c->rdim / 2; j++)
        c->inv[j] = 1.0f / powf(theta, (float)(2 * j) / (float)c->rdim);

    c->ctx_max = (int)jnum(o, "max_position_embeddings", 4096);
    free(buf); free(arena);
}

static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    float *p = falloc(n); st_read_f32(&m->S, name, p, 0); return p;
}
/* Qwen3_5MoeRMSNorm scales by (1 + weight), NOT weight -- unlike Qwen3_5MoeRMSNormGated
 * (the linear_attn norm), which uses weight directly. Two conventions in one model. Folding
 * the +1 in at load keeps the shared rmsnorm_row kernel untouched and pays for it once.
 * This is load-bearing: with weight used raw, the tiny oracle scores 0/36 and ppl 254 vs 50. */
static float *load_norm(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    float *p = falloc(n); st_read_f32(&m->S, name, p, 0);
    for (int64_t i = 0; i < n; i++) p[i] += 1.f;
    return p;
}
static uint8_t *load_u8(Model *m, const char *name) {
    int64_t n = st_nbytes(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    uint8_t *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM %lld\n", (long long)n); exit(1); }
    st_read_raw(&m->S, name, p, 0); return p;
}
/* resident weight in the model's dtype: bf16 kept raw (half the bytes, exact widen in
 * matmul_bf16_k), f32 for the tiny oracle. */
static void *load_res(Model *m, const char *name) {
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    if (m->res_dt == 1 && t->dtype == 0) {
        uint16_t *p = malloc((size_t)t->numel * 2);
        if (!p) { fprintf(stderr, "OOM %s\n", name); exit(1); }
        st_read_raw(&m->S, name, p, 0); return p;
    }
    return load_t(m, name);
}
static void resmm(Model *m, float *y, const float *x, const void *W, int S, int I, int O) {
    if (m->res_dt == 1) matmul_bf16_k(y, x, (const uint16_t *)W, S, I, O, 0, g_exact);
    else matmul_f32(y, x, (const float *)W, S, I, O, g_exact);
}

/* The checkpoint nests text weights under model.language_model.; the converted container
 * strips that. Probe once and use whichever this snapshot has. */
static const char *g_pre = "model.";
static void detect_prefix(Model *m) {
    if (st_has(&m->S, "model.language_model.embed_tokens.weight")) g_pre = "model.language_model.";
    else if (st_has(&m->S, "model.embed_tokens.weight")) g_pre = "model.";
    else { fprintf(stderr, "no embed_tokens found (checked model. and model.language_model.)\n"); exit(1); }
}
#define NM(buf, fmt, ...) (snprintf(buf, sizeof(buf), "%s" fmt, g_pre, ##__VA_ARGS__), buf)

static void model_load(Model *m, const char *snap) {
    st_init(&m->S, snap);
    cfg_load(&m->c, snap);
    Cfg *c = &m->c;
    detect_prefix(m);
    char nm[512], base[256];

    NM(nm, "embed_tokens.weight");
    m->res_dt = (st_find(&m->S, nm)->dtype == 0) ? 1 : 0;

    /* int4 container? the converter emits fused experts with .qs row scales */
    snprintf(base, sizeof(base), "layers.0.mlp.experts.gate_up_proj.qs");
    m->xq = st_has(&m->S, NM(nm, "%s", base));

    m->embed = load_res(m, NM(nm, "embed_tokens.weight"));
    m->norm = load_norm(m, NM(nm, "norm.weight"));
    m->lm_head = st_has(&m->S, "lm_head.weight") ? load_res(m, "lm_head.weight") : m->embed;

    for (int i = 0; i < c->n_layers; i++) {
        Layer *L = &m->L[i];
        snprintf(base, sizeof(base), "layers.%d.input_layernorm.weight", i);
        L->ln1 = load_norm(m, NM(nm, "%s", base));
        snprintf(base, sizeof(base), "layers.%d.post_attention_layernorm.weight", i);
        L->ln2 = load_norm(m, NM(nm, "%s", base));

        if (c->is_linear[i]) {
            const char *sfx[] = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"};
            void **dst[] = {&L->w_qkv, &L->w_z, &L->w_b, &L->w_a, &L->w_out};
            for (int j = 0; j < 5; j++) {
                snprintf(base, sizeof(base), "layers.%d.linear_attn.%s.weight", i, sfx[j]);
                *dst[j] = load_res(m, NM(nm, "%s", base));
            }
            snprintf(base, sizeof(base), "layers.%d.linear_attn.conv1d.weight", i);
            L->conv_w = load_t(m, NM(nm, "%s", base));
            snprintf(base, sizeof(base), "layers.%d.linear_attn.dt_bias", i);
            L->dt_bias = load_t(m, NM(nm, "%s", base));
            snprintf(base, sizeof(base), "layers.%d.linear_attn.A_log", i);
            L->A_log = load_t(m, NM(nm, "%s", base));
            snprintf(base, sizeof(base), "layers.%d.linear_attn.norm.weight", i);
            L->gnorm = load_t(m, NM(nm, "%s", base));
        } else {
            const char *sfx[] = {"q_proj", "k_proj", "v_proj", "o_proj"};
            void **dst[] = {&L->wq, &L->wk, &L->wv, &L->wo};
            for (int j = 0; j < 4; j++) {
                snprintf(base, sizeof(base), "layers.%d.self_attn.%s.weight", i, sfx[j]);
                *dst[j] = load_res(m, NM(nm, "%s", base));
            }
            snprintf(base, sizeof(base), "layers.%d.self_attn.q_norm.weight", i);
            L->qn = load_norm(m, NM(nm, "%s", base));
            snprintf(base, sizeof(base), "layers.%d.self_attn.k_norm.weight", i);
            L->kn = load_norm(m, NM(nm, "%s", base));
            /* q_proj must be 2x the query width: query and gate interleaved per head */
            int64_t qn_el = st_numel(&m->S, NM(nm, "layers.%d.self_attn.q_proj.weight", i));
            int64_t want = (int64_t)c->n_heads * c->head_dim * 2 * c->D;
            if (qn_el != want) {
                fprintf(stderr, "layer %d q_proj has %lld elements, expected %lld "
                        "(n_heads*head_dim*2*D — attn_output_gate must be on)\n",
                        i, (long long)qn_el, (long long)want); exit(1); }
        }

        snprintf(base, sizeof(base), "layers.%d.mlp.gate.weight", i);
        L->router = load_t(m, NM(nm, "%s", base));
        snprintf(base, sizeof(base), "layers.%d.mlp.shared_expert_gate.weight", i);
        L->sgate = load_t(m, NM(nm, "%s", base));
        const char *ssfx[] = {"gate_proj", "up_proj", "down_proj"};
        void **sdst[] = {&L->sg, &L->su, &L->sd};
        for (int j = 0; j < 3; j++) {
            snprintf(base, sizeof(base), "layers.%d.mlp.shared_expert.%s.weight", i, ssfx[j]);
            *sdst[j] = load_res(m, NM(nm, "%s", base));
        }

        if (m->xq) {                            /* fused int4 */
            snprintf(base, sizeof(base), "layers.%d.mlp.experts.gate_up_proj", i);
            L->gu_q = load_u8(m, NM(nm, "%s", base));
            snprintf(base, sizeof(base), "layers.%d.mlp.experts.gate_up_proj.qs", i);
            L->gu_s = load_t(m, NM(nm, "%s", base));
            snprintf(base, sizeof(base), "layers.%d.mlp.experts.down_proj", i);
            L->dn_q = load_u8(m, NM(nm, "%s", base));
            snprintf(base, sizeof(base), "layers.%d.mlp.experts.down_proj.qs", i);
            L->dn_s = load_t(m, NM(nm, "%s", base));
        } else {
            /* f32, either fused [E,2I,D]/[E,D,I] or per-expert (what save_pretrained
             * writes, and what the tiny oracle therefore has). Normalize to per-expert
             * pointers so the forward path has one shape. */
            L->eg = malloc(sizeof(float *) * c->E);
            L->eu = malloc(sizeof(float *) * c->E);
            L->ed = malloc(sizeof(float *) * c->E);
            snprintf(base, sizeof(base), "layers.%d.mlp.experts.gate_up_proj", i);
            if (st_has(&m->S, NM(nm, "%s", base))) {
                float *gu = load_t(m, nm);
                snprintf(base, sizeof(base), "layers.%d.mlp.experts.down_proj", i);
                float *dn = load_t(m, NM(nm, "%s", base));
                int I = c->moe_I, D = c->D;
                for (int e = 0; e < c->E; e++) {
                    L->eg[e] = gu + (int64_t)e * 2 * I * D;
                    L->eu[e] = gu + (int64_t)e * 2 * I * D + (int64_t)I * D;
                    L->ed[e] = dn + (int64_t)e * D * I;
                }
            } else {
                for (int e = 0; e < c->E; e++) {
                    snprintf(base, sizeof(base), "layers.%d.mlp.experts.%d.gate_proj.weight", i, e);
                    L->eg[e] = load_t(m, NM(nm, "%s", base));
                    snprintf(base, sizeof(base), "layers.%d.mlp.experts.%d.up_proj.weight", i, e);
                    L->eu[e] = load_t(m, NM(nm, "%s", base));
                    snprintf(base, sizeof(base), "layers.%d.mlp.experts.%d.down_proj.weight", i, e);
                    L->ed[e] = load_t(m, NM(nm, "%s", base));
                }
            }
        }
    }
    fprintf(stderr, "[qwen35] %d layers (%d linear, %d full), D=%d V=%d E=%d top-%d, "
            "experts=%s residents=%s\n", c->n_layers,
            (int)({ int n = 0; for (int i = 0; i < c->n_layers; i++) n += c->is_linear[i]; n; }),
            (int)({ int n = 0; for (int i = 0; i < c->n_layers; i++) n += !c->is_linear[i]; n; }),
            c->D, c->V, c->E, c->K, m->xq ? "int4" : "f32", m->res_dt ? "bf16" : "f32");
}

/* ---------- state ---------- */
static void state_init(Model *m, State *s, int ctx) {
    Cfg *c = &m->c;
    s->k = calloc(c->n_layers, sizeof(float *));
    s->v = calloc(c->n_layers, sizeof(float *));
    s->ls = calloc(c->n_layers, sizeof(LinAttnState));
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_linear[i]) {
            s->ls[i].state = calloc((size_t)c->lin_vh * c->lin_kd * c->lin_vd, sizeof(float));
            s->ls[i].conv = calloc((size_t)c->conv_dim * c->conv_k, sizeof(float));
            if (!s->ls[i].state || !s->ls[i].conv) { fprintf(stderr, "OOM lin state\n"); exit(1); }
        } else {
            s->k[i] = falloc((int64_t)ctx * c->n_kv * c->head_dim);
            s->v[i] = falloc((int64_t)ctx * c->n_kv * c->head_dim);
        }
    }
    s->len = 0;
}
static void state_free(Model *m, State *s) {
    for (int i = 0; i < m->c.n_layers; i++) {
        free(s->k[i]); free(s->v[i]);
        free(s->ls[i].state); free(s->ls[i].conv);
    }
    free(s->k); free(s->v); free(s->ls);
}

/* ---------- MoE ---------- */
/* softmax over ALL experts, then top-k, then renormalize by the top-k sum. Order matters:
 * top-k-then-softmax gives different weights and is the common way to get this wrong. */
static void moe_route(const float *logit, int E, int K, int *sel, float *w) {
    float mx = -INFINITY;
    for (int e = 0; e < E; e++) if (logit[e] > mx) mx = logit[e];
    double sum = 0;
    float *p = falloc(E);
    for (int e = 0; e < E; e++) { p[e] = expf(logit[e] - mx); sum += p[e]; }
    for (int e = 0; e < E; e++) p[e] = (float)(p[e] / sum);
    /* top-k by value; ties broken by lower index, matching torch.topk */
    for (int a = 0; a < K; a++) {
        int best = -1;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int b = 0; b < a; b++) if (sel[b] == e) { taken = 1; break; }
            if (taken) continue;
            if (best < 0 || p[e] > p[best]) best = e;
        }
        sel[a] = best; w[a] = p[best];
    }
    double tot = 0;
    for (int a = 0; a < K; a++) tot += w[a];
    for (int a = 0; a < K; a++) w[a] = (float)(w[a] / tot);
    free(p);
}

static void moe_forward(Model *m, Layer *L, const float *x, float *out) {
    Cfg *c = &m->c;
    int D = c->D, I = c->moe_I, E = c->E, K = c->K, SI = c->shared_I;
    float *logit = falloc(E);
    matmul_f32(logit, x, L->router, 1, D, E, g_exact);
    int *sel = malloc(sizeof(int) * K);
    float *w = falloc(K);
    moe_route(logit, E, K, sel, w);

    for (int i = 0; i < D; i++) out[i] = 0.f;
    float *gu = falloc(2 * I), *glu = falloc(I), *eo = falloc(D);
    for (int a = 0; a < K; a++) {
        int e = sel[a];
        if (m->xq) {
            matmul_q4_k(gu, x, L->gu_q + (int64_t)(e * 2 * I) * (D / 2),
                        L->gu_s + (int64_t)e * 2 * I, D, 2 * I, MOE_Q4_F32, g_exact);
            for (int i = 0; i < I; i++) glu[i] = siluf(gu[i]) * gu[I + i];
            matmul_q4_k(eo, glu, L->dn_q + (int64_t)(e * D) * (I / 2),
                        L->dn_s + (int64_t)e * D, I, D, MOE_Q4_F32, g_exact);
        } else {
            matmul_f32(gu, x, L->eg[e], 1, D, I, g_exact);
            matmul_f32(gu + I, x, L->eu[e], 1, D, I, g_exact);
            for (int i = 0; i < I; i++) glu[i] = siluf(gu[i]) * gu[I + i];
            matmul_f32(eo, glu, L->ed[e], 1, I, D, g_exact);
        }
        for (int i = 0; i < D; i++) out[i] += w[a] * eo[i];
    }

    /* shared expert, scaled by sigmoid of its own 1-wide gate */
    float *sg = falloc(SI), *su = falloc(SI), *so = falloc(D), gval;
    matmul_f32(&gval, x, L->sgate, 1, D, 1, g_exact);
    float gscale = 1.f / (1.f + expf(-gval));
    resmm(m, sg, x, L->sg, 1, D, SI);
    resmm(m, su, x, L->su, 1, D, SI);
    for (int i = 0; i < SI; i++) sg[i] = siluf(sg[i]) * su[i];
    resmm(m, so, sg, L->sd, 1, SI, D);
    for (int i = 0; i < D; i++) out[i] += gscale * so[i];

    free(logit); free(sel); free(w); free(gu); free(glu); free(eo);
    free(sg); free(su); free(so);
}

/* ---------- mixers ---------- */
static void rope_apply(float *h, int pos, const float *inv, int rdim) {
    int half = rdim / 2;
    for (int j = 0; j < half; j++) {
        float ang = pos * inv[j];
        float cs = cosf(ang), sn = sinf(ang);
        float a = h[j], b = h[j + half];
        h[j] = a * cs - b * sn;
        h[j + half] = b * cs + a * sn;
    }
}

/* full attention over `n` tokens starting at absolute position pos0 */
static void attn_forward(Model *m, Layer *L, State *s, int li, const float *xn, int n,
                         int pos0, float *out) {
    Cfg *c = &m->c;
    int D = c->D, H = c->n_heads, hd = c->head_dim, nkv = c->n_kv;
    int qw = H * hd * 2, kvw = nkv * hd, grp = H / nkv;
    float scale = 1.f / sqrtf((float)hd);

    float *qp = falloc((int64_t)n * qw);
    float *kp = falloc((int64_t)n * kvw), *vp = falloc((int64_t)n * kvw);
    resmm(m, qp, xn, L->wq, n, D, qw);
    resmm(m, kp, xn, L->wk, n, D, kvw);
    resmm(m, vp, xn, L->wv, n, D, kvw);

    float *ao = falloc((int64_t)n * H * hd);
    float *sc = falloc(pos0 + n + 1);
    for (int t = 0; t < n; t++) {
        int pos = pos0 + t;
        /* K/V: per-head RMSNorm then rope, stored post-norm/post-rope */
        float *kdst = s->k[li] + (int64_t)pos * kvw, *vdst = s->v[li] + (int64_t)pos * kvw;
        for (int h = 0; h < nkv; h++) {
            rmsnorm_row(kdst + h * hd, kp + (int64_t)t * kvw + h * hd, L->kn, hd, c->eps);
            rope_apply(kdst + h * hd, pos, c->inv, c->rdim);
        }
        memcpy(vdst, vp + (int64_t)t * kvw, (size_t)kvw * sizeof(float));

        for (int h = 0; h < H; h++) {
            /* query and gate are interleaved per head: [q(hd) | gate(hd)] per head */
            const float *qsrc = qp + (int64_t)t * qw + (int64_t)h * hd * 2;
            const float *gate = qsrc + hd;
            float q[512];
            if (hd > 512) { fprintf(stderr, "head_dim %d > 512\n", hd); exit(1); }
            rmsnorm_row(q, qsrc, L->qn, hd, c->eps);
            rope_apply(q, pos, c->inv, c->rdim);
            float *o = ao + (int64_t)t * H * hd + (int64_t)h * hd;
            /* tau is a score MULTIPLIER in sdpa_head (sc = tau*(dot*scale + bias)), not a
             * softcap -- passing 0 makes every score equal and attention a uniform mean of V.
             * Qwen3.5 has no logit cap, so tau = 1. */
            sdpa_head(q, s->k[li] + (int64_t)(h / grp) * hd, s->v[li] + (int64_t)(h / grp) * hd,
                      kvw, hd, 0, pos, scale, 1.f, NULL, 0,
                      g_exact ? MOE_QK_DBL : MOE_QK_SIMD, o, sc, 0);
            /* per-element output gate */
            for (int d = 0; d < hd; d++) o[d] *= 1.f / (1.f + expf(-gate[d]));
        }
    }
    resmm(m, out, ao, L->wo, n, H * hd, D);
    free(qp); free(kp); free(vp); free(ao); free(sc);
}

/* gated delta net over `n` tokens; state and conv window carry across calls */
static void linattn_forward(Model *m, Layer *L, State *s, int li, const float *xn, int n,
                            float *out) {
    Cfg *c = &m->c;
    int D = c->D, cd = c->conv_dim, vdim = c->value_dim, kdim = c->key_dim, nv = c->lin_vh;

    float *mix = falloc((int64_t)n * cd);
    float *z = falloc((int64_t)n * vdim);
    float *bb = falloc((int64_t)n * nv), *aa = falloc((int64_t)n * nv);
    resmm(m, mix, xn, L->w_qkv, n, D, cd);
    resmm(m, z, xn, L->w_z, n, D, vdim);
    resmm(m, bb, xn, L->w_b, n, D, nv);
    resmm(m, aa, xn, L->w_a, n, D, nv);

    linattn_conv_span(s->ls[li].conv, mix, L->conv_w, NULL, mix, n, cd, c->conv_k);

    /* beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias) */
    float *g = falloc((int64_t)n * nv);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nv; h++) {
            int64_t o = (int64_t)t * nv + h;
            g[o] = -expf(L->A_log[h]) * linattn_softplus(aa[o] + L->dt_bias[h]);
            bb[o] = 1.f / (1.f + expf(-bb[o]));
        }

    /* conv output is [q | k | v] along the channel axis; linattn_scan wants them separate */
    float *q = falloc((int64_t)n * kdim), *k = falloc((int64_t)n * kdim);
    float *v = falloc((int64_t)n * vdim);
    for (int t = 0; t < n; t++) {
        const float *r = mix + (int64_t)t * cd;
        memcpy(q + (int64_t)t * kdim, r, (size_t)kdim * sizeof(float));
        memcpy(k + (int64_t)t * kdim, r + kdim, (size_t)kdim * sizeof(float));
        memcpy(v + (int64_t)t * vdim, r + 2 * kdim, (size_t)vdim * sizeof(float));
    }
    float *core = falloc((int64_t)n * vdim);
    linattn_scan(&s->ls[li], q, k, v, g, bb, core, n, c->lin_kh, nv, c->lin_kd, c->lin_vd);

    /* gated RMSNorm per value head, then out_proj */
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nv; h++)
            linattn_gated_norm(core + (int64_t)t * vdim + h * c->lin_vd,
                               z + (int64_t)t * vdim + h * c->lin_vd, L->gnorm, c->lin_vd, c->eps);
    resmm(m, out, core, L->w_out, n, vdim, D);

    free(mix); free(z); free(bb); free(aa); free(g);
    free(q); free(k); free(v); free(core);
}

/* ---------- forward ---------- */
/* Runs `n` tokens at absolute positions pos0..pos0+n-1. If `logits_all`, writes logits for
 * every position (teacher forcing); otherwise only the last. */
static void forward(Model *m, State *s, const int *ids, int n, int pos0, float *logits,
                    int logits_all) {
    Cfg *c = &m->c;
    int D = c->D, V = c->V;
    float *x = falloc((int64_t)n * D), *xn = falloc((int64_t)n * D);
    for (int t = 0; t < n; t++) {
        const float *row;
        float tmp[1];
        (void)tmp; (void)row;
        if (m->res_dt == 1) {
            const uint16_t *e = (const uint16_t *)m->embed + (int64_t)ids[t] * D;
            for (int i = 0; i < D; i++) x[(int64_t)t * D + i] = bf16_f32(e[i]);
        } else {
            memcpy(x + (int64_t)t * D, (const float *)m->embed + (int64_t)ids[t] * D,
                   (size_t)D * sizeof(float));
        }
    }

    int dbg = getenv("QW_DUMP") ? 1 : 0;
#define DUMP(tag, ptr) do { if (dbg) { const float *_v = (ptr) + (int64_t)(n-1)*D; \
      double _s = 0; for (int _i = 0; _i < D; _i++) _s += (double)_v[_i]*_v[_i]; \
      fprintf(stderr, "%-9s L2=%.6f  ", tag, sqrt(_s)); \
      for (int _i = 0; _i < 5 && _i < D; _i++) fprintf(stderr, "%+.5f ", _v[_i]); \
      fprintf(stderr, "\n"); } } while (0)
    DUMP("embed", x);
    float *mo = falloc((int64_t)n * D);
    for (int li = 0; li < c->n_layers; li++) {
        Layer *L = &m->L[li];
        for (int t = 0; t < n; t++)
            rmsnorm_row(xn + (int64_t)t * D, x + (int64_t)t * D, L->ln1, D, c->eps);
        if (c->is_linear[li]) linattn_forward(m, L, s, li, xn, n, mo);
        else                  attn_forward(m, L, s, li, xn, n, pos0, mo);
        if (dbg) { char tg[16]; snprintf(tg, sizeof(tg), "mix%d", li); DUMP(tg, mo); }
        for (int64_t i = 0; i < (int64_t)n * D; i++) x[i] += mo[i];

        for (int t = 0; t < n; t++)
            rmsnorm_row(xn + (int64_t)t * D, x + (int64_t)t * D, L->ln2, D, c->eps);
        for (int t = 0; t < n; t++)
            moe_forward(m, L, xn + (int64_t)t * D, mo + (int64_t)t * D);
        if (dbg) { char tg[16]; snprintf(tg, sizeof(tg), "moe%d", li); DUMP(tg, mo); }
        for (int64_t i = 0; i < (int64_t)n * D; i++) x[i] += mo[i];
        if (dbg) { char tg[16]; snprintf(tg, sizeof(tg), "layer%d", li); DUMP(tg, x); }
    }
    s->len = pos0 + n;

    int t0 = logits_all ? 0 : n - 1;
    for (int t = t0; t < n; t++) {
        float hn[8192];
        float *hp = D <= 8192 ? hn : falloc(D);
        rmsnorm_row(hp, x + (int64_t)t * D, m->norm, D, c->eps);
        resmm(m, logits + (int64_t)(logits_all ? t : 0) * V, hp, m->lm_head, 1, D, V);
        if (D > 8192) free(hp);
    }
    free(x); free(xn); free(mo);
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
    int *prompt = malloc(sizeof(int) * np), *full = malloc(sizeof(int) * nf);
    int *tf = malloc(sizeof(int) * tj->len);
    for (int i = 0; i < np; i++) prompt[i] = (int)pj->kids[i]->num;
    for (int i = 0; i < nf; i++) full[i] = (int)fj->kids[i]->num;
    for (int i = 0; i < tj->len; i++) tf[i] = (int)tj->kids[i]->num;
    double ppl_ref = jnum(r, "ppl_ref", 0);

    Cfg *c = &m->c;
    int V = c->V;

    /* 1. teacher forcing: argmax at every position, plus perplexity */
    State s; state_init(m, &s, nf + 8);
    float *lg = falloc((int64_t)nf * V);
    forward(m, &s, full, nf, 0, lg, 1);
    int tf_ok = 0;
    double nll = 0;
    for (int t = 0; t < nf; t++) {
        const float *l = lg + (int64_t)t * V;
        int arg = 0;
        for (int i = 1; i < V; i++) if (l[i] > l[arg]) arg = i;
        if (t < tj->len && arg == tf[t]) tf_ok++;
        else if (getenv("QW_TF") && t < tj->len)
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
    state_free(m, &s);
    free(lg);

    /* 2. greedy generation with the KV/recurrent state, which must reproduce full_ids —
     * this is the check that the incremental path agrees with the one-shot path. */
    State s2; state_init(m, &s2, nf + 8);
    float *l1 = falloc(V);
    int gen_ok = 0, ngen = nf - np;
    forward(m, &s2, prompt, np, 0, l1, 0);
    int *got = malloc(sizeof(int) * ngen);
    for (int i = 0; i < ngen; i++) {
        int arg = 0;
        for (int v = 1; v < V; v++) if (l1[v] > l1[arg]) arg = v;
        got[i] = arg;
        if (arg == full[np + i]) gen_ok++;
        if (i + 1 < ngen) forward(m, &s2, &arg, 1, np + i, l1, 0);
    }
    state_free(m, &s2);

    printf("teacher forcing: %d/%d   greedy: %d/%d   ppl %.4f (ref %.4f, %+.2f%%)\n",
           tf_ok, (int)tj->len, gen_ok, ngen, ppl, ppl_ref,
           ppl_ref > 0 ? (ppl - ppl_ref) / ppl_ref * 100 : 0);
    if (gen_ok != ngen) {
        printf("  ref: "); for (int i = 0; i < ngen; i++) printf("%d ", full[np + i]);
        printf("\n  got: "); for (int i = 0; i < ngen; i++) printf("%d ", got[i]);
        printf("\n");
    }
    int ok = (tf_ok == (int)tj->len) && (gen_ok == ngen);
    printf("%s\n", ok ? "ORACLE PASS" : "ORACLE FAIL");
    free(rbuf); free(arena); free(prompt); free(full); free(tf); free(l1); free(got);
    return !ok;
}

/* ---------- main ---------- */
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
    fprintf(stderr, "[qwen35] loaded in %.1fs (rss %.1f GB)\n", now_s() - t0, rss_gb());

    if (ref) { g_exact = 1; return run_oracle(&m, ref); }

    /* generation: tokenize, prefill, decode greedily or by sampling */
    Tok T;
    char tp[1024];
    snprintf(tp, sizeof(tp), "%s/tokenizer.json", snap);
    tok_load(&T, tp);
    char *text = NULL;
    long tn = 0;
    if (pfile) text = read_file(pfile, &tn);
    else if (prompt) { text = strdup(prompt); tn = strlen(text); }
    else { fprintf(stderr, "give -f <file> or -p <prompt>, or a ref .json for the oracle\n"); return 1; }

    int cap = (int)tn + 16;
    int *ids = malloc(sizeof(int) * cap);
    int np = tok_encode(&T, text, (int)tn, ids, cap);
    fprintf(stderr, "[qwen35] prompt %d tokens\n", np);

    Cfg *c = &m.c;
    int ctx = np + n_gen + 8;
    State s; state_init(&m, &s, ctx);
    float *lg = falloc(c->V);
    uint64_t rng = 0x243F6A8885A308D3ULL;

    double tp0 = now_s();
    forward(&m, &s, ids, np, 0, lg, 0);
    double pre = now_s() - tp0;
    fprintf(stderr, "[qwen35] prefill %.2fs (%.1f tok/s)\n", pre, np / pre);

    double td0 = now_s();
    char buf[512];
    for (int i = 0; i < n_gen; i++) {
        int nxt = temp > 0 ? sample_logits(lg, c->V, temp, top_p)
                           : ({ int a = 0; for (int v = 1; v < c->V; v++) if (lg[v] > lg[a]) a = v; a; });
        (void)rng;
        int one = nxt;
        if (tok_decode(&T, &one, 1, buf, sizeof(buf)) > 0) { fputs(buf, stdout); fflush(stdout); }
        if (i + 1 < n_gen) forward(&m, &s, &nxt, 1, np + i, lg, 0);
    }
    double dec = now_s() - td0;
    printf("\n");
    fprintf(stderr, "[qwen35] decode %d tok in %.2fs (%.2f tok/s)\n", n_gen, dec, n_gen / dec);
    state_free(&m, &s);
    return 0;
}
