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
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "compat.h"
#include "json.h"
#include "moe_math.h"
#include "moe_matmul.h"
#include "moe_quant.h"
#include "moe_attn.h"
#include "moe_linattn.h"
#include "moe_sample.h"
#include "moe_serve.h"
#include "st.h"
#include "tok.h"

#ifdef COLI_CUDA
/* The MoE/GEMM kernels in backend_cuda_laguna.{cu,h} are arch-generic -- bf16 GEMM,
 * int4 row-scaled GEMM, and the fused/grouped expert paths take device pointers and
 * dimensions, nothing Laguna-specific. Reused here rather than duplicated; the file
 * keeps its name so laguna.c's build is untouched. */
#include "backend_cuda_laguna.h"
#endif

#define MAXL 80
static int g_exact = 0;          /* double accumulation, for the oracle */
#ifdef COLI_CUDA
static int g_cuda = 0;           /* 1 = GPU tier active */
#endif
static double g_vram_gb = 0;     /* what we actually uploaded */

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
    int eos[8], n_eos;                         /* stop tokens (generation_config.json) */
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
    /* VRAM copies; NULL = this tensor stays on the CPU path */
    void *d_wq, *d_wk, *d_wv, *d_wo;
    void *d_w_qkv, *d_w_z, *d_w_b, *d_w_a, *d_w_out;
    void *d_sg, *d_su, *d_sd;
    void *d_gu_q, *d_dn_q;
    float *d_gu_s, *d_dn_s;
} Layer;

typedef struct {
    shards S;
    Cfg c;
    Layer L[MAXL];
    void *embed, *lm_head;
    void *d_lm_head;
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

    /* Stop tokens. generation_config.json is authoritative and lists BOTH <|im_end|>
     * (end of an assistant turn) and <|endoftext|>; config.json's single eos_token_id
     * would stop only on the latter, so a chat turn would never terminate. Accept a
     * scalar or an array from either file. */
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
/* resident GEMM. Wdev is the weight's VRAM copy (NULL = CPU only). The GPU path is
 * skipped under g_exact so the oracle keeps its double-accumulate reference. */
static void resmm(Model *m, float *y, const float *x, const void *W, const void *Wdev,
                  int S, int I, int O) {
#ifdef COLI_CUDA
    if (Wdev && !g_exact && lag_cuda_matmul_bf16(y, x, Wdev, S, I, O) == 0) return;
#else
    (void)Wdev;
#endif
    if (m->res_dt == 1) matmul_bf16_k(y, x, (const uint16_t *)W, S, I, O, 0, g_exact);
    else matmul_f32(y, x, (const float *)W, S, I, O, g_exact);
}

/* Upload a bf16 resident to VRAM. Only bf16 (the real container); the f32 tiny oracle
 * stays on the CPU, which is also where its exactness contract lives. */
static void *dev_up(Model *m, const char *name, const void *host) {
#ifdef COLI_CUDA
    if (!g_cuda || m->res_dt != 1) return NULL;
    int64_t ne = st_numel(&m->S, name);
    if (ne <= 0) return NULL;
    void *d = lag_cuda_upload(host, (size_t)ne * 2);
    if (d) g_vram_gb += (double)ne * 2 / 1e9;
    return d;
#else
    (void)m; (void)name; (void)host; return NULL;
#endif
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
    /* lm_head is a real D->V GEMM and the single widest read per token (248320x2048 bf16
     * = 1.0 GB), so it earns VRAM. embed_tokens stays host-side: it is a row gather. */
    m->d_lm_head = dev_up(m, st_has(&m->S, "lm_head.weight") ? "lm_head.weight"
                                                             : "embed_tokens.weight", m->lm_head);

    for (int i = 0; i < c->n_layers; i++) {
        Layer *L = &m->L[i];
        snprintf(base, sizeof(base), "layers.%d.input_layernorm.weight", i);
        L->ln1 = load_norm(m, NM(nm, "%s", base));
        snprintf(base, sizeof(base), "layers.%d.post_attention_layernorm.weight", i);
        L->ln2 = load_norm(m, NM(nm, "%s", base));

        if (c->is_linear[i]) {
            const char *sfx[] = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"};
            void **dst[] = {&L->w_qkv, &L->w_z, &L->w_b, &L->w_a, &L->w_out};
            void **ddst[] = {&L->d_w_qkv, &L->d_w_z, &L->d_w_b, &L->d_w_a, &L->d_w_out};
            for (int j = 0; j < 5; j++) {
                snprintf(base, sizeof(base), "layers.%d.linear_attn.%s.weight", i, sfx[j]);
                *dst[j] = load_res(m, NM(nm, "%s", base));
                *ddst[j] = dev_up(m, nm, *dst[j]);
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
            void **ddst[] = {&L->d_wq, &L->d_wk, &L->d_wv, &L->d_wo};
            for (int j = 0; j < 4; j++) {
                snprintf(base, sizeof(base), "layers.%d.self_attn.%s.weight", i, sfx[j]);
                *dst[j] = load_res(m, NM(nm, "%s", base));
                *ddst[j] = dev_up(m, nm, *dst[j]);
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
        void **sddst[] = {&L->d_sg, &L->d_su, &L->d_sd};
        for (int j = 0; j < 3; j++) {
            snprintf(base, sizeof(base), "layers.%d.mlp.shared_expert.%s.weight", i, ssfx[j]);
            *sdst[j] = load_res(m, NM(nm, "%s", base));
            *sddst[j] = dev_up(m, nm, *sdst[j]);
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
#ifdef COLI_CUDA
            /* The whole int4 container is ~21.7 GB against 48 GB of VRAM, so every layer's
             * experts fit -- no cache, no pager, no CPU expert tier. That is the structural
             * difference from Laguna, whose 57 GB of experts do not fit and whose resulting
             * CPU fallback costs 70% of expert time. Partial upload still degrades safely:
             * a NULL device pointer just routes that layer through the CPU kernel. */
            if (g_cuda) {
                int64_t gub = st_nbytes(&m->S, NM(nm, "layers.%d.mlp.experts.gate_up_proj", i));
                int64_t gus = st_nbytes(&m->S, NM(nm, "layers.%d.mlp.experts.gate_up_proj.qs", i));
                int64_t dnb = st_nbytes(&m->S, NM(nm, "layers.%d.mlp.experts.down_proj", i));
                int64_t dns = st_nbytes(&m->S, NM(nm, "layers.%d.mlp.experts.down_proj.qs", i));
                size_t need = (size_t)(gub + gus + dnb + dns);
                if (lag_cuda_free_bytes() > need + (size_t)1e9) {
                    L->d_gu_q = lag_cuda_upload(L->gu_q, (size_t)gub);
                    L->d_gu_s = (float *)lag_cuda_upload(L->gu_s, (size_t)gus);
                    L->d_dn_q = lag_cuda_upload(L->dn_q, (size_t)dnb);
                    L->d_dn_s = (float *)lag_cuda_upload(L->dn_s, (size_t)dns);
                    if (L->d_gu_q && L->d_gu_s && L->d_dn_q && L->d_dn_s) g_vram_gb += need / 1e9;
                    else { L->d_gu_q = L->d_dn_q = NULL; L->d_gu_s = L->d_dn_s = NULL; }
                }
            }
#endif
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
/* Per-sequence memory: KV for the full-attention layers grows with context, the recurrent
 * state and conv window of the linear layers do not. Reporting both makes the architecture's
 * whole point visible at a glance -- and caps how many slots fit. */
static double state_bytes(Model *m, int ctx) {
    Cfg *c = &m->c;
    double kv = 0, rec = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_linear[i])
            rec += (double)linattn_state_bytes(c->lin_vh, c->lin_kd, c->lin_vd,
                                               c->conv_dim, c->conv_k);
        else
            kv += 2.0 * ctx * c->n_kv * c->head_dim * sizeof(float);
    }
    return kv + rec;
}

static void state_free(Model *m, State *s) {
    for (int i = 0; i < m->c.n_layers; i++) {
        free(s->k[i]); free(s->v[i]);
        free(s->ls[i].state); free(s->ls[i].conv);
    }
    free(s->k); free(s->v); free(s->ls);
}

/* ---------- profiling ----------
 * Decode in an agentic loop is the case that matters and the one that is hardest to reason
 * about, because the cost mix changes with context length: the MoE and the projections are
 * flat per token, the 10 full-attention layers grow linearly, and the 30 recurrent layers
 * are flat by construction. QW_PROF=1 attributes a run so that is measurable rather than
 * argued about. Phases sum to a checked fraction of wall time so a missing one is visible. */
static int g_prof = 0;
/* QW_NOGROUP=1 forces the per-head SDPA path in non-exact mode, so the grouped kernel has
 * something to be A/B'd against in ONE process on ONE build (the two use different softmax
 * structures, so greedy-token identity is the bar, not bit-identity). */
static int g_nogroup = 0;
/* QW_SPLIT=N overrides the KV-split chunk count. Two different N are both correct by
 * construction (kernel-check proves the merge is exact algebra), so comparing them
 * calibrates how much greedy divergence is pure float noise -- the control that tells a
 * real kernel bug apart from reassociation amplified through 40 layers. */
static int g_split_override = 0;
/* QW_ATTN_DBL=1 keeps the per-head path but accumulates QK in double. Both precisions are
 * correct, so this is the second benign control: it measures how far a legitimate change of
 * accumulation order alone moves a greedy continuation. */
static int g_attn_dbl = 0;
static double pr_qkv, pr_attn, pr_gate, pr_lin_proj, pr_lin_conv, pr_lin_scan, pr_lin_out,
              pr_route, pr_rgemm, pr_experts, pr_shared, pr_lmhead, pr_sample;
#define PROF_T0() (g_prof ? now_s() : 0.0)
#define PROF_ADD(acc, t0) do { if (g_prof) (acc) += now_s() - (t0); } while (0)

static void prof_report(const char *what, double wall, int steps) {
    if (!g_prof || steps <= 0) return;
    struct { const char *n; double v; } ph[] = {
        {"qkv+rope proj", pr_qkv}, {"attention (SDPA)", pr_attn}, {"gate+o_proj", pr_gate},
        {"linattn proj", pr_lin_proj}, {"linattn conv", pr_lin_conv},
        {"linattn recurrence", pr_lin_scan}, {"linattn norm+out", pr_lin_out},
        {"router GEMM (cpu)", pr_rgemm}, {"routing top-k+sort", pr_route}, {"experts", pr_experts}, {"shared expert", pr_shared},
        {"lm_head", pr_lmhead}, {"sampling", pr_sample},
    };
    double tot = 0;
    for (unsigned i = 0; i < sizeof(ph) / sizeof(ph[0]); i++) tot += ph[i].v;
    fprintf(stderr, "[prof] %s: %.2fs wall over %d steps (%.1f ms/step); phases sum to "
            "%.2fs (%.0f%%)\n", what, wall, steps, wall / steps * 1e3, tot,
            wall > 0 ? tot / wall * 100 : 0);
    for (unsigned i = 0; i < sizeof(ph) / sizeof(ph[0]); i++)
        fprintf(stderr, "[prof]   %-20s %7.2fs %6.1f%%  %8.2f ms/step\n", ph[i].n, ph[i].v,
                wall > 0 ? ph[i].v / wall * 100 : 0, ph[i].v / steps * 1e3);
}
static void prof_reset(void) {
    pr_qkv = pr_attn = pr_gate = pr_lin_proj = pr_lin_conv = pr_lin_scan = pr_lin_out =
        pr_route = pr_rgemm = pr_experts = pr_shared = pr_lmhead = pr_sample = 0;
}

/* ---------- MoE ---------- */
/* softmax over ALL experts, then top-k, then renormalize by the top-k sum. Order matters:
 * top-k-then-softmax gives different weights and is the common way to get this wrong. */
/* softmax over ALL experts, then top-k, then renormalize by the top-k sum. Order matters:
 * top-k-then-softmax gives different weights and is the common way to get this wrong.
 *
 * Called once per (row, layer) -- 133k times for a 3321-token prefill at 40 layers -- so the
 * constants matter. The first version scanned `sel` to test "already taken", making selection
 * O(K^2 E) (16k comparisons per row at E=256, K=8), and malloc'd the probability buffer per
 * call. Both showed up: routing measured 11% of prefill, more than the 256-expert MoE itself.
 * A used-mask makes it O(K E) and the buffers come off the stack. Bit-identical: the scan is
 * still ascending with a strict >, so ties still resolve to the lower index like torch.topk. */
#define MOE_ROUTE_STACK_E 1024
static void moe_route(const float *logit, int E, int K, int *sel, float *w) {
    float ps[MOE_ROUTE_STACK_E];
    unsigned char us[MOE_ROUTE_STACK_E];
    float *p = ps;
    unsigned char *used = us;
    void *heap = NULL;
    if (E > MOE_ROUTE_STACK_E) {
        heap = malloc((size_t)E * (sizeof(float) + 1));
        if (!heap) { fprintf(stderr, "OOM routing E=%d\n", E); exit(1); }
        p = (float *)heap;
        used = (unsigned char *)(p + E);
    }
    float mx = -INFINITY;
    for (int e = 0; e < E; e++) if (logit[e] > mx) mx = logit[e];
    double sum = 0;
    for (int e = 0; e < E; e++) { p[e] = expf(logit[e] - mx); sum += p[e]; }
    for (int e = 0; e < E; e++) p[e] = (float)(p[e] / sum);
    memset(used, 0, (size_t)E);
    for (int a = 0; a < K; a++) {
        int best = -1;
        for (int e = 0; e < E; e++) {
            if (used[e]) continue;
            if (best < 0 || p[e] > p[best]) best = e;
        }
        used[best] = 1;
        sel[a] = best; w[a] = p[best];
    }
    double tot = 0;
    for (int a = 0; a < K; a++) tot += w[a];
    for (int a = 0; a < K; a++) w[a] = (float)(w[a] / tot);
    free(heap);
}

/* Routed MoE + shared expert over S token rows.
 *
 * Grouped by expert (Megablocks-style): the S*K (row, expert) pairs are counting-sorted by
 * expert id, so each DISTINCT expert's int4 blob is read once and applied to all its rows as
 * a GEMM. At S=1 that degenerates to the per-expert path; the win is prefill and batched
 * decode, where B streams sharing an expert pay for it once.
 *
 * Bit-exact against the per-row path by construction, which is why BATCH=N can assert token
 * identity: each row's GEMM is an independent per-row reduction, and the combine sums over
 * ranks in top-k order regardless of how rows were packed. */
static void moe_forward_rows(Model *m, Layer *L, const float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->D, I = c->moe_I, E = c->E, K = c->K, SI = c->shared_I;
    int64_t T = (int64_t)S * K;

    double trg = PROF_T0();
    float *logit = falloc((int64_t)S * E);
    matmul_f32(logit, x, L->router, S, D, E, g_exact);
    PROF_ADD(pr_rgemm, trg);
    double tr = PROF_T0();
    int *sel = malloc(sizeof(int) * T);
    float *w = falloc(T);
    for (int s = 0; s < S; s++)
        moe_route(logit + (int64_t)s * E, E, K, sel + (int64_t)s * K, w + (int64_t)s * K);

    /* counting-sort the (row, expert) pairs into per-expert groups, ascending expert id */
    int *cnt = calloc((size_t)E, sizeof(int));
    for (int64_t i = 0; i < T; i++) cnt[sel[i]]++;
    int *gexp = malloc((size_t)E * sizeof(int)), *gbeg = malloc((size_t)E * sizeof(int));
    int *gcnt = malloc((size_t)E * sizeof(int)), *pos = malloc((size_t)E * sizeof(int));
    int G = 0, off = 0;
    for (int e = 0; e < E; e++) {
        if (!cnt[e]) continue;
        gexp[G] = e; gbeg[G] = off; gcnt[G] = cnt[e]; pos[e] = off; off += cnt[e]; G++;
    }
    int *grow = malloc((size_t)T * sizeof(int));      /* packed row -> source row */
    int *rowof = malloc((size_t)T * sizeof(int));     /* (row, rank) -> packed row */
    for (int s = 0; s < S; s++)
        for (int a = 0; a < K; a++) {
            int slot = pos[sel[(int64_t)s * K + a]]++;
            grow[slot] = s; rowof[(int64_t)s * K + a] = slot;
        }
    PROF_ADD(pr_route, tr);

    double tx = PROF_T0();
    int done = 0;
#ifdef COLI_CUDA
    if (m->xq && L->d_gu_q && !g_exact &&
        lag_cuda_moe_group(out, x, L->d_gu_q, L->d_gu_s, L->d_dn_q, L->d_dn_s,
                           gexp, gbeg, gcnt, G, grow, rowof, w, S, K, I, D) == 0) done = 1;
#endif
    if (!done) {
        float *slots = falloc(T * D);
        float *xg = falloc((int64_t)S * D), *gu = falloc((int64_t)S * 2 * I);
        float *glu = falloc((int64_t)S * I);
        for (int g = 0; g < G; g++) {
            int e = gexp[g], ng = gcnt[g], beg = gbeg[g];
            for (int j = 0; j < ng; j++)
                memcpy(xg + (int64_t)j * D, x + (int64_t)grow[beg + j] * D,
                       (size_t)D * sizeof(float));
            if (m->xq) {
                matmul_q4_kb(gu, xg, L->gu_q + (int64_t)(e * 2 * I) * (D / 2),
                             L->gu_s + (int64_t)e * 2 * I, ng, D, 2 * I, MOE_Q4_F32, g_exact);
                for (int j = 0; j < ng; j++) {
                    const float *gr = gu + (int64_t)j * 2 * I;
                    float *o = glu + (int64_t)j * I;
                    for (int i = 0; i < I; i++) o[i] = siluf(gr[i]) * gr[I + i];
                }
                matmul_q4_kb(slots + (int64_t)beg * D, glu,
                             L->dn_q + (int64_t)(e * D) * (I / 2), L->dn_s + (int64_t)e * D,
                             ng, I, D, MOE_Q4_F32, g_exact);
            } else {
                float *gg = gu, *uu = gu + (int64_t)ng * I;
                matmul_f32(gg, xg, L->eg[e], ng, D, I, g_exact);
                matmul_f32(uu, xg, L->eu[e], ng, D, I, g_exact);
                for (int64_t i = 0; i < (int64_t)ng * I; i++) glu[i] = siluf(gg[i]) * uu[i];
                matmul_f32(slots + (int64_t)beg * D, glu, L->ed[e], ng, I, D, g_exact);
            }
        }
        /* combine in RANK order, so the sum matches the per-token path element for element */
        for (int s = 0; s < S; s++) {
            float *o = out + (int64_t)s * D;
            for (int i = 0; i < D; i++) o[i] = 0.f;
            for (int a = 0; a < K; a++) {
                const float *v = slots + (int64_t)rowof[(int64_t)s * K + a] * D;
                float wa = w[(int64_t)s * K + a];
                for (int i = 0; i < D; i++) o[i] += wa * v[i];
            }
        }
        free(slots); free(xg); free(gu); free(glu);
    }
    PROF_ADD(pr_experts, tx);

    /* shared expert over all S rows (its residents are read once), scaled per row by
     * sigmoid of its own 1-wide gate */
    double tsh = PROF_T0();
    float *sg = falloc((int64_t)S * SI), *su = falloc((int64_t)S * SI);
    float *so = falloc((int64_t)S * D), *gv = falloc((int64_t)S);
    matmul_f32(gv, x, L->sgate, S, D, 1, g_exact);
    resmm(m, sg, x, L->sg, L->d_sg, S, D, SI);
    resmm(m, su, x, L->su, L->d_su, S, D, SI);
    for (int64_t i = 0; i < (int64_t)S * SI; i++) sg[i] = siluf(sg[i]) * su[i];
    resmm(m, so, sg, L->sd, L->d_sd, S, SI, D);
    for (int s = 0; s < S; s++) {
        float gs = 1.f / (1.f + expf(-gv[s]));
        float *o = out + (int64_t)s * D;
        const float *sr = so + (int64_t)s * D;
        for (int i = 0; i < D; i++) o[i] += gs * sr[i];
    }
    PROF_ADD(pr_shared, tsh);

    free(logit); free(sel); free(w); free(cnt);
    free(gexp); free(gbeg); free(gcnt); free(pos); free(grow); free(rowof);
    free(sg); free(su); free(so); free(gv);
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
    double tq = PROF_T0();
    resmm(m, qp, xn, L->wq, L->d_wq, n, D, qw);
    resmm(m, kp, xn, L->wk, L->d_wk, n, D, kvw);
    resmm(m, vp, xn, L->wv, L->d_wv, n, D, kvw);
    PROF_ADD(pr_qkv, tq);

    float *ao = falloc((int64_t)n * H * hd);
    double ta = PROF_T0();

    /* Pass 1: write every token's K/V (per-head RMSNorm then rope) before attending.
     * Separating this from the attention pass is what lets pass 2 be parallel: with K/V
     * already in place there is no dependency between (token, kv-head) work items. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 1)
#endif
    for (int t = 0; t < n; t++) {
        int pos = pos0 + t;
        float *kdst = s->k[li] + (int64_t)pos * kvw, *vdst = s->v[li] + (int64_t)pos * kvw;
        for (int h = 0; h < nkv; h++) {
            rmsnorm_row(kdst + h * hd, kp + (int64_t)t * kvw + h * hd, L->kn, hd, c->eps);
            rope_apply(kdst + h * hd, pos, c->inv, c->rdim);
        }
        memcpy(vdst, vp + (int64_t)t * kvw, (size_t)kvw * sizeof(float));
    }

    /* Pass 2: attention.
     * Work items are (token, kv-head) pairs. In PREFILL that is n*n_kv items -- plenty, so
     * parallelize across them and use the plain grouped kernel. In DECODE there is one token
     * and n_kv=2, which cannot fill 12 cores; there the parallelism has to come from splitting
     * the KV window instead (sdpa_group_split). Both read each KV element exactly once.
     * The oracle keeps the scalar per-head path, which is what its exactness is validated on. */
    int items = n * nkv;
    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads();
#endif
    (void)nthr;
    /* RULED OUT, with numbers: splitting the KV window (sdpa_group_split, flash-decoding
     * split-K) is 2.5x SLOWER here than not splitting, despite reading the same bytes.
     * Measured on this box at 3321-token context, same binary, back to back:
     *     QW_SPLIT=1  22.8 tok/s   attention 16.7 ms/step
     *     QW_SPLIT=2   9.0 tok/s   attention 52.0 ms/step
     *     QW_SPLIT=6   9.7 tok/s   attention 45.8 ms/step
     * The cause is OpenMP region overhead, not memory: with split>1 the (token, kv-head)
     * loop must run serially and the split kernel opens a fresh parallel region per call --
     * 20 regions per token on a 24-thread team, each with only 2-6 chunks of real work.
     * Amortizing that needs ONE region per layer spanning (token, kv-head, chunk) with the
     * merge outside, which is the shape to build if decode attention needs more than the
     * 2-way parallelism the (token, kv-head) loop gives at batch 1. The kernel stays in
     * moe_attn.h, validated by kernel-check, reachable via QW_SPLIT for that work. */
    int split = 1;
    if (g_split_override > 0 && !g_exact && !g_nogroup && items < nthr) split = g_split_override;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2) if (items > 1 && split == 1)
#endif
    for (int t = 0; t < n; t++) {
        for (int kh = 0; kh < nkv; kh++) {
            int pos = pos0 + t;
            float qg[SDPA_MAX_GROUP * 512];
            for (int j = 0; j < grp; j++) {
                int h = kh * grp + j;
                /* query and gate are interleaved per head: [q(hd) | gate(hd)] per head */
                const float *qsrc = qp + (int64_t)t * qw + (int64_t)h * hd * 2;
                rmsnorm_row(qg + (int64_t)j * hd, qsrc, L->qn, hd, c->eps);
                rope_apply(qg + (int64_t)j * hd, pos, c->inv, c->rdim);
            }
            float *og = ao + (int64_t)t * H * hd + (int64_t)kh * grp * hd;
            const float *Kb = s->k[li] + (int64_t)kh * hd;
            const float *Vb = s->v[li] + (int64_t)kh * hd;
            if (g_exact || g_nogroup) {
                float *scl = falloc(pos + 1);
                for (int j = 0; j < grp; j++)
                    sdpa_head(qg + (int64_t)j * hd, Kb, Vb, kvw, hd, 0, pos, scale, 1.f,
                              NULL, 0, (g_exact || g_attn_dbl) ? MOE_QK_DBL : MOE_QK_SIMD,
                              og + (int64_t)j * hd, scl, 0);
                free(scl);
            } else if (split > 1) {
                sdpa_group_split(qg, hd, Kb, Vb, kvw, hd, 0, pos, scale, 1.f, og, hd, grp,
                                 0, split, NULL);
            } else {
                sdpa_group(qg, hd, Kb, Vb, kvw, hd, 0, pos, scale, 1.f, og, hd, grp, 0);
            }
            /* per-element output gate, applied to each head's own slice */
            for (int j = 0; j < grp; j++) {
                const float *gate = qp + (int64_t)t * qw + (int64_t)(kh * grp + j) * hd * 2 + hd;
                float *o = og + (int64_t)j * hd;
                for (int d = 0; d < hd; d++) o[d] *= 1.f / (1.f + expf(-gate[d]));
            }
        }
    }
    PROF_ADD(pr_attn, ta);
    double tg = PROF_T0();
    resmm(m, out, ao, L->wo, L->d_wo, n, H * hd, D);
    PROF_ADD(pr_gate, tg);
    free(qp); free(kp); free(vp); free(ao);
}

/* gated delta net over `n` tokens; state and conv window carry across calls */
static void linattn_forward(Model *m, Layer *L, State *s, int li, const float *xn, int n,
                            float *out) {
    Cfg *c = &m->c;
    int D = c->D, cd = c->conv_dim, vdim = c->value_dim, kdim = c->key_dim, nv = c->lin_vh;

    float *mix = falloc((int64_t)n * cd);
    float *z = falloc((int64_t)n * vdim);
    float *bb = falloc((int64_t)n * nv), *aa = falloc((int64_t)n * nv);
    double tp = PROF_T0();
    resmm(m, mix, xn, L->w_qkv, L->d_w_qkv, n, D, cd);
    resmm(m, z, xn, L->w_z, L->d_w_z, n, D, vdim);
    resmm(m, bb, xn, L->w_b, L->d_w_b, n, D, nv);
    resmm(m, aa, xn, L->w_a, L->d_w_a, n, D, nv);
    PROF_ADD(pr_lin_proj, tp);

    double tc = PROF_T0();
    linattn_conv_span(s->ls[li].conv, mix, L->conv_w, NULL, mix, n, cd, c->conv_k);

    PROF_ADD(pr_lin_conv, tc);

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
    double ts = PROF_T0();
    linattn_scan(&s->ls[li], q, k, v, g, bb, core, n, c->lin_kh, nv, c->lin_kd, c->lin_vd);
    PROF_ADD(pr_lin_scan, ts);

    /* gated RMSNorm per value head, then out_proj */
    double to = PROF_T0();
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nv; h++)
            linattn_gated_norm(core + (int64_t)t * vdim + h * c->lin_vd,
                               z + (int64_t)t * vdim + h * c->lin_vd, L->gnorm, c->lin_vd, c->eps);
    resmm(m, out, core, L->w_out, L->d_w_out, n, vdim, D);
    PROF_ADD(pr_lin_out, to);

    free(mix); free(z); free(bb); free(aa); free(g);
    free(q); free(k); free(v); free(core);
}

/* ---------- forward ---------- */
/* Runs `n` tokens at absolute positions pos0..pos0+n-1. If `logits_all`, writes logits for
 * every position (teacher forcing); otherwise only the last. */
/* ---------- batched decode ----------
 * One token for each of B streams, in lockstep. This is the throughput lever and it also
 * fixes decode's parallelism problem: attention work items go from n_kv_heads (2 at batch 1,
 * which cannot fill 12 cores) to B*n_kv_heads, and the MoE groups B rows so an expert shared
 * by several streams is read once.
 *
 * Every stream keeps its OWN state -- KV for the 10 attention layers, recurrent state plus
 * conv window for the 30 linear ones -- and its own position, so streams of different lengths
 * batch together. Per-stream arithmetic is identical to the single-stream path, which is what
 * BATCH=N asserts. */
static void attn_batch(Model *m, Layer *L, State **st, int li, const float *xn,
                       const int *pos, int B, float *out) {
    Cfg *c = &m->c;
    int D = c->D, H = c->n_heads, hd = c->head_dim, nkv = c->n_kv;
    int qw = H * hd * 2, kvw = nkv * hd, grp = H / nkv;
    float scale = 1.f / sqrtf((float)hd);

    double tq = PROF_T0();
    float *qp = falloc((int64_t)B * qw);
    float *kp = falloc((int64_t)B * kvw), *vp = falloc((int64_t)B * kvw);
    resmm(m, qp, xn, L->wq, L->d_wq, B, D, qw);
    resmm(m, kp, xn, L->wk, L->d_wk, B, D, kvw);
    resmm(m, vp, xn, L->wv, L->d_wv, B, D, kvw);
    PROF_ADD(pr_qkv, tq);

    float *ao = falloc((int64_t)B * H * hd);
    double ta = PROF_T0();
    for (int b = 0; b < B; b++) {              /* each stream writes into its own cache */
        float *kdst = st[b]->k[li] + (int64_t)pos[b] * kvw;
        float *vdst = st[b]->v[li] + (int64_t)pos[b] * kvw;
        for (int h = 0; h < nkv; h++) {
            rmsnorm_row(kdst + h * hd, kp + (int64_t)b * kvw + h * hd, L->kn, hd, c->eps);
            rope_apply(kdst + h * hd, pos[b], c->inv, c->rdim);
        }
        memcpy(vdst, vp + (int64_t)b * kvw, (size_t)kvw * sizeof(float));
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2) if (B * nkv > 1)
#endif
    for (int b = 0; b < B; b++) {
        for (int kh = 0; kh < nkv; kh++) {
            float qg[SDPA_MAX_GROUP * 512];
            for (int j = 0; j < grp; j++) {
                const float *qsrc = qp + (int64_t)b * qw + (int64_t)(kh * grp + j) * hd * 2;
                rmsnorm_row(qg + (int64_t)j * hd, qsrc, L->qn, hd, c->eps);
                rope_apply(qg + (int64_t)j * hd, pos[b], c->inv, c->rdim);
            }
            float *og = ao + (int64_t)b * H * hd + (int64_t)kh * grp * hd;
            const float *Kb = st[b]->k[li] + (int64_t)kh * hd;
            const float *Vb = st[b]->v[li] + (int64_t)kh * hd;
            if (g_exact || g_nogroup) {
                float *scl = falloc(pos[b] + 1);
                for (int j = 0; j < grp; j++)
                    sdpa_head(qg + (int64_t)j * hd, Kb, Vb, kvw, hd, 0, pos[b], scale, 1.f,
                              NULL, 0, g_exact ? MOE_QK_DBL : MOE_QK_SIMD,
                              og + (int64_t)j * hd, scl, 0);
                free(scl);
            } else {
                sdpa_group(qg, hd, Kb, Vb, kvw, hd, 0, pos[b], scale, 1.f, og, hd, grp, 0);
            }
            for (int j = 0; j < grp; j++) {
                const float *gate = qp + (int64_t)b * qw
                                  + (int64_t)(kh * grp + j) * hd * 2 + hd;
                float *o = og + (int64_t)j * hd;
                for (int d = 0; d < hd; d++) o[d] *= 1.f / (1.f + expf(-gate[d]));
            }
        }
    }
    PROF_ADD(pr_attn, ta);
    double tg = PROF_T0();
    resmm(m, out, ao, L->wo, L->d_wo, B, H * hd, D);
    PROF_ADD(pr_gate, tg);
    free(qp); free(kp); free(vp); free(ao);
}

static void linattn_batch(Model *m, Layer *L, State **st, int li, const float *xn,
                          int B, float *out) {
    Cfg *c = &m->c;
    int D = c->D, cd = c->conv_dim, vdim = c->value_dim, kdim = c->key_dim, nv = c->lin_vh;

    double tp = PROF_T0();
    float *mix = falloc((int64_t)B * cd), *z = falloc((int64_t)B * vdim);
    float *bb = falloc((int64_t)B * nv), *aa = falloc((int64_t)B * nv);
    resmm(m, mix, xn, L->w_qkv, L->d_w_qkv, B, D, cd);
    resmm(m, z, xn, L->w_z, L->d_w_z, B, D, vdim);
    resmm(m, bb, xn, L->w_b, L->d_w_b, B, D, nv);
    resmm(m, aa, xn, L->w_a, L->d_w_a, B, D, nv);
    PROF_ADD(pr_lin_proj, tp);

    /* conv window and recurrent state are PER STREAM, so these stay per-stream calls even
     * though the projections above batched. Each is one token, so the conv is a shift. */
    double tc = PROF_T0();
    for (int b = 0; b < B; b++)
        linattn_conv_span(st[b]->ls[li].conv, mix + (int64_t)b * cd, L->conv_w, NULL,
                          mix + (int64_t)b * cd, 1, cd, c->conv_k);
    PROF_ADD(pr_lin_conv, tc);

    float *g = falloc((int64_t)B * nv);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < nv; h++) {
            int64_t o = (int64_t)b * nv + h;
            g[o] = -expf(L->A_log[h]) * linattn_softplus(aa[o] + L->dt_bias[h]);
            bb[o] = 1.f / (1.f + expf(-bb[o]));
        }

    float *core = falloc((int64_t)B * vdim);
    double ts = PROF_T0();
    for (int b = 0; b < B; b++) {
        const float *r = mix + (int64_t)b * cd;
        linattn_scan(&st[b]->ls[li], r, r + kdim, r + 2 * kdim,
                     g + (int64_t)b * nv, bb + (int64_t)b * nv,
                     core + (int64_t)b * vdim, 1, c->lin_kh, nv, c->lin_kd, c->lin_vd);
    }
    PROF_ADD(pr_lin_scan, ts);

    double to = PROF_T0();
    for (int b = 0; b < B; b++)
        for (int h = 0; h < nv; h++)
            linattn_gated_norm(core + (int64_t)b * vdim + h * c->lin_vd,
                               z + (int64_t)b * vdim + h * c->lin_vd, L->gnorm,
                               c->lin_vd, c->eps);
    resmm(m, out, core, L->w_out, L->d_w_out, B, vdim, D);
    PROF_ADD(pr_lin_out, to);
    free(mix); free(z); free(bb); free(aa); free(g); free(core);
}

/* logits[B][V] */
static void forward_batch(Model *m, State **st, const int *tok, const int *pos, int B,
                          float *logits) {
    Cfg *c = &m->c;
    int D = c->D, V = c->V;
    float *x = falloc((int64_t)B * D), *xn = falloc((int64_t)B * D);
    for (int b = 0; b < B; b++) {
        if (m->res_dt == 1) {
            const uint16_t *e = (const uint16_t *)m->embed + (int64_t)tok[b] * D;
            for (int i = 0; i < D; i++) x[(int64_t)b * D + i] = bf16_f32(e[i]);
        } else {
            memcpy(x + (int64_t)b * D, (const float *)m->embed + (int64_t)tok[b] * D,
                   (size_t)D * sizeof(float));
        }
    }
    float *mo = falloc((int64_t)B * D);
    for (int li = 0; li < c->n_layers; li++) {
        Layer *L = &m->L[li];
        for (int b = 0; b < B; b++)
            rmsnorm_row(xn + (int64_t)b * D, x + (int64_t)b * D, L->ln1, D, c->eps);
        if (c->is_linear[li]) linattn_batch(m, L, st, li, xn, B, mo);
        else                  attn_batch(m, L, st, li, xn, pos, B, mo);
        for (int64_t i = 0; i < (int64_t)B * D; i++) x[i] += mo[i];

        for (int b = 0; b < B; b++)
            rmsnorm_row(xn + (int64_t)b * D, x + (int64_t)b * D, L->ln2, D, c->eps);
        moe_forward_rows(m, L, xn, B, mo);
        for (int64_t i = 0; i < (int64_t)B * D; i++) x[i] += mo[i];
    }
    for (int b = 0; b < B; b++) st[b]->len = pos[b] + 1;

    double tl = PROF_T0();
    float *hn = falloc((int64_t)B * D);
    for (int b = 0; b < B; b++)
        rmsnorm_row(hn + (int64_t)b * D, x + (int64_t)b * D, m->norm, D, c->eps);
    resmm(m, logits, hn, m->lm_head, m->d_lm_head, B, D, V);
    PROF_ADD(pr_lmhead, tl);
    free(x); free(xn); free(mo); free(hn);
}

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
        moe_forward_rows(m, L, xn, n, mo);
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
        double tl = PROF_T0();
        resmm(m, logits + (int64_t)(logits_all ? t : 0) * V, hp, m->lm_head, m->d_lm_head, 1, D, V);
        PROF_ADD(pr_lmhead, tl);
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

/* ---------- serve mode: the openai_server.py gateway protocol ----------
 * stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n / CANCEL <id>
 * stdout: READY sentinel + STAT, then DATA <id> <n>\n<bytes>\n frames and
 *         DONE <id> STAT <tok> <tps> <hit%> <rss> <prompt_tok> <limited>\n
 * SReq, the submit queue, stdin_readable and serve_read_cmd are shared (moe_serve.h).
 *
 * Serial only: one request finishes before the next starts. qwen35.c has no continuous
 * batching (no slot pool, no group-by-expert GEMM), so `slot` is ignored and there is no
 * prefix reuse -- every request re-prefills. That is the honest state; see
 * docs/linear-attention.md for why the recurrent state makes reuse cheap but rewind
 * impossible, which is the design that has to land before batching is worth writing. */
static void serve_one(Model *m, Tok *T, SReq *q) {
    Cfg *c = &m->c;
    int cap = q->plen + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { printf("ERROR %s empty prompt\n", q->id); fflush(stdout); free(ids); return; }
    int ctx_max = getenv("CTX_MAX") ? atoi(getenv("CTX_MAX")) : 8192;
    if (np + q->max_tok > ctx_max) {
        printf("ERROR %s context exceeds CTX_MAX\n", q->id); fflush(stdout); free(ids); return; }

    State s; state_init(m, &s, np + q->max_tok + 8);
    float *lg = falloc(c->V);
    double t0 = now_s();
    forward(m, &s, ids, np, 0, lg, 0);
    char buf[512];
    int gen = 0, limited = 1, pos = np, cancelled = 0;
    for (int i = 0; i < q->max_tok && !cancelled; i++) {
        int tk = sample_logits(lg, c->V, q->temp, q->top_p);
        int is_eos = 0;
        for (int e = 0; e < c->n_eos; e++) if (tk == c->eos[e]) is_eos = 1;
        if (is_eos) { limited = 0; break; }
        int nb = tok_decode(T, &tk, 1, buf, sizeof(buf) - 1);
        printf("DATA %s %d\n", q->id, nb);
        fwrite(buf, 1, (size_t)nb, stdout); fputc('\n', stdout); fflush(stdout);
        gen++;
        forward(m, &s, &tk, 1, pos, lg, 0); pos++;
        while (stdin_readable()) {
            int r = serve_read_cmd(q->id);
            if (r < 0) { free(ids); free(lg); state_free(m, &s); return; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
    }
    double dt = now_s() - t0;
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen,
           dt > 0 ? gen / dt : 0.0, 100.0, rss_gb(), np, limited);
    printf("PROF %.3f %d %d 0.000 0.000 %.3f 0.000 0.000 %d\n", dt, np, gen, dt, gen + 1);
    fflush(stdout);
    free(ids); free(lg); state_free(m, &s);
}

/* ---------- continuous batching (KV_SLOTS > 1) ----------
 * The serial loop finishes one request before starting the next, so a second client waits out
 * the first and the aggregate throughput BATCH measures is unreachable from the gateway. Here
 * up to KV_SLOTS requests stay in flight: each owns a slot (KV for the attention layers,
 * recurrent state + conv window for the linear ones), new arrivals prefill while others decode,
 * and every ready slot advances one token per forward_batch.
 *
 * PREFIX REUSE, and why it is restricted here: an agent loop re-sends a growing context, so a
 * slot usually already holds a prefix of the next request. Reusing it skips that prefill --
 * which is worth a lot (a measured turn re-prefilled 4998 tokens to emit ONE token). But a
 * recurrent state cannot be rewound: the state after n tokens does not contain the state after
 * n-k, and the decay has already multiplied that information away. So reuse is allowed only
 * when the new prompt EXTENDS what the slot holds exactly (lcp == hist_len). A divergence
 * anywhere inside the held prefix forces a full recompute, even though the 10 attention layers
 * could have been truncated. See docs/linear-attention.md.
 *
 * The invariant that makes this safe -- and whose violation silently corrupted laguna's reuse
 * once -- is hist_len == the number of tokens actually fed through the model. History is
 * appended when a token is FED, never when it is emitted. */
typedef struct {
    int used, ready;
    char id[64];
    State st;
    int *hist, hist_len;            /* tokens whose K/V and recurrent state are in `st` */
    int *ids, np, np_done;          /* the request's full prompt, and how much is prefilled */
    int gen, max_tok, limited, cur;
    float temp, top_p;
    double t0;
} SSlot;

static int slot_prefix_reuse(SSlot *sl, const int *ids, int np) {
    if (!sl->hist_len || sl->hist_len > np) return 0;
    for (int i = 0; i < sl->hist_len; i++) if (sl->hist[i] != ids[i]) return 0;
    return sl->hist_len;            /* pure extension: the whole held prefix matches */
}

static void serve_loop_batched(Model *m, Tok *T, int nslots) {
    Cfg *c = &m->c;
    int ctx_max = getenv("CTX_MAX") ? atoi(getenv("CTX_MAX")) : 8192;
    int chunk = getenv("PREFILL_CHUNK") ? atoi(getenv("PREFILL_CHUNK")) : 256;
    if (chunk < 1) chunk = 1;
    SSlot *S = calloc(nslots, sizeof(SSlot));
    for (int i = 0; i < nslots; i++) {
        state_init(m, &S[i].st, ctx_max);
        S[i].hist = malloc(sizeof(int) * ctx_max);
    }
    float *lg = falloc((int64_t)nslots * c->V);
    State **sp = malloc(sizeof(State *) * nslots);
    int *pos = malloc(sizeof(int) * nslots), *tk = malloc(sizeof(int) * nslots);
    int *map = malloc(sizeof(int) * nslots);

    setvbuf(stdin, NULL, _IONBF, 0);
    const char *sd = getenv("SEED");
    g_rng ^= sd ? (uint64_t)strtoull(sd, NULL, 10) : (uint64_t)time(NULL) * 2654435761u;
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb());
    fflush(stdout);

    for (;;) {
        int busy = 0;
        for (int i = 0; i < nslots; i++) if (S[i].used) busy = 1;
        /* block for work only when nothing is in flight, else poll so decode keeps moving */
        if (!busy && !g_qn) { if (serve_read_cmd(NULL) < 0) break; }
        while (stdin_readable()) if (serve_read_cmd(NULL) < 0) goto done;

        /* --- admit --- */
        while (g_qn) {
            int free_i = -1;
            for (int i = 0; i < nslots && free_i < 0; i++) if (!S[i].used) free_i = i;
            if (free_i < 0) break;
            SReq q = g_q[0];
            memmove(g_q, g_q + 1, (size_t)(--g_qn) * sizeof(SReq));
            if (serve_take_cancel(q.id)) {
                printf("DONE %s STAT 0 0.000 100.0 %.2f 0 0\n", q.id, rss_gb());
                fflush(stdout); free(q.payload); continue;
            }
            int cap = q.plen + 16;
            int *ids = malloc((size_t)cap * sizeof(int));
            int np = tok_encode(T, q.payload, q.plen, ids, cap);
            free(q.payload);
            if (np <= 0 || np + q.max_tok > ctx_max) {
                printf("ERROR %s %s\n", q.id, np <= 0 ? "empty prompt" : "context exceeds CTX_MAX");
                fflush(stdout); free(ids); continue;
            }
            /* prefer the slot that already holds the longest usable prefix */
            int best = free_i, best_reuse = 0;
            for (int i = 0; i < nslots; i++) {
                if (S[i].used) continue;
                int r = slot_prefix_reuse(&S[i], ids, np);
                if (r > best_reuse) { best_reuse = r; best = i; }
            }
            SSlot *sl = &S[best];
            if (!best_reuse) {                       /* fresh state; nothing reusable */
                state_free(m, &sl->st);
                state_init(m, &sl->st, ctx_max);
                sl->hist_len = 0;
            }
            sl->used = 1; sl->ready = 0;
            snprintf(sl->id, sizeof(sl->id), "%s", q.id);
            sl->ids = ids; sl->np = np; sl->np_done = best_reuse;
            sl->gen = 0; sl->max_tok = q.max_tok; sl->limited = 1;
            sl->temp = q.temp; sl->top_p = q.top_p; sl->t0 = now_s();
            if (best_reuse)
                fprintf(stderr, "[qwen35] slot %d reused %d/%d prompt tokens\n",
                        best, best_reuse, np);
        }

        /* --- chunked prefill: one chunk per slot per pass, so decode is not starved --- */
        for (int i = 0; i < nslots; i++) {
            SSlot *sl = &S[i];
            if (!sl->used || sl->ready) continue;
            int take = sl->np - sl->np_done;
            if (take > chunk) take = chunk;
            forward(m, &sl->st, sl->ids + sl->np_done, take, sl->np_done,
                    lg + (int64_t)i * c->V, 0);
            for (int j = 0; j < take; j++) sl->hist[sl->hist_len++] = sl->ids[sl->np_done + j];
            sl->np_done += take;
            if (sl->np_done >= sl->np) sl->ready = 1;   /* logits for the last position are live */
        }

        /* --- one decode step for every ready slot, in lockstep --- */
        int B = 0;
        for (int i = 0; i < nslots; i++) {
            SSlot *sl = &S[i];
            if (!sl->used || !sl->ready) continue;
            const float *l = lg + (int64_t)i * c->V;
            int t = sample_logits(l, c->V, sl->temp, sl->top_p);
            int is_eos = 0;
            for (int e = 0; e < c->n_eos; e++) if (t == c->eos[e]) is_eos = 1;
            int cancelled = serve_take_cancel(sl->id);
            if (is_eos || cancelled || sl->gen >= sl->max_tok) {
                double dt = now_s() - sl->t0;
                printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", sl->id, sl->gen,
                       dt > 0 ? sl->gen / dt : 0.0, 100.0, rss_gb(), sl->np,
                       (is_eos || cancelled) ? 0 : sl->limited);
                fflush(stdout);
                free(sl->ids); sl->ids = NULL;
                sl->used = 0; sl->ready = 0;
                continue;
            }
            char buf[512];
            int nb = tok_decode(T, &t, 1, buf, sizeof(buf) - 1);
            printf("DATA %s %d\n", sl->id, nb);
            fwrite(buf, 1, (size_t)nb, stdout); fputc('\n', stdout); fflush(stdout);
            sl->gen++;
            sl->cur = t;
            sp[B] = &sl->st; tk[B] = t; pos[B] = sl->hist_len; map[B] = i;
            sl->hist[sl->hist_len++] = t;      /* fed below: history tracks FED tokens */
            B++;
        }
        if (B) {
            float *blg = falloc((int64_t)B * c->V);
            forward_batch(m, sp, tk, pos, B, blg);
            for (int b = 0; b < B; b++)
                memcpy(lg + (int64_t)map[b] * c->V, blg + (int64_t)b * c->V,
                       (size_t)c->V * sizeof(float));
            free(blg);
        }
    }
done:
    for (int i = 0; i < nslots; i++) { state_free(m, &S[i].st); free(S[i].hist); free(S[i].ids); }
    free(S); free(lg); free(sp); free(pos); free(tk); free(map);
}

static void serve_loop(Model *m, Tok *T) {
    setvbuf(stdin, NULL, _IONBF, 0);
    const char *sd = getenv("SEED");
    g_rng ^= sd ? (uint64_t)strtoull(sd, NULL, 10) : (uint64_t)time(NULL) * 2654435761u;
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb());
    fflush(stdout);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(NULL) < 0) return;
        SReq q = g_q[0];
        memmove(g_q, g_q + 1, (size_t)(--g_qn) * sizeof(SReq));
        if (serve_take_cancel(q.id)) {
            printf("DONE %s STAT 0 0.000 100.0 %.2f 0 0\n", q.id, rss_gb()); fflush(stdout);
        } else serve_one(m, T, &q);
        free(q.payload);
    }
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    if (getenv("EXACT")) g_exact = 1;
    if (getenv("QW_PROF")) g_prof = 1;
    if (getenv("QW_NOGROUP")) g_nogroup = 1;
    if (getenv("QW_SPLIT")) g_split_override = atoi(getenv("QW_SPLIT"));
    if (getenv("QW_ATTN_DBL")) g_attn_dbl = 1;

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
#ifdef COLI_CUDA
    /* NOCUDA=1 forces the CPU path for an in-process A/B against the GPU tier. */
    if (!getenv("NOCUDA")) {
        int dev = getenv("CUDA_DEV") ? atoi(getenv("CUDA_DEV")) : 0;
        g_cuda = (lag_cuda_init(dev) == 0);
        if (g_cuda)
            fprintf(stderr, "[qwen35] CUDA device %d: %.1f GB free of %.1f GB\n", dev,
                    lag_cuda_free_bytes() / 1e9, lag_cuda_total_bytes() / 1e9);
        else fprintf(stderr, "[qwen35] CUDA init failed; CPU only\n");
    }
#endif
    double t0 = now_s();
    model_load(&m, snap);
    fprintf(stderr, "[qwen35] loaded in %.1fs (rss %.1f GB, VRAM %.1f GB)\n",
            now_s() - t0, rss_gb(), g_vram_gb);

    if (ref) { g_exact = 1; return run_oracle(&m, ref); }

    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok Ts; tok_load(&Ts, tkp);
        int nslots = getenv("KV_SLOTS") ? atoi(getenv("KV_SLOTS")) : 1;
        if (nslots < 1) nslots = 1;
        if (nslots > SRV_QMAX) nslots = SRV_QMAX;
        int ctxm = getenv("CTX_MAX") ? atoi(getenv("CTX_MAX")) : 8192;
        fprintf(stderr, "[qwen35] serve: %s (KV_SLOTS=%d), %d stop tokens, "
                "state %.2f GB/slot at %d ctx\n",
                nslots > 1 ? "continuous batching" : "serial", nslots, m.c.n_eos,
                state_bytes(&m, ctxm) / 1e9, ctxm);
        printf("serve: %s (KV_SLOTS=%d)\n", nslots > 1 ? "batched" : "serial", nslots);
        fflush(stdout);
        if (nslots > 1) serve_loop_batched(&m, &Ts, nslots);
        else serve_loop(&m, &Ts);
        return 0;
    }

    /* ---- BATCH throughput bench ----
     * BATCH=N decodes N streams in lockstep. Two modes, and the difference matters:
     *   BATCH=N            N identical copies of the prompt. Every stream MUST reproduce the
     *                      single-stream greedy output, which is the correctness assertion --
     *                      but identical streams route to identical experts, so grouping
     *                      collapses to top-k and OVERSTATES throughput.
     *   BATCH=N BATCH_VARY=1  streams seeded to diverge, so experts actually differ. This is
     *                      the honest throughput number; correctness is not asserted because
     *                      the streams are meant to differ. */
    if (getenv("BATCH")) {
        int B = atoi(getenv("BATCH"));
        if (B < 1) B = 1;
        if (B > 64) B = 64;
        int vary = getenv("BATCH_VARY") != NULL;
        Tok Tb;
        char tpb[1024];
        snprintf(tpb, sizeof(tpb), "%s/tokenizer.json", snap);
        tok_load(&Tb, tpb);
        long tnb = 0;
        char *txt = pfile ? read_file(pfile, &tnb) : (prompt ? strdup(prompt) : NULL);
        if (!txt) { fprintf(stderr, "BATCH needs -f or -p\n"); return 1; }
        int capb = (int)tnb + 16;
        int *pids = malloc(sizeof(int) * capb);
        int npb = tok_encode(&Tb, txt, (int)tnb, pids, capb);
        int ctxb = npb + n_gen + 8;

        /* reference: the same prompt decoded alone, greedily */
        State rs;
        state_init(&m, &rs, ctxb);
        float *rl = falloc(m.c.V);
        int *ref = malloc(sizeof(int) * n_gen);
        forward(&m, &rs, pids, npb, 0, rl, 0);
        for (int i = 0; i < n_gen; i++) {
            int a = 0;
            for (int v = 1; v < m.c.V; v++) if (rl[v] > rl[a]) a = v;
            ref[i] = a;
            if (i + 1 < n_gen) forward(&m, &rs, &a, 1, npb + i, rl, 0);
        }
        state_free(&m, &rs);
        free(rl);

        State *sl = calloc(B, sizeof(State));
        State **sp = malloc(sizeof(State *) * B);
        int *pos = malloc(sizeof(int) * B), *tk = malloc(sizeof(int) * B);
        uint64_t *rs2 = malloc(sizeof(uint64_t) * B);
        float *lg = falloc((int64_t)B * m.c.V);
        int **got = malloc(sizeof(int *) * B);
        for (int b = 0; b < B; b++) {
            state_init(&m, &sl[b], ctxb);
            sp[b] = &sl[b];
            got[b] = malloc(sizeof(int) * n_gen);
            rs2[b] = 0x9E3779B97F4A7C15ULL * (uint64_t)(b + 1);
        }
        double tpre = now_s();
        for (int b = 0; b < B; b++)                /* prefill each stream */
            forward(&m, sp[b], pids, npb, 0, lg + (int64_t)b * m.c.V, 0);
        double pre_s = now_s() - tpre;
        for (int b = 0; b < B; b++) pos[b] = npb;

        prof_reset();
        double td = now_s();
        for (int i = 0; i < n_gen; i++) {
            for (int b = 0; b < B; b++) {
                const float *l = lg + (int64_t)b * m.c.V;
                int a = 0;
                for (int v = 1; v < m.c.V; v++) if (l[v] > l[a]) a = v;
                if (vary) {                        /* nudge each stream off the greedy path */
                    rs2[b] = rs2[b] * 6364136223846793005ULL + 1442695040888963407ULL;
                    if (i > 0 && (rs2[b] >> 33) % 4 == 0) {
                        int alt = (int)((rs2[b] >> 13) % (uint64_t)m.c.V);
                        a = alt;
                    }
                }
                got[b][i] = a;
                tk[b] = a;
            }
            forward_batch(&m, sp, tk, pos, B, lg);
            for (int b = 0; b < B; b++) pos[b]++;
        }
        double dt = now_s() - td;
        fprintf(stderr, "[qwen35] BATCH=%d%s prompt=%d tok: prefill %.2fs (%.1f tok/s over %d "
                "streams), decode %d steps in %.2fs = %.2f tok/s aggregate (%.2f per stream)\n",
                B, vary ? " BATCH_VARY" : "", npb, pre_s, B * npb / pre_s, B, n_gen, dt,
                B * n_gen / dt, n_gen / dt);
        prof_report("batched decode", dt, n_gen);
        if (!vary) {
            int bad = 0;
            for (int b = 0; b < B; b++)
                for (int i = 0; i < n_gen; i++)
                    if (got[b][i] != ref[i]) bad++;
            fprintf(stderr, "[qwen35] BATCH correctness: %d/%d stream-tokens differ from the "
                    "single-stream reference -- %s\n", bad, B * n_gen, bad ? "FAIL" : "OK");
            if (bad) return 1;
        }
        for (int b = 0; b < B; b++) { state_free(&m, &sl[b]); free(got[b]); }
        return 0;
    }

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

    prof_reset();
    double tp0 = now_s();
    forward(&m, &s, ids, np, 0, lg, 0);
    double pre = now_s() - tp0;
    fprintf(stderr, "[qwen35] prefill %.2fs (%.1f tok/s)\n", pre, np / pre);
    prof_report("prefill", pre, 1);
    prof_reset();

    double td0 = now_s();
    char buf[512];
    for (int i = 0; i < n_gen; i++) {
        double tsm = PROF_T0();
        int nxt = temp > 0 ? sample_logits(lg, c->V, temp, top_p)
                           : ({ int a = 0; for (int v = 1; v < c->V; v++) if (lg[v] > lg[a]) a = v; a; });
        PROF_ADD(pr_sample, tsm);
        (void)rng;
        int stop = 0;
        for (int e = 0; e < c->n_eos; e++) if (nxt == c->eos[e]) stop = 1;
        if (stop) break;
        int one = nxt;
        if (tok_decode(&T, &one, 1, buf, sizeof(buf)) > 0) { fputs(buf, stdout); fflush(stdout); }
        if (i + 1 < n_gen) forward(&m, &s, &nxt, 1, np + i, lg, 0);
    }
    double dec = now_s() - td0;
    printf("\n");
    fprintf(stderr, "[qwen35] decode %d tok in %.2fs (%.2f tok/s)\n", n_gen, dec, n_gen / dec);
    prof_report("decode", dec, n_gen);
    state_free(&m, &s);
    return 0;
}
