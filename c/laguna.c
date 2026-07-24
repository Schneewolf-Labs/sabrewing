/* laguna.c — Stage-A C inference engine for Poolside's Laguna (LagunaForCausalLM).
 *
 * A second standalone engine alongside inkling.c, sharing the container/tokenizer
 * substrate. Laguna is a 118B/8B-active MoE coding model: GQA with interleaved
 * 512-sliding / global attention, per-head QK-RMSNorm, a per-head softplus
 * attention output gate, RoPE (YaRN partial-rotary on global layers, plain full
 * rotary on sliding layers), a dense layer 0 + sigmoid loss-free-routed MoE
 * (256 experts top-10 + 1 shared expert, routed_scaling 2.5). No RoPE-free
 * relative bias and no short-convs (those are Inkling's) — Laguna is closer to a
 * Qwen2-MoE / GLM / DeepSeek-V3 hybrid.
 *
 * Stage A: f32, all weights resident, full-sequence recompute (no KV cache) — the
 * numerics oracle. Reads the HF safetensors snapshot directly (bits=0). Validated
 * token-exact against a tiny transformers reference (tools/make_tiny_laguna.py).
 *
 *   SNAP=tiny_laguna ./laguna ref_laguna.json
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/select.h>                              /* serve-loop stdin poll (POSIX) */
#endif
#include "st.h"
#include "tok.h"
#include "json.h"
#include "moe_util.h"          /* falloc, jnum, read_int_array */
#include "moe_math.h"          /* now_s, rss_gb, bf16_f32, siluf, softplusf, rmsnorm_row */
#include "moe_matmul.h"        /* matmul_f32 (shared batched f32 GEMM) */
#include "moe_quant.h"         /* matmul_q4_k (shared int4 GEMV, f32/idot contracts) */
#include "moe_sample.h"        /* g_rng, rng_next, sample_logits (temp/top-p) */
#include "moe_serve.h"         /* SReq, serve queue, serve_read_cmd (gateway protocol) */
#include "moe_arch.h"          /* MoeDesc, MoeHooks (descriptor-driven MoE block) */
#include "moe_block.h"         /* moe_block (shared route/top-k/combine) */
#include "moe_attn.h"          /* sdpa_head (shared scaled-dot-product attention) */
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#ifdef COLI_CUDA
#include "backend_cuda_laguna.h"
static int g_cuda = 0;              /* 1 = A6000 tier active (bf16 residents in VRAM) */
/* Offload a resident matmul to the GPU only when its weight has >= this many
 * elements (O*I). Default 0 = offload ALL residents: a sweep on the A6000 box
 * showed offload-all (5.26 tok/s) beats a 6M threshold (4.74) — the spin-sync
 * round-trip is cheap enough that the DDR5 bandwidth saved on even the small
 * k/v/g/shared projections wins, so keeping any resident on the CPU only adds
 * un-overlapped DDR5 reads. Kept as an env knob (LAG_GPU_MINEL) for other GPUs. */
static int64_t g_gpu_minel = 0;
/* Expert VRAM cache: after residents upload, each MoE layer's int4 expert blobs
 * (~1.2 GB) go to VRAM greedily until free VRAM drops below the headroom (or the
 * CUDA_EXPERT_GB cap is hit); the rest fall back to the CPU int4 path. */
/* Activation-staging headroom left free after the expert cache fills. The
 * staging buffers themselves are tiny (<1 MB), but squeezing to ~0 free starves
 * the lazy cudaMalloc for them and the big resident matmuls (lm_head/o_proj)
 * fall back to the CPU — measured 9.0 tok/s at 0 free (36 layers) vs 11.3 at
 * ~0.7 GB free (35 layers). 2 GB is the reproducible sweet spot; CUDA_HEADROOM_MB
 * overrides for other GPUs. */
static size_t g_cuda_headroom = 2ULL << 30;
static int g_exp_gpu_layers = 0;               /* # MoE layers cached in VRAM */
static double g_exp_gpu_gb = 0;                /* expert VRAM used (GB) */
#endif

#define MAXL 128
#define MAXRD 128          /* max rotary dim */

typedef struct {
    int hidden, n_layers, n_kv, head_dim, vocab;
    int n_experts, topk, moe_inter, shared_inter, dense_inter;
    int sliding_window, norm_topk;
    int eos[4], n_eos;
    float rms_eps, route_scale;
    int heads[MAXL];          /* per-layer query-head count */
    int is_sliding[MAXL];     /* 1 = sliding, 0 = global */
    int is_moe[MAXL];         /* 1 = MoE block, 0 = dense MLP */
    /* precomputed RoPE (inv_freq over rotary_dim/2, + attention scaling) */
    int   g_rdim, s_rdim;     /* rotary dims: global (yarn partial), sliding (full) */
    float g_inv[MAXRD/2], s_inv[MAXRD/2];
    float g_scale, s_scale;
} Cfg;

typedef struct {
    float *ln1, *ln2;                 /* input / post-attn layernorm [hidden] (f32) */
    void  *wq, *wk, *wv, *wo, *wg;    /* attn projections (bf16 real / f32 tiny) */
    float *qn, *kn;                   /* qk RMSNorm [head_dim] (f32) */
    /* MLP: dense uses gate/up/down; MoE uses router+ebias+experts+shared */
    void  *gate, *up, *down;          /* dense MLP (bf16/f32) */
    float *router, *ebias;            /* [E,hidden], [E] (f32) */
    float **eg, **eu, **ed;           /* f32 experts (bits=0): [E] × gate/up/down */
    uint8_t *gu_q, *dn_q;             /* int4 container: packed [E*2I, D/2], [E*D, I/2] */
    float *gu_s, *dn_s;               /* per-row f32 scales [E*2I], [E*D] */
    void  *sg, *su, *sd;              /* shared expert (bf16/f32) */
    /* CUDA: device (VRAM) copies of the bf16 residents, NULL = CPU only */
    void  *d_wq, *d_wk, *d_wv, *d_wo, *d_wg;
    void  *d_gate, *d_up, *d_down, *d_sg, *d_su, *d_sd;
    /* CUDA: device copies of this layer's int4 routed-expert blobs (fused expert
     * runs on the GPU when non-NULL; NULL = CPU int4 path). All-or-nothing per
     * layer — the whole [E,...] blob uploads or none of it does. */
    void  *d_gu_q, *d_dn_q; float *d_gu_s, *d_dn_s;
} Layer;

typedef struct {
    Cfg c; shards S;
    int xq;                            /* 1 = int4 packed-expert container */
    int res_dt;                        /* resident dtype: 0=f32, 1=bf16, 2=int8 */
    void *embed, *lm_head;             /* bf16 (real) / f32 (tiny) */
    void *d_lm_head;                   /* CUDA: VRAM copy of lm_head (NULL = CPU) */
    float *final_norm;
    Layer *L;
} Model;

/* ---------- math ---------- */
/* g_exact: force the double-accumulate reference kernels (bit-exact vs the oracle).
 * Generation defaults to the AVX-512 float path (within quant noise, ~SIMD-fast). */
static int g_exact = 0;
/* f32 GEMV is the shared batched matmul_f32 (moe_matmul.h) at S=1. */
static void matmul(float *y, const float *x, const float *W, int I, int O) { matmul_f32(y, x, W, 1, I, O, g_exact); }

/* bf16 weight dot: W is raw bf16 [O,I]. bf16->f32 (bf16_f32, moe_math.h) is exact
 * (top 16 bits), so the result equals matmul() on the f32-expanded weights but
 * reads half the bytes — the real container stores residents bf16. */
static int g_res_dt = 0;   /* resident dtype: 0=f32 (tiny), 1=bf16 (real), 2=int8 (RES8) */
/* bf16 residents use the shared kernel at laguna's contract (round_x=0: keep
 * activations in f32, weight bf16->f32 exact). */
static void matmul_bf16(float *y, const float *x, const uint16_t *W, int I, int O) {
    matmul_bf16_k(y, x, W, 1, I, O, 0, g_exact);
}
/* int8 residents (RES8): W = [int8 O*I][f32 scale O] in one buffer, per-row scale
 * (~lossless, 4 GB vs 8 GB bf16). Split into (q, scale) and use the shared kernel
 * at laguna's contract (f32 activations, AVX-512 cvtepi8->f32). */
static void matmul_q8(float *y, const float *x, const void *W, int I, int O) {
    const int8_t *q = (const int8_t*)W;
    const float *scale = (const float*)(q + (int64_t)I * O);
    matmul_q8_k(y, x, q, scale, I, O, MOE_Q8_F32, g_exact);
}
/* resident GEMM: dispatch VRAM-bf16 (CUDA) / f32 (tiny/oracle) / bf16 (real) /
 * int8 (RES8). Wdev is the weight's VRAM copy (NULL = not resident on GPU); the
 * GPU path is skipped under g_exact so the oracle keeps its double-accumulate. */
static void resmm(float *y, const float *x, const void *W, const void *Wdev, int I, int O) {
#ifdef COLI_CUDA
    if (Wdev && !g_exact) {   /* VRAM resident: int8 (RES8) or bf16 kernel */
        int rc = (g_res_dt == 2) ? lag_cuda_matmul_q8(y, x, Wdev, 1, I, O)
                                 : lag_cuda_matmul_bf16(y, x, Wdev, 1, I, O);
        if (rc == 0) return;
    }
#else
    (void)Wdev;
#endif
    if (g_res_dt == 2) matmul_q8(y, x, W, I, O);
    else if (g_res_dt == 1) matmul_bf16(y, x, (const uint16_t*)W, I, O);
    else matmul(y, x, (const float*)W, I, O);
}
/* y[O] = x[I] @ dequant(packed)^T; packed [O,I/2] int4 (nibble-8)*scale[o] */
/* int4 experts (the CPU tier). Default = f32 activations (accurate). LAG_IDOT=1
 * switches to the int8-activation VNNI path (~2x faster, ~0.4% quant noise) for
 * the CPU-resident expert layers — the decode bottleneck when most layers are on
 * the GPU. g_exact always forces f32 (the oracle stays exact). */
static void matmul_q4(float *y, const float *x, const uint8_t *packed, const float *scale, int I, int O) {
    static int idot = -1;
    if (idot < 0) idot = getenv("LAG_IDOT") ? 1 : 0;
    matmul_q4_k(y, x, packed, scale, I, O, (idot && !g_exact) ? MOE_Q4_IDOT : MOE_Q4_F32, g_exact);
}
/* rmsnorm_row, siluf, softplusf, softmax_row are shared (moe_math.h). */

/* ---------- config ---------- */
/* YaRN inv_freq + attention scaling (mirrors transformers _compute_yarn_parameters) */
static void yarn_rope(Cfg *c, float base, float factor, int orig_max, float beta_fast, float beta_slow, int head_dim, float partial) {
    int dim = (int)(head_dim * partial);   /* rotary dim */
    int half = dim / 2;
    c->g_rdim = dim;
    double lb = 2.0 * log(base);
    double low = floor(dim * log(orig_max / (beta_fast * 2.0 * M_PI)) / lb);
    double high = ceil(dim * log(orig_max / (beta_slow * 2.0 * M_PI)) / lb);
    if (low < 0) low = 0;
    if (high > dim - 1) high = dim - 1;
    double span = (high == low) ? 0.001 : (high - low);
    for (int j = 0; j < half; j++) {
        double pos_freq = pow(base, (2.0 * j) / dim);
        double inv_extra = 1.0 / pos_freq;
        double inv_interp = 1.0 / (factor * pos_freq);
        double ramp = (j - low) / span; if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
        double ext = 1.0 - ramp;                       /* extrapolation factor */
        c->g_inv[j] = (float)(inv_interp * (1.0 - ext) + inv_extra * ext);
    }
    c->g_scale = (float)(0.1 * log(factor) + 1.0);
}
static void default_rope(float *inv, int *rdim_out, float *scale_out, float base, int head_dim, float partial) {
    int dim = (int)(head_dim * partial);
    *rdim_out = dim; *scale_out = 1.f;
    for (int j = 0; j < dim / 2; j++) inv[j] = (float)(1.0 / pow(base, (2.0 * j) / dim));
}

static void cfg_load(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1); if (fread(buf, 1, n, f) != (size_t)n) {} buf[n] = 0; fclose(f);
    char *arena = NULL; jval *o = json_parse(buf, &arena);

    c->hidden       = (int)jnum(o, "hidden_size", 0);
    c->n_layers     = (int)jnum(o, "num_hidden_layers", 0);
    c->n_kv         = (int)jnum(o, "num_key_value_heads", 0);
    c->head_dim     = (int)jnum(o, "head_dim", 0);
    c->vocab        = (int)jnum(o, "vocab_size", 0);
    c->n_experts    = (int)jnum(o, "num_experts", 0);
    c->topk         = (int)jnum(o, "num_experts_per_tok", 0);
    c->moe_inter    = (int)jnum(o, "moe_intermediate_size", 0);
    c->shared_inter = (int)jnum(o, "shared_expert_intermediate_size", 0);
    c->dense_inter  = (int)jnum(o, "intermediate_size", 0);
    c->sliding_window = (int)jnum(o, "sliding_window", 0);
    c->rms_eps      = (float)jnum(o, "rms_norm_eps", 1e-6);
    c->route_scale  = (float)jnum(o, "moe_routed_scaling_factor", 1.0);
    jval *nt = json_get(o, "norm_topk_prob"); c->norm_topk = nt && nt->t == J_BOOL ? nt->boolean : 1;
    if (c->n_layers > MAXL) { fprintf(stderr, "n_layers %d > MAXL\n", c->n_layers); exit(1); }

    c->n_eos = 0;
    jval *es = json_get(o, "eos_token_id");
    if (es && es->t == J_NUM) c->eos[c->n_eos++] = (int)es->num;
    else if (es && es->t == J_ARR) for (int i = 0; i < es->len && c->n_eos < 4; i++) if (es->kids[i]->t == J_NUM) c->eos[c->n_eos++] = (int)es->kids[i]->num;

    int def_heads = (int)jnum(o, "num_attention_heads", 0);
    jval *hpl = json_get(o, "num_attention_heads_per_layer");
    jval *lt  = json_get(o, "layer_types");
    jval *mol = json_get(o, "mlp_only_layers");
    for (int i = 0; i < c->n_layers; i++) {
        c->heads[i] = (hpl && hpl->t == J_ARR && i < hpl->len) ? (int)hpl->kids[i]->num : def_heads;
        c->is_sliding[i] = (lt && lt->t == J_ARR && i < lt->len && !strcmp(lt->kids[i]->str, "sliding_attention")) ? 1 : 0;
        c->is_moe[i] = 1;
    }
    if (mol && mol->t == J_ARR) for (int i = 0; i < mol->len; i++) { int li = (int)mol->kids[i]->num; if (li >= 0 && li < c->n_layers) c->is_moe[li] = 0; }

    /* RoPE — full_attention (YaRN partial) drives global layers, sliding_attention (plain) the rest */
    jval *rp = json_get(o, "rope_parameters");
    jval *full = rp ? json_get(rp, "full_attention") : NULL;
    jval *swa  = rp ? json_get(rp, "sliding_attention") : NULL;
    if (full && !strcmp(json_get(full, "rope_type")->str, "yarn")) {
        yarn_rope(c, (float)jnum(full, "rope_theta", 500000), (float)jnum(full, "factor", 1),
                  (int)jnum(full, "original_max_position_embeddings", 8192),
                  (float)jnum(full, "beta_fast", 32), (float)jnum(full, "beta_slow", 1),
                  c->head_dim, (float)jnum(full, "partial_rotary_factor", 1.0));
    } else {
        default_rope(c->g_inv, &c->g_rdim, &c->g_scale, (float)jnum(full, "rope_theta", 500000),
                     c->head_dim, (float)jnum(full, "partial_rotary_factor", 1.0));
    }
    default_rope(c->s_inv, &c->s_rdim, &c->s_scale, (float)jnum(swa, "rope_theta", 10000),
                 c->head_dim, (float)jnum(swa, "partial_rotary_factor", 1.0));
    free(buf); free(arena);
}

/* ---------- weights ---------- */
static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    float *p = falloc(n); st_read_f32(&m->S, name, p, 0); return p;
}
static uint8_t *load_u8(Model *m, const char *name) {
    int64_t n = st_nbytes(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    uint8_t *p = malloc(n); if (!p) { fprintf(stderr, "OOM %lld\n", (long long)n); exit(1); }
    st_read_raw(&m->S, name, p, 0); return p;
}
/* resident matmul weight in the model's resident dtype (g_res_dt):
 *  0 f32   — read/convert to f32 (tiny oracle).
 *  1 bf16  — keep bf16 raw (half the bytes; bf16->f32 exact in matmul_bf16).
 *  2 int8  — read f32, per-row symmetric int8 (row length = indim), store
 *            [int8 O*I][f32 scale O] in one buffer (~lossless, quarter of f32).
 */
static void *load_res(Model *m, const char *name, int indim) {
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    if (g_res_dt == 2) {
        int64_t O = t->numel / indim, I = indim;
        float *f = falloc(t->numel); st_read_f32(&m->S, name, f, 0);
        int8_t *buf = malloc((size_t)t->numel + (size_t)O * 4);
        if (!buf) { fprintf(stderr, "OOM int8 %s\n", name); exit(1); }
        float *sc = (float*)(buf + O * I);
        for (int64_t o = 0; o < O; o++) {
            const float *w = f + o * I; float mx = 0;
            for (int64_t i = 0; i < I; i++) { float a = fabsf(w[i]); if (a > mx) mx = a; }
            float s = mx / 127.f; if (s < 1e-12f) s = 1e-12f; sc[o] = s;
            int8_t *q = buf + o * I;
            for (int64_t i = 0; i < I; i++) { int v = (int)lrintf(w[i] / s); q[i] = v < -127 ? -127 : v > 127 ? 127 : v; }
        }
        free(f); return buf;
    }
    if (g_res_dt == 1 && t->dtype == 0) {   /* bf16 raw */
        uint16_t *p = malloc((size_t)t->numel * 2);
        if (!p) { fprintf(stderr, "OOM %lld\n", (long long)t->numel); exit(1); }
        st_read_raw(&m->S, name, p, 0); return p;
    }
    return load_t(m, name);
}
static void model_load(Model *m, const char *snap) {
    st_init(&m->S, snap);
    cfg_load(&m->c, snap);
    Cfg *c = &m->c;
    /* int4 container? (converter fuses experts into packed gate_up_proj + .qs) */
    m->xq = 0;
    for (int i = 0; i < c->n_layers && !m->xq; i++) if (c->is_moe[i]) {
        char nm[256]; snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.gate_up_proj.qs", i);
        m->xq = st_has(&m->S, nm);
    }
    int file_bf16 = (st_find(&m->S, "model.embed_tokens.weight")->dtype == 0);
    m->res_dt = file_bf16 ? (getenv("RES8") ? 2 : 1) : 0;   /* bf16 file -> bf16/int8; f32 -> f32 */
    g_res_dt = m->res_dt;
    m->embed = load_res(m, "model.embed_tokens.weight", c->hidden);
    m->lm_head = st_has(&m->S, "lm_head.weight") ? load_res(m, "lm_head.weight", c->hidden) : m->embed;
    /* lm_head is a real D->V matmul (embed is a gather, stays on CPU); put its
     * weight in VRAM too (bf16, or the int8 combined buffer under RES8). */
    m->d_lm_head = NULL;
#ifdef COLI_CUDA
    if (g_cuda && (g_res_dt == 1 || g_res_dt == 2)) {
        const char *lmn = st_has(&m->S, "lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight";
        int64_t ne = st_numel(&m->S, lmn);
        size_t b = (g_res_dt == 2) ? (size_t)ne + (size_t)(ne / c->hidden) * 4 : (size_t)ne * 2;
        m->d_lm_head = lag_cuda_upload(m->lm_head, b);
    }
#endif
    m->final_norm = load_t(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *L = &m->L[i];
        int D = c->hidden, hix = c->heads[i] * c->head_dim;   /* o_proj in-dim = H*hd */
#define LD(field, suffix) do { snprintf(nm, sizeof(nm), "model.layers.%d." suffix, i); L->field = load_t(m, nm); } while (0)
/* DEV: upload a just-loaded resident to VRAM (nm still holds its name). Sizes the
 * bytes by dtype: bf16 = numel*2; int8 (RES8) = the combined [int8 O*I][f32 O]
 * buffer = numel + (numel/indim)*4. NULL when the GPU tier is off (CPU path). */
#ifdef COLI_CUDA
#define DEV(field, indim) do { int64_t _ne = st_numel(&m->S, nm); \
    size_t _b = (g_res_dt == 2) ? (size_t)_ne + (size_t)(_ne / (indim)) * 4 : (size_t)_ne * 2; \
    L->d_##field = (g_cuda && (g_res_dt == 1 || g_res_dt == 2) && _ne >= g_gpu_minel) ? lag_cuda_upload(L->field, _b) : NULL; } while (0)
#else
#define DEV(field, indim) do {} while (0)
#endif
#define LDR(field, suffix, indim) do { snprintf(nm, sizeof(nm), "model.layers.%d." suffix, i); L->field = load_res(m, nm, indim); DEV(field, indim); } while (0)
        LD(ln1, "input_layernorm.weight");
        LD(ln2, "post_attention_layernorm.weight");
        LDR(wq, "self_attn.q_proj.weight", D); LDR(wk, "self_attn.k_proj.weight", D);
        LDR(wv, "self_attn.v_proj.weight", D); LDR(wo, "self_attn.o_proj.weight", hix);
        LDR(wg, "self_attn.g_proj.weight", D);
        LD(qn, "self_attn.q_norm.weight"); LD(kn, "self_attn.k_norm.weight");
        if (!c->is_moe[i]) {
            LDR(gate, "mlp.gate_proj.weight", D); LDR(up, "mlp.up_proj.weight", D); LDR(down, "mlp.down_proj.weight", c->dense_inter);
        } else {
            LD(router, "mlp.gate.weight");
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.e_score_correction_bias", i);
            L->ebias = st_has(&m->S, nm) ? load_t(m, nm) : calloc(c->n_experts, sizeof(float));
            if (m->xq) {
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.gate_up_proj", i);    L->gu_q = load_u8(m, nm);
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.gate_up_proj.qs", i);  L->gu_s = load_t(m, nm);
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.down_proj", i);        L->dn_q = load_u8(m, nm);
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.down_proj.qs", i);     L->dn_s = load_t(m, nm);
#ifdef COLI_CUDA
                if (g_cuda) {   /* cache this layer's experts in VRAM if they fit */
                    int64_t E = c->n_experts, I = c->moe_inter;
                    size_t gub = (size_t)E * (2 * I) * (D / 2), dnb = (size_t)E * D * (I / 2);
                    size_t gus = (size_t)E * 2 * I * 4, dns = (size_t)E * D * 4;
                    size_t need = gub + dnb + gus + dns;
                    size_t cap = getenv("CUDA_EXPERT_GB") ? (size_t)(atof(getenv("CUDA_EXPERT_GB")) * 1e9) : (size_t)-1;
                    if (lag_cuda_free_bytes() > need + g_cuda_headroom && g_exp_gpu_gb * 1e9 + need <= cap) {
                        L->d_gu_q = lag_cuda_upload(L->gu_q, gub); L->d_gu_s = (float*)lag_cuda_upload(L->gu_s, gus);
                        L->d_dn_q = lag_cuda_upload(L->dn_q, dnb); L->d_dn_s = (float*)lag_cuda_upload(L->dn_s, dns);
                        if (L->d_gu_q && L->d_gu_s && L->d_dn_q && L->d_dn_s) { g_exp_gpu_layers++; g_exp_gpu_gb += need / 1e9; }
                        else {   /* partial upload (OOM mid-layer): drop back to CPU for this layer */
                            lag_cuda_free(L->d_gu_q); lag_cuda_free(L->d_gu_s); lag_cuda_free(L->d_dn_q); lag_cuda_free(L->d_dn_s);
                            L->d_gu_q = L->d_dn_q = NULL; L->d_gu_s = L->d_dn_s = NULL;
                        }
                    }
                }
#endif
            } else {
                L->eg = malloc(c->n_experts * sizeof(float*)); L->eu = malloc(c->n_experts * sizeof(float*)); L->ed = malloc(c->n_experts * sizeof(float*));
                for (int e = 0; e < c->n_experts; e++) {
                    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.gate_proj.weight", i, e); L->eg[e] = load_t(m, nm);
                    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.up_proj.weight", i, e);   L->eu[e] = load_t(m, nm);
                    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.down_proj.weight", i, e); L->ed[e] = load_t(m, nm);
                }
            }
            LDR(sg, "mlp.shared_expert.gate_proj.weight", D); LDR(su, "mlp.shared_expert.up_proj.weight", D); LDR(sd, "mlp.shared_expert.down_proj.weight", c->shared_inter);
        }
#undef LD
#undef LDR
#undef DEV
    }
}

/* ---------- forward ---------- */
/* apply RoPE to a head-dim vector `h` (len head_dim); rotate first rdim dims */
static void rope_apply(float *h, int pos, const float *inv, int rdim, float scale) {
    int half = rdim / 2;
    for (int j = 0; j < half; j++) {
        float ang = pos * inv[j];
        float cs = cosf(ang) * scale, sn = sinf(ang) * scale;
        float a = h[j], b = h[j + half];
        h[j]        = a * cs - b * sn;
        h[j + half] = b * cs + a * sn;
    }
}

/* fills h_out[n*hidden] with post-final-norm hidden states */
/* persistent K/V cache: k[layer]/v[layer] indexed by absolute position, holding
 * post-qk-norm, post-rope K and raw V — so decode processes one token instead of
 * re-running the whole sequence (O(n) generation instead of O(n^2)). */
typedef struct { float **k, **v; int max_pos, len; } KVCache;
static void kv_init(KVCache *kv, Model *m, int max_pos) {
    int nkv = m->c.n_kv, hd = m->c.head_dim, L = m->c.n_layers;
    kv->k = malloc(L * sizeof(float*)); kv->v = malloc(L * sizeof(float*));
    for (int l = 0; l < L; l++) { kv->k[l] = falloc((int64_t)max_pos * nkv * hd); kv->v[l] = falloc((int64_t)max_pos * nkv * hd); }
    kv->max_pos = max_pos; kv->len = 0;
}
static void kv_free(KVCache *kv, int L) { for (int l = 0; l < L; l++) { free(kv->k[l]); free(kv->v[l]); } free(kv->k); free(kv->v); }

/* ---------- MoE block hooks (moe_arch.h / moe_block.h) ----------
 * laguna's weight-layout-specific ops behind the shared moe_block interface.
 * ctx carries the layer + per-layer scratch (allocated once in forward). */
typedef struct { Model *m; Layer *L; int I, D; float *gu, *eg, *eu, *sg, *su, *scr; } LagMoeCtx;
static void lag_router(void *cx, const float *x, float *lg) {
    LagMoeCtx *c = cx; matmul(lg, x, c->L->router, c->D, c->m->c.n_experts);
}
static void lag_expert(void *cx, int e, const float *x, float *out) {
    LagMoeCtx *c = cx; Model *m = c->m; Layer *L = c->L; int I = c->I, D = c->D;
#ifdef COLI_CUDA
    if (m->xq && L->d_gu_q &&                      /* fused expert on the VRAM-resident layer */
        lag_cuda_expert_q4(out, x,
            (const uint8_t*)L->d_gu_q + (int64_t)(e * 2 * I) * (D / 2), L->d_gu_s + (int64_t)e * 2 * I,
            (const uint8_t*)L->d_dn_q + (int64_t)(e * D) * (I / 2), L->d_dn_s + (int64_t)e * D, I, D) == 0) return;
#endif
    if (m->xq) {                                   /* int4 container: fused gate_up + down (CPU) */
        matmul_q4(c->gu, x, L->gu_q + (int64_t)(e * 2 * I) * (D / 2), L->gu_s + (int64_t)e * 2 * I, D, 2 * I);
        for (int i = 0; i < I; i++) c->gu[i] = siluf(c->gu[i]) * c->gu[I + i];
        matmul_q4(out, c->gu, L->dn_q + (int64_t)(e * D) * (I / 2), L->dn_s + (int64_t)e * D, I, D);
    } else {                                       /* f32 experts */
        matmul(c->eg, x, L->eg[e], D, I);
        matmul(c->eu, x, L->eu[e], D, I);
        for (int i = 0; i < I; i++) c->eg[i] = siluf(c->eg[i]) * c->eu[i];
        matmul(out, c->eg, L->ed[e], I, D);
    }
}
static void lag_shared(void *cx, const float *x, float *out) {
    LagMoeCtx *c = cx; Layer *L = c->L; int SI = c->m->c.shared_inter, D = c->D;
    resmm(c->sg, x, L->sg, L->d_sg, D, SI);
    resmm(c->su, x, L->su, L->d_su, D, SI);
    for (int i = 0; i < SI; i++) c->sg[i] = siluf(c->sg[i]) * c->su[i];
    resmm(out, c->sg, L->sd, L->d_sd, SI, D);
}
/* acc[D] = sum_k w[k]*expert(sel[k], x). For a VRAM-resident layer, all K experts
 * run in ONE GPU submission (one sync instead of K) — the decode win. Otherwise
 * (CPU int4 / f32) it's the per-expert loop, bit-identical to the block default. */
static void lag_expert_batch(void *cx, const int *sel, const float *w, int K, const float *x, float *acc) {
    LagMoeCtx *c = cx; int D = c->D;
#ifdef COLI_CUDA
    Model *m = c->m; Layer *L = c->L;
    if (m->xq && L->d_gu_q &&
        lag_cuda_moe_experts(acc, x, L->d_gu_q, L->d_gu_s, L->d_dn_q, L->d_dn_s, sel, w, K, c->I, D) == 0) return;
#endif
    for (int a = 0; a < K; a++) {
        lag_expert(c, sel[a], x, c->scr);
        for (int i = 0; i < D; i++) acc[i] += w[a] * c->scr[i];
    }
}

/* process S tokens at absolute positions pos0..pos0+S-1, appending K/V to the
 * cache; writes each token's post-final-norm hidden state to h_out[S*D].
 * Prefill: S=np, pos0=0. Decode: S=1, pos0=current length. */
static void forward(Model *m, KVCache *kv, const int *ids, int S, int pos0, float *h_out) {
    Cfg *c = &m->c;
    int D = c->hidden, hd = c->head_dim, nkv = c->n_kv;
    float *h = falloc((int64_t)S * D);
    for (int t = 0; t < S; t++) {
        if (m->res_dt == 2) {   /* int8 embed: [int8 V*D][f32 scale V] */
            const int8_t *q = (const int8_t*)m->embed + (int64_t)ids[t] * D;
            float s = ((const float*)((const int8_t*)m->embed + (int64_t)c->vocab * D))[ids[t]];
            for (int i = 0; i < D; i++) h[t * D + i] = q[i] * s;
        } else if (m->res_dt == 1) {
            const uint16_t *e = (const uint16_t*)m->embed + (int64_t)ids[t] * D;
            for (int i = 0; i < D; i++) h[t * D + i] = bf16_f32(e[i]);
        } else memcpy(h + t * D, (const float*)m->embed + (int64_t)ids[t] * D, D * sizeof(float));
    }

    float *xn = falloc((int64_t)S * D);
    float *scr = falloc(D);
    for (int li = 0; li < c->n_layers; li++) {
        Layer *L = &m->L[li];
        int H = c->heads[li];
        int rdim = c->is_sliding[li] ? c->s_rdim : c->g_rdim;
        const float *inv = c->is_sliding[li] ? c->s_inv : c->g_inv;
        float rscale = c->is_sliding[li] ? c->s_scale : c->g_scale;
        int win = c->is_sliding[li] ? c->sliding_window : 0;   /* 0 = global (unbounded) */
        int group = H / nkv;
        float *Kc = kv->k[li], *Vc = kv->v[li];

        /* --- project + qk-norm + rope; store K/V into the cache at abs position --- */
        float *Q = falloc((int64_t)S * H * hd);
        float *AO = falloc((int64_t)S * H * hd);
        for (int t = 0; t < S; t++) {
            int pos = pos0 + t;
            rmsnorm_row(xn + t * D, h + t * D, L->ln1, D, c->rms_eps);
            resmm(Q + (int64_t)t * H * hd, xn + t * D, L->wq, L->d_wq, D, H * hd);
            float *kt = Kc + (int64_t)pos * nkv * hd, *vt = Vc + (int64_t)pos * nkv * hd;
            resmm(kt, xn + t * D, L->wk, L->d_wk, D, nkv * hd);
            resmm(vt, xn + t * D, L->wv, L->d_wv, D, nkv * hd);
            for (int hh = 0; hh < H; hh++) {   /* q_norm + rope per head */
                float *q = Q + ((int64_t)t * H + hh) * hd;
                rmsnorm_row(q, q, L->qn, hd, c->rms_eps);
                rope_apply(q, pos, inv, rdim, rscale);
            }
            for (int hh = 0; hh < nkv; hh++) {  /* k_norm + rope, in place in the cache */
                float *k = kt + hh * hd;
                rmsnorm_row(k, k, L->kn, hd, c->rms_eps);
                rope_apply(k, pos, inv, rdim, rscale);
            }
        }
        float scaling = 1.f / sqrtf((float)hd);
        #pragma omp parallel for schedule(dynamic) if(S > 1)
        for (int t = 0; t < S; t++) {
            int pos = pos0 + t;
            int j0 = (win > 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;   /* sliding window */
            float *att = falloc(pos - j0 + 1);
            for (int hh = 0; hh < H; hh++) {   /* attend cached K/V [j0..pos] (shared SDPA) */
                int kh = hh / group;                       /* GQA: this head's KV head */
                /* laguna's contract: scale 1/sqrt(hd), no rel-bias/tau, QK in double.
                 * cache is [pos][kv-head][hd] so this head's rows stride by nkv*hd. */
                sdpa_head(Q + ((int64_t)t * H + hh) * hd, Kc + kh * hd, Vc + kh * hd, nkv * hd,
                          hd, j0, pos, scaling, 1.f, NULL, 0, 1, AO + ((int64_t)t * H + hh) * hd, att);
            }
            free(att);
        }
        /* per-head softplus gate (of the layer input xn) applied before o_proj */
        float *gate = falloc(H);
        for (int t = 0; t < S; t++) {
            resmm(gate, xn + t * D, L->wg, L->d_wg, D, H);   /* g_proj: hidden -> num_heads */
            for (int hh = 0; hh < H; hh++) {
                float gv = softplusf(gate[hh]);
                float *ao = AO + ((int64_t)t * H + hh) * hd;
                for (int d = 0; d < hd; d++) ao[d] *= gv;
            }
            resmm(scr, AO + (int64_t)t * H * hd, L->wo, L->d_wo, H * hd, D);
            for (int i = 0; i < D; i++) h[t * D + i] += scr[i];
        }
        free(gate); free(Q); free(AO);

        /* --- MLP / MoE (per-token) --- */
        int n = S;
        for (int t = 0; t < n; t++) rmsnorm_row(xn + t * D, h + t * D, L->ln2, D, c->rms_eps);
        if (!c->is_moe[li]) {
            int I = c->dense_inter;
            float *g = falloc(I), *u = falloc(I);
            for (int t = 0; t < n; t++) {
                resmm(g, xn + t * D, L->gate, L->d_gate, D, I);
                resmm(u, xn + t * D, L->up, L->d_up, D, I);
                for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                resmm(scr, g, L->down, L->d_down, I, D);
                for (int i = 0; i < D; i++) h[t * D + i] += scr[i];
            }
            free(g); free(u);
        } else {
            /* sigmoid loss-free routing, corr-bias selection, norm_topk, routed
             * *route_scale, one always-on unscaled shared expert (see moe-arch-survey). */
            int I = c->moe_inter, SI = c->shared_inter;
            LagMoeCtx lc = { m, L, I, D, falloc(2 * I), falloc(I), falloc(I), falloc(SI), falloc(SI), falloc(D) };
            MoeDesc desc = { c->n_experts, c->topk, MOE_ROUTE_SIGMOID, 1, c->norm_topk, c->route_scale, MOE_SHARED_UNSCALED };
            MoeHooks hooks = { &lc, lag_router, L->ebias, lag_expert, lag_shared, lag_expert_batch };
            for (int t = 0; t < n; t++) moe_block(&desc, &hooks, xn + t * D, h + t * D, D);
            free(lc.gu); free(lc.eg); free(lc.eu); free(lc.sg); free(lc.su); free(lc.scr);
        }
    }
    for (int t = 0; t < S; t++) rmsnorm_row(h_out + t * D, h + t * D, m->final_norm, D, c->rms_eps);
    if (pos0 + S > kv->len) kv->len = pos0 + S;
    free(h); free(xn); free(scr);
}
static int argmax_logits(Model *m, const float *h_pos, int *unused) {
    (void)unused; int V = m->c.vocab, D = m->c.hidden;
    float *lg = falloc(V); resmm(lg, h_pos, m->lm_head, m->d_lm_head, D, V);
    int best = 0; for (int v = 1; v < V; v++) if (lg[v] > lg[best]) best = v;
    free(lg); return best;
}
static void compute_logits(Model *m, const float *h_pos, float *lg) {
    resmm(lg, h_pos, m->lm_head, m->d_lm_head, m->c.hidden, m->c.vocab);
}

/* ---------- serve mode: openai_server.py engine protocol (like colibri/inkling) ----------
 * stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n / CANCEL <id>\n
 * stdout: READY sentinel + STAT, then per request DATA <id> <n>\n<bytes>\n frames and
 *         DONE <id> STAT <tok> <tps> <hit%> <rss> <prompt_tok> <limited>\n + PROF.  */
/* g_rng, rng_next, PI, pi_desc, sample_logits are shared (moe_sample.h). */
/* SReq, SRV_QMAX, g_q/g_qn, stdin_readable, serve_read_cmd are shared (moe_serve.h). */
static void serve_one(Model *m, Tok *T, SReq *q) {
    Cfg *c = &m->c; int D = c->hidden;
    int cap = q->plen + 16; int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { printf("ERROR %s empty prompt\n", q->id); fflush(stdout); free(ids); return; }
    int ctx_max = getenv("CTX_MAX") ? atoi(getenv("CTX_MAX")) : 8192;
    if (np + q->max_tok > ctx_max) { printf("ERROR %s context exceeds CTX_MAX\n", q->id); fflush(stdout); free(ids); return; }
    KVCache kv; kv_init(&kv, m, np + q->max_tok + 8);
    double t0 = now_s();
    float *Hp = falloc((int64_t)np * D); forward(m, &kv, ids, np, 0, Hp);
    float *lg = falloc(c->vocab); compute_logits(m, Hp + (int64_t)(np - 1) * D, lg); free(Hp);
    float *Hd = falloc(D); char buf[512];
    int gen = 0, limited = 1, pos = np, cancelled = 0;
    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        int tk = sample_logits(lg, c->vocab, q->temp, q->top_p);
        int is_eos = 0; for (int e = 0; e < c->n_eos; e++) if (tk == c->eos[e]) is_eos = 1;
        if (is_eos) { limited = 0; break; }
        int nb = tok_decode(T, &tk, 1, buf, sizeof(buf) - 1);
        printf("DATA %s %d\n", q->id, nb); fwrite(buf, 1, (size_t)nb, stdout); fputc('\n', stdout); fflush(stdout);
        gen++;
        forward(m, &kv, &tk, 1, pos, Hd); pos++;
        compute_logits(m, Hd, lg);
        while (stdin_readable()) { int r = serve_read_cmd(q->id); if (r < 0) { free(ids); free(lg); free(Hd); kv_free(&kv, c->n_layers); return; } if (r > 0) { cancelled = 1; limited = 0; } }
    }
    double dt = now_s() - t0;
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen, dt > 0 ? gen / dt : 0.0, 100.0, rss_gb(), np, limited);
    printf("PROF %.3f %d %d 0.000 0.000 %.3f 0.000 0.000 %d\n", dt, np, gen, dt, gen + 1);
    fflush(stdout);
    free(ids); free(lg); free(Hd); kv_free(&kv, c->n_layers);
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
        serve_one(m, T, &q);
        free(q.payload);
    }
}

/* ---------- oracle harness ---------- */
/* falloc, jnum, read_int_array are shared (moe_util.h). */
#ifdef COLI_CUDA
/* Kernel-level validation: random data through the CPU kernels and the GPU
 * kernels, reporting max abs / max relative difference. bf16 & q4 should agree
 * to float-noise (~1e-3 rel bf16, ~1e-4 rel q4) — only the reduction order
 * differs. No model needed: `LAG_CUDA_TEST=1 ./laguna` (SNAP unused). */
static void frand_fill(float *p, int64_t n) { for (int64_t i = 0; i < n; i++) p[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f; }
static int cuda_selftest(void) {
    if (lag_cuda_init(getenv("GPU_DEV") ? atoi(getenv("GPU_DEV")) : 0) != 0) { fprintf(stderr, "[cuda-test] init failed\n"); return 1; }
    srand(1234); g_exact = 0; g_res_dt = 1;
    int rc = 0;
    /* --- bf16 matmul: weight bf16, x f32 --- */
    {
        int I = 3072, O = 4096;
        float *x = falloc(I), *yc = falloc(O), *yg = falloc(O);
        uint16_t *w = malloc((size_t)O * I * 2);
        frand_fill(x, I);
        for (int64_t k = 0; k < (int64_t)O * I; k++) { float f = (rand() / (float)RAND_MAX) * 2.f - 1.f; uint32_t u; memcpy(&u, &f, 4); w[k] = u >> 16; }
        matmul_bf16(yc, x, w, I, O);
        void *dw = lag_cuda_upload(w, (size_t)O * I * 2);
        if (!dw || lag_cuda_matmul_bf16(yg, x, dw, 1, I, O) != 0) { fprintf(stderr, "[cuda-test] bf16 gpu call failed\n"); return 1; }
        float md = 0, mr = 0; for (int o = 0; o < O; o++) { float d = fabsf(yc[o] - yg[o]); if (d > md) md = d; float r = d / (fabsf(yc[o]) + 1e-6f); if (r > mr) mr = r; }
        printf("[cuda-test] bf16 matmul  [%d->%d]  max|abs|=%.3e  max_rel=%.3e  %s\n", I, O, md, mr, mr < 5e-3 ? "OK" : "FAIL");
        if (mr >= 5e-3) rc = 1;
        lag_cuda_free(dw); free(x); free(yc); free(yg); free(w);
    }
    /* --- q4 matmul: packed nibble-8, per-row scale --- */
    {
        int I = 3072, O = 2048;
        float *x = falloc(I), *yc = falloc(O), *yg = falloc(O), *sc = falloc(O);
        uint8_t *p = malloc((size_t)O * (I / 2));
        frand_fill(x, I);
        for (int64_t k = 0; k < (int64_t)O * (I / 2); k++) p[k] = rand() & 0xFF;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() / (float)RAND_MAX) * 0.05f;
        matmul_q4(yc, x, p, sc, I, O);
        void *dp = lag_cuda_upload(p, (size_t)O * (I / 2)), *ds = lag_cuda_upload(sc, (size_t)O * 4);
        if (!dp || !ds || lag_cuda_matmul_q4(yg, x, dp, ds, 1, I, O) != 0) { fprintf(stderr, "[cuda-test] q4 gpu call failed\n"); return 1; }
        float md = 0, mr = 0; for (int o = 0; o < O; o++) { float d = fabsf(yc[o] - yg[o]); if (d > md) md = d; float r = d / (fabsf(yc[o]) + 1e-6f); if (r > mr) mr = r; }
        printf("[cuda-test] q4 matmul    [%d->%d]  max|abs|=%.3e  max_rel=%.3e  %s\n", I, O, md, mr, mr < 1e-3 ? "OK" : "FAIL");
        if (mr >= 1e-3) rc = 1;
        lag_cuda_free(dp); lag_cuda_free(ds); free(x); free(yc); free(yg); free(sc); free(p);
    }
    /* --- fused expert (int4): out[D] = W2 @ siluglu(W13 @ x), block-concat gate_up --- */
    {
        int D = 3072, I = 1024;
        float *x = falloc(D), *yc = falloc(D), *yg = falloc(D);
        float *gu = falloc(2 * I), *s13 = falloc(2 * I), *s2 = falloc(D);
        uint8_t *p13 = malloc((size_t)(2 * I) * (D / 2)), *p2 = malloc((size_t)D * (I / 2));
        frand_fill(x, D);
        for (int64_t k = 0; k < (int64_t)(2 * I) * (D / 2); k++) p13[k] = rand() & 0xFF;
        for (int64_t k = 0; k < (int64_t)D * (I / 2); k++) p2[k] = rand() & 0xFF;
        for (int r = 0; r < 2 * I; r++) s13[r] = 0.01f + (rand() / (float)RAND_MAX) * 0.03f;
        for (int r = 0; r < D; r++) s2[r] = 0.01f + (rand() / (float)RAND_MAX) * 0.03f;
        /* CPU reference: same as the MoE loop's int4 expert */
        matmul_q4(gu, x, p13, s13, D, 2 * I);
        for (int i = 0; i < I; i++) gu[i] = siluf(gu[i]) * gu[I + i];
        matmul_q4(yc, gu, p2, s2, I, D);
        void *dp13 = lag_cuda_upload(p13, (size_t)(2 * I) * (D / 2)), *ds13 = lag_cuda_upload(s13, (size_t)(2 * I) * 4);
        void *dp2 = lag_cuda_upload(p2, (size_t)D * (I / 2)), *ds2 = lag_cuda_upload(s2, (size_t)D * 4);
        if (!dp13 || !ds13 || !dp2 || !ds2 || lag_cuda_expert_q4(yg, x, dp13, ds13, dp2, ds2, I, D) != 0) { fprintf(stderr, "[cuda-test] expert gpu call failed\n"); return 1; }
        /* chained q4->silu->q4: relative error blows up on near-zero outputs, so
         * abs error (vs the ~0.03-scale, D-wide accumulation) is the honest signal. */
        float md = 0, mr = 0; int nbig = 0; for (int o = 0; o < D; o++) { float d = fabsf(yc[o] - yg[o]); if (d > md) md = d; float r = d / (fabsf(yc[o]) + 1e-6f); if (r > mr) mr = r; if (r > 1e-3f) nbig++; }
        printf("[cuda-test] expert_q4    [D=%d I=%d] max|abs|=%.3e  max_rel=%.3e (%d/%d elems >1e-3 rel)  %s\n", D, I, md, mr, nbig, D, md < 1e-3 ? "OK" : "FAIL");
        if (md >= 1e-3) rc = 1;
        lag_cuda_free(dp13); lag_cuda_free(ds13); lag_cuda_free(dp2); lag_cuda_free(ds2);
        free(x); free(yc); free(yg); free(gu); free(s13); free(s2); free(p13); free(p2);
    }
    printf("[cuda-test] %s\n", rc ? "FAILURES" : "ALL PASS");
    return rc;
}
#endif

int main(int argc, char **argv) {
#ifdef COLI_CUDA
    if (getenv("LAG_CUDA_TEST")) return cuda_selftest();
#endif
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    const char *prompt = NULL, *refpath = "ref_laguna.json";
    int n_new = 128;
    char *pbuf = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) {   /* prompt from file (chat template etc.) */
            FILE *pf = fopen(argv[++i], "rb"); if (!pf) { perror(argv[i]); return 1; }
            fseek(pf, 0, SEEK_END); long pn = ftell(pf); fseek(pf, 0, SEEK_SET);
            pbuf = malloc(pn + 1); if (fread(pbuf, 1, pn, pf) != (size_t)pn) {} pbuf[pn] = 0; fclose(pf);
            prompt = pbuf;
        }
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_new = atoi(argv[++i]);
        else refpath = argv[i];
    }

#ifdef COLI_CUDA
    /* bring up the A6000 tier before load so bf16 residents upload as they read */
    if (!getenv("NOGPU")) {
        int dev = getenv("GPU_DEV") ? atoi(getenv("GPU_DEV")) : 0;
        if (lag_cuda_init(dev) == 0) { g_cuda = 1;
            if (getenv("LAG_GPU_MINEL")) g_gpu_minel = strtoll(getenv("LAG_GPU_MINEL"), NULL, 10);
            if (getenv("CUDA_HEADROOM_MB")) g_cuda_headroom = (size_t)atoll(getenv("CUDA_HEADROOM_MB")) << 20;
            fprintf(stderr, "[cuda] device %d up, %.1f GB VRAM free, offload minel=%lld\n", dev, lag_cuda_free_bytes() / 1e9, (long long)g_gpu_minel); }
        else fprintf(stderr, "[cuda] init failed — CPU only\n");
    }
#endif
    Model m; model_load(&m, snap);
    printf("== Laguna C engine (Stage A, %s) ==\n", m.xq ? "int4 container" : "f32");
#ifdef COLI_CUDA
    if (g_cuda) {
        int moe = 0; for (int i = 0; i < m.c.n_layers; i++) moe += m.c.is_moe[i];
        printf("CUDA: %s residents + %d/%d MoE layers' experts in VRAM (%.1f GB experts) | %.1f GB VRAM free\n",
               m.res_dt == 2 ? "int8" : "bf16", g_exp_gpu_layers, moe, g_exp_gpu_gb, lag_cuda_free_bytes() / 1e9);
    }
#endif
    printf("cfg: D=%d L=%d kv=%d hd=%d V=%d E=%d+1 topk=%d moe_I=%d win=%d g_rdim=%d(scale %.4f) s_rdim=%d\n",
           m.c.hidden, m.c.n_layers, m.c.n_kv, m.c.head_dim, m.c.vocab, m.c.n_experts, m.c.topk,
           m.c.moe_inter, m.c.sliding_window, m.c.g_rdim, m.c.g_scale, m.c.s_rdim);
    printf("resident weights loaded | RSS %.2f GB\n", rss_gb());

    /* ---- serve mode: the openai_server.py gateway drives us over stdin/stdout ---- */
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        serve_loop(&m, &T);
        return 0;
    }

    /* ---- prompt / generate mode (greedy, streaming; KV-cached) ---- */
    if (prompt) {
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        int plen = (int)strlen(prompt), cap = plen + 16;
        int *seq = malloc((cap + n_new) * sizeof(int));
        int np = tok_encode(&T, prompt, plen, seq, cap);
        if (np <= 0) { fprintf(stderr, "empty prompt after tokenization\n"); return 1; }
        printf("[%d prompt tokens]\n%s", np, prompt); fflush(stdout);
        int D = m.c.hidden; char buf[512];
        KVCache kv; kv_init(&kv, &m, np + n_new + 8);
        double t0 = now_s();
        float *Hp = falloc((int64_t)np * D);
        forward(&m, &kv, seq, np, 0, Hp);                       /* prefill the prompt */
        int cur = argmax_logits(&m, Hp + (int64_t)(np - 1) * D, NULL);
        free(Hp);
        double t1 = now_s();
        float *Hd = falloc(D);
        int pos = np, gen = 0;
        for (int s = 0; s < n_new; s++) {
            int is_eos = 0; for (int e = 0; e < m.c.n_eos; e++) if (cur == m.c.eos[e]) is_eos = 1;
            if (is_eos) { printf("\n[eos]"); break; }
            int nb = tok_decode(&T, &cur, 1, buf, sizeof(buf) - 1); buf[nb] = 0;
            fputs(buf, stdout); fflush(stdout);
            gen++;
            forward(&m, &kv, &cur, 1, pos, Hd);                 /* decode one token at abs pos */
            cur = argmax_logits(&m, Hd, NULL);
            pos++;
        }
        free(Hd); kv_free(&kv, m.c.n_layers);
        printf("\n[prefill %.1fs (%d tok) | %d tok in %.1fs = %.2f tok/s]\n",
               t1 - t0, np, gen, now_s() - t1, gen > 0 ? gen / (now_s() - t1) : 0.0);
        return 0;
    }

    g_exact = 1;   /* the oracle validates the exact double-accumulate kernels */
    if (getenv("LAG_SIMD")) g_exact = 0;   /* opt into the SIMD path for a noise check */
    FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1); if (fread(buf, 1, n, f) != (size_t)n) {} buf[n] = 0; fclose(f);
    char *arena = NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull, ntf;
    int *pids = read_int_array(ref, "prompt_ids", &np);
    int *full = read_int_array(ref, "full_ids", &nfull);
    int *tfref = read_int_array(ref, "tf_pred", &ntf);
    int ngen = nfull - np;

    double t0 = now_s();
    int D = m.c.hidden;
    /* pass 1: teacher-forced argmax + perplexity over the full reference */
    KVCache kv; kv_init(&kv, &m, nfull + 8);
    float *H = falloc((int64_t)nfull * D);
    forward(&m, &kv, full, nfull, 0, H);
    int ok = 0; double nll = 0;
    for (int i = 0; i < nfull; i++) {
        float *lg = falloc(m.c.vocab); resmm(lg, H + (int64_t)i * D, m.lm_head, m.d_lm_head, D, m.c.vocab);
        int am = 0; for (int v = 1; v < m.c.vocab; v++) if (lg[v] > lg[am]) am = v;
        if (tfref && i < ntf && am == tfref[i]) ok++;
        if (i < nfull - 1) {   /* NLL of the true next token */
            softmax_row(lg, m.c.vocab);
            nll += -log((double)lg[full[i + 1]] + 1e-30);
        }
        free(lg);
    }
    double ppl = exp(nll / (nfull - 1));
    printf("teacher-forced argmax: %d/%d match | perplexity: %.4f\n", ok, nfull, ppl);
    jval *pr = json_get(ref, "ppl_ref");
    if (pr && pr->t == J_NUM) printf("ppl_ref: %.4f (%.2f%% diff)\n", pr->num, 100.0 * fabs(ppl - pr->num) / pr->num);
    free(H);

    /* pass 2: greedy generation, token-for-token vs the oracle (KV-cached, fresh cache) */
    int *seq = malloc((nfull + 1) * sizeof(int));
    memcpy(seq, pids, np * sizeof(int));
    int match = 0;
    KVCache kv2; kv_init(&kv2, &m, nfull + 8);
    float *Hg = falloc((int64_t)np * D);
    forward(&m, &kv2, pids, np, 0, Hg);
    seq[np] = argmax_logits(&m, Hg + (int64_t)(np - 1) * D, NULL);
    free(Hg);
    float *Hd = falloc(D);
    for (int s = 1; s < ngen; s++) {
        forward(&m, &kv2, &seq[np + s - 1], 1, np + s - 1, Hd);
        seq[np + s] = argmax_logits(&m, Hd, NULL);
    }
    free(Hd); kv_free(&kv2, m.c.n_layers); kv_free(&kv, m.c.n_layers);
    printf("Reference: "); for (int i = np; i < nfull; i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i = np; i < nfull; i++) { printf("%d ", seq[i]); if (seq[i] == full[i]) match++; }
    printf("\nMatching tokens: %d/%d | %.2fs\n", match, ngen, now_s() - t0);
    free(buf); free(arena);
    return (match == ngen) ? 0 : 1;
}
