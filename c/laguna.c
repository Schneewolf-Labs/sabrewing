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
#endif
#include "st.h"
#include "tok.h"
#include "json.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#define MAXL 128
#define MAXRD 128          /* max rotary dim */

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec/1e9; }

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
} Layer;

typedef struct {
    Cfg c; shards S;
    int xq;                            /* 1 = int4 packed-expert container */
    int res_dt;                        /* resident dtype: 0=f32, 1=bf16, 2=int8 */
    void *embed, *lm_head;             /* bf16 (real) / f32 (tiny) */
    float *final_norm;
    Layer *L;
} Model;

/* ---------- math ---------- */
static float *falloc(int64_t n) { float *p = malloc(n * sizeof(float)); if (!p) { fprintf(stderr, "OOM %lld\n", (long long)n); exit(1); } return p; }
/* g_exact: force the double-accumulate scalar kernels (bit-exact vs the oracle).
 * Generation defaults to the AVX-512 float path (within quant noise, ~SIMD-fast). */
static int g_exact = 0;
/* y[O] = x[I] @ W[O,I]^T  (W row-major [out,in], HF Linear convention). */
static void matmul(float *y, const float *x, const float *W, int I, int O) {
#if defined(__AVX512F__)
    if (!g_exact) {
        #pragma omp parallel for schedule(static) if(O >= 512)
        for (int o = 0; o < O; o++) {
            const float *w = W + (int64_t)o * I;
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 16 <= I; i += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(w + i), acc);
            float s = _mm512_reduce_add_ps(acc);
            for (; i < I; i++) s += x[i] * w[i];
            y[o] = s;
        }
        return;
    }
#endif
    /* exact double-accumulate reference (the oracle path) */
    #pragma omp parallel for schedule(static) if(O >= 512)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        double s = 0; for (int i = 0; i < I; i++) s += (double)x[i] * w[i];
        y[o] = (float)s;
    }
}

/* bf16 weight dot: W is raw bf16 [O,I]. bf16->f32 is exact (top 16 bits), so the
 * result equals matmul() on the f32-expanded weights but reads half the bytes —
 * the real container stores residents bf16, so this halves resident bandwidth. */
static float bf16_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static int g_res_dt = 0;   /* resident dtype: 0=f32 (tiny), 1=bf16 (real), 2=int8 (RES8) */
static void matmul_bf16(float *y, const float *x, const uint16_t *W, int I, int O) {
    #pragma omp parallel for schedule(static) if(O >= 512)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
#if defined(__AVX512F__)
        if (!g_exact) {
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 16 <= I; i += 16) {
                __m512i we = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(w + i))), 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), _mm512_castsi512_ps(we), acc);
            }
            float s = _mm512_reduce_add_ps(acc);
            for (; i < I; i++) s += x[i] * bf16_f32(w[i]);
            y[o] = s;
            continue;
        }
#endif
        double s = 0; for (int i = 0; i < I; i++) s += (double)x[i] * bf16_f32(w[i]);
        y[o] = (float)s;
    }
}
/* int8 residents: W = [int8 O*I][f32 scale O] in one buffer (per-row scale).
 * ~lossless on residents (proven by inkling's Q8), 4 GB vs 8 GB bf16. */
static void matmul_q8(float *y, const float *x, const void *W, int I, int O) {
    const int8_t *q = (const int8_t*)W;
    const float *scale = (const float*)(q + (int64_t)I * O);
    #pragma omp parallel for schedule(static) if(O >= 512)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
#if defined(__AVX512F__)
        if (!g_exact) {
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 16 <= I; i += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + i),
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w + i)))), acc);
            float s = _mm512_reduce_add_ps(acc);
            for (; i < I; i++) s += x[i] * w[i];
            y[o] = s * scale[o];
            continue;
        }
#endif
        double s = 0; for (int i = 0; i < I; i++) s += (double)x[i] * w[i];
        y[o] = (float)(s * scale[o]);
    }
}
/* resident GEMM: dispatch f32 (tiny/oracle) / bf16 (real) / int8 (RES8) */
static void resmm(float *y, const float *x, const void *W, int I, int O) {
    if (g_res_dt == 2) matmul_q8(y, x, W, I, O);
    else if (g_res_dt == 1) matmul_bf16(y, x, (const uint16_t*)W, I, O);
    else matmul(y, x, (const float*)W, I, O);
}
/* y[O] = x[I] @ dequant(packed)^T; packed [O,I/2] int4 (nibble-8)*scale[o] */
static void matmul_q4(float *y, const float *x, const uint8_t *packed, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static) if(O >= 512)
    for (int o = 0; o < O; o++) {
        const uint8_t *p = packed + (int64_t)o * (I / 2);
#if defined(__AVX512F__)
        if (!g_exact) {
            __m512 acc = _mm512_setzero_ps();
            const __m128i m0f = _mm_set1_epi8(0x0F);
            const __m512 v8 = _mm512_set1_ps(8.f);
            int c = 0;
            for (; c + 16 <= I / 2; c += 16) {                 /* 16 bytes -> 32 weights */
                __m128i b  = _mm_loadu_si128((const __m128i*)(p + c));
                __m128i lo = _mm_and_si128(b, m0f);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
                __m128i il0 = _mm_unpacklo_epi8(lo, hi);        /* [lo0,hi0,lo1,hi1,...] = x[2c..2c+15] order */
                __m128i il1 = _mm_unpackhi_epi8(lo, hi);        /* x[2c+16..2c+31] order */
                __m512 f0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(il0)), v8);
                __m512 f1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(il1)), v8);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + 2 * c), f0, acc);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + 2 * c + 16), f1, acc);
            }
            float s = _mm512_reduce_add_ps(acc);
            for (; c < I / 2; c++) { uint8_t b = p[c]; s += x[2 * c] * ((int)(b & 0xF) - 8) + x[2 * c + 1] * ((int)(b >> 4) - 8); }
            y[o] = s * scale[o];
            continue;
        }
#endif
        double s = 0;
        for (int c = 0; c < I / 2; c++) {
            uint8_t b = p[c];
            s += (double)x[2 * c] * ((int)(b & 0xF) - 8) + (double)x[2 * c + 1] * ((int)(b >> 4) - 8);
        }
        y[o] = (float)(s * scale[o]);
    }
}
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}
static float siluf(float x) { return x / (1.f + expf(-x)); }
static float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }  /* stable */
static void softmax(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    float inv = (float)(1.0 / s); for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ---------- config ---------- */
static double jnum(jval *o, const char *k, double d) { jval *v = json_get(o, k); return (v && v->t == J_NUM) ? v->num : d; }

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
    m->final_norm = load_t(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *L = &m->L[i];
        int D = c->hidden, hix = c->heads[i] * c->head_dim;   /* o_proj in-dim = H*hd */
#define LD(field, suffix) do { snprintf(nm, sizeof(nm), "model.layers.%d." suffix, i); L->field = load_t(m, nm); } while (0)
#define LDR(field, suffix, indim) do { snprintf(nm, sizeof(nm), "model.layers.%d." suffix, i); L->field = load_res(m, nm, indim); } while (0)
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
            rmsnorm(xn + t * D, h + t * D, L->ln1, D, c->rms_eps);
            resmm(Q + (int64_t)t * H * hd, xn + t * D, L->wq, D, H * hd);
            float *kt = Kc + (int64_t)pos * nkv * hd, *vt = Vc + (int64_t)pos * nkv * hd;
            resmm(kt, xn + t * D, L->wk, D, nkv * hd);
            resmm(vt, xn + t * D, L->wv, D, nkv * hd);
            for (int hh = 0; hh < H; hh++) {   /* q_norm + rope per head */
                float *q = Q + ((int64_t)t * H + hh) * hd;
                rmsnorm(q, q, L->qn, hd, c->rms_eps);
                rope_apply(q, pos, inv, rdim, rscale);
            }
            for (int hh = 0; hh < nkv; hh++) {  /* k_norm + rope, in place in the cache */
                float *k = kt + hh * hd;
                rmsnorm(k, k, L->kn, hd, c->rms_eps);
                rope_apply(k, pos, inv, rdim, rscale);
            }
        }
        float scaling = 1.f / sqrtf((float)hd);
        #pragma omp parallel for schedule(dynamic) if(S > 1)
        for (int t = 0; t < S; t++) {
            int pos = pos0 + t;
            int j0 = (win > 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;   /* sliding window */
            float *att = falloc(pos - j0 + 1);
            for (int hh = 0; hh < H; hh++) {
                int kh = hh / group;
                const float *q = Q + ((int64_t)t * H + hh) * hd;
                int m0 = 0;
                for (int j = j0; j <= pos; j++) {                 /* attend cached K [j0..pos] */
                    const float *k = Kc + ((int64_t)j * nkv + kh) * hd;
                    double s = 0; for (int d = 0; d < hd; d++) s += (double)q[d] * k[d];
                    att[m0++] = (float)s * scaling;
                }
                softmax(att, m0);
                float *ao = AO + ((int64_t)t * H + hh) * hd;
                for (int d = 0; d < hd; d++) ao[d] = 0;
                for (int j = j0, mi = 0; j <= pos; j++, mi++) {
                    const float *v = Vc + ((int64_t)j * nkv + kh) * hd;
                    float a = att[mi];
                    for (int d = 0; d < hd; d++) ao[d] += a * v[d];
                }
            }
            free(att);
        }
        /* per-head softplus gate (of the layer input xn) applied before o_proj */
        float *gate = falloc(H);
        for (int t = 0; t < S; t++) {
            resmm(gate, xn + t * D, L->wg, D, H);   /* g_proj: hidden -> num_heads */
            for (int hh = 0; hh < H; hh++) {
                float gv = softplusf(gate[hh]);
                float *ao = AO + ((int64_t)t * H + hh) * hd;
                for (int d = 0; d < hd; d++) ao[d] *= gv;
            }
            resmm(scr, AO + (int64_t)t * H * hd, L->wo, H * hd, D);
            for (int i = 0; i < D; i++) h[t * D + i] += scr[i];
        }
        free(gate); free(Q); free(AO);

        /* --- MLP / MoE (per-token) --- */
        int n = S;
        for (int t = 0; t < n; t++) rmsnorm(xn + t * D, h + t * D, L->ln2, D, c->rms_eps);
        if (!c->is_moe[li]) {
            int I = c->dense_inter;
            float *g = falloc(I), *u = falloc(I);
            for (int t = 0; t < n; t++) {
                resmm(g, xn + t * D, L->gate, D, I);
                resmm(u, xn + t * D, L->up, D, I);
                for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                resmm(scr, g, L->down, I, D);
                for (int i = 0; i < D; i++) h[t * D + i] += scr[i];
            }
            free(g); free(u);
        } else {
            int E = c->n_experts, K = c->topk, I = c->moe_inter, SI = c->shared_inter;
            float *rl = falloc(E), *sc = falloc(E);
            int *sel = malloc(K * sizeof(int)); float *w = falloc(K);
            float *eg = falloc(I), *eu = falloc(I), *acc = falloc(D), *gu = falloc(2 * I);
            float *sg = falloc(SI), *su = falloc(SI);
            for (int t = 0; t < n; t++) {
                matmul(rl, xn + t * D, L->router, D, E);
                for (int e = 0; e < E; e++) sc[e] = 1.f / (1.f + expf(-rl[e]));   /* sigmoid */
                /* top-K by (sigmoid + correction bias) */
                for (int a = 0; a < K; a++) {
                    int best = -1; float bv = -1e30f;
                    for (int e = 0; e < E; e++) {
                        int used = 0; for (int b = 0; b < a; b++) if (sel[b] == e) { used = 1; break; }
                        if (used) continue;
                        float s = sc[e] + L->ebias[e];
                        if (s > bv) { bv = s; best = e; }
                    }
                    sel[a] = best; w[a] = sc[best];   /* unbiased weight */
                }
                if (c->norm_topk) { float sm = 0; for (int a = 0; a < K; a++) sm += w[a]; for (int a = 0; a < K; a++) w[a] /= sm; }
                for (int i = 0; i < D; i++) acc[i] = 0;
                for (int a = 0; a < K; a++) {
                    int e = sel[a];
                    if (m->xq) {   /* int4 container: fused gate_up + down */
                        matmul_q4(gu, xn + t * D, L->gu_q + (int64_t)(e * 2 * I) * (D / 2), L->gu_s + (int64_t)e * 2 * I, D, 2 * I);
                        for (int i = 0; i < I; i++) gu[i] = siluf(gu[i]) * gu[I + i];
                        matmul_q4(scr, gu, L->dn_q + (int64_t)(e * D) * (I / 2), L->dn_s + (int64_t)e * D, I, D);
                    } else {
                        matmul(eg, xn + t * D, L->eg[e], D, I);
                        matmul(eu, xn + t * D, L->eu[e], D, I);
                        for (int i = 0; i < I; i++) eg[i] = siluf(eg[i]) * eu[i];
                        matmul(scr, eg, L->ed[e], I, D);
                    }
                    for (int i = 0; i < D; i++) acc[i] += w[a] * scr[i];
                }
                for (int i = 0; i < D; i++) acc[i] *= c->route_scale;
                /* shared expert (always on, unscaled) */
                resmm(sg, xn + t * D, L->sg, D, SI);
                resmm(su, xn + t * D, L->su, D, SI);
                for (int i = 0; i < SI; i++) sg[i] = siluf(sg[i]) * su[i];
                resmm(scr, sg, L->sd, SI, D);
                for (int i = 0; i < D; i++) h[t * D + i] += acc[i] + scr[i];
            }
            free(rl); free(sc); free(sel); free(w); free(eg); free(eu); free(acc); free(gu); free(sg); free(su);
        }
    }
    for (int t = 0; t < S; t++) rmsnorm(h_out + t * D, h + t * D, m->final_norm, D, c->rms_eps);
    if (pos0 + S > kv->len) kv->len = pos0 + S;
    free(h); free(xn); free(scr);
}
static int argmax_logits(Model *m, const float *h_pos, int *unused) {
    (void)unused; int V = m->c.vocab, D = m->c.hidden;
    float *lg = falloc(V); resmm(lg, h_pos, m->lm_head, D, V);
    int best = 0; for (int v = 1; v < V; v++) if (lg[v] > lg[best]) best = v;
    free(lg); return best;
}

/* ---------- oracle harness ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { *n_out = 0; return NULL; }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}
int main(int argc, char **argv) {
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

    Model m; model_load(&m, snap);
    printf("== Laguna C engine (Stage A, %s) ==\n", m.xq ? "int4 container" : "f32");
    printf("cfg: D=%d L=%d kv=%d hd=%d V=%d E=%d+1 topk=%d moe_I=%d win=%d g_rdim=%d(scale %.4f) s_rdim=%d\n",
           m.c.hidden, m.c.n_layers, m.c.n_kv, m.c.head_dim, m.c.vocab, m.c.n_experts, m.c.topk,
           m.c.moe_inter, m.c.sliding_window, m.c.g_rdim, m.c.g_scale, m.c.s_rdim);
    printf("resident weights loaded | RSS %.2f GB\n", (double)0);

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
        float *lg = falloc(m.c.vocab); resmm(lg, H + (int64_t)i * D, m.lm_head, D, m.c.vocab);
        int am = 0; for (int v = 1; v < m.c.vocab; v++) if (lg[v] > lg[am]) am = v;
        if (tfref && i < ntf && am == tfref[i]) ok++;
        if (i < nfull - 1) {   /* NLL of the true next token */
            softmax(lg, m.c.vocab);
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
