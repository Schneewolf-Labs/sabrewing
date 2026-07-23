/* Pure-C inference engine for Thinking Machines "Inkling" (text-only), Stage A.
 * Goal, like olmoe.c before GLM-5.2: reproduce the EXACT token ids of the HF
 * transformers reference (ref_inkling.json from tools/make_tiny_inkling.py)
 * to validate the core math before scaling to the 975B checkpoint.
 *
 * Architecture (vs glm.c's MLA/RoPE/DSA — shares almost nothing):
 *  - hybrid attention: sliding-window layers (window=512, 16 KV heads) and
 *    global layers (8 KV heads) interleaved 5:1; conventional GQA, no RoPE
 *  - learned relative-position bias: r_proj(x) mixes a per-layer bank
 *    proj[d_rel, rel_extent] into one bias per backward distance
 *  - log-length scaling tau on global layers past n_floor tokens
 *  - depthwise-causal short convs (kernel 4, residual inside, fp32):
 *    on K and V inside attention, after attention, and after the MLP
 *  - MoE: sigmoid router + loss-free bias for top-k selection; combine
 *    weights are sigmoids of the raw logits jointly normalized over
 *    topk routed + n_shared shared experts, x route_scale x global_scale
 *  - logits: hidden / logits_mup_width_multiplier, sliced to unpadded vocab
 *
 * Dense weights (attn, norms, convs, router, shared experts, dense MLP)
 * resident in RAM as f32; routed experts streamed from disk per-expert out
 * of the fused [E, 2I, D] / [E, D, I] tensors, LRU-cached, optionally
 * int-quantized (bits=0 keeps them f32 for bit-exact oracle validation).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/select.h>
#endif
#include "st.h"
#include "lora.h"
#include "tok.h"
#include "tier.h"
#ifdef COLI_CUDA
#include "backend_cuda_ink.h"
static int g_cuda = 0;
#endif

#define MAXL 256

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, vocab, unpad_vocab;
    int n_heads, n_kv, head_dim;          /* global ("hybrid") layers */
    int swa_heads, swa_kv, swa_hd;        /* sliding ("hybrid_sliding") layers */
    int window, d_rel, rel_extent, conv_k;
    double log_floor;                     /* <=0: log scaling off */
    float log_alpha;
    int n_experts, topk, n_shared, moe_inter, dense_inter;
    int eos, img_tok, aud_tok, ctx_max;   /* multimodal placeholders; serve KV bound */
    float eps, route_scale, mup;
    unsigned char local[MAXL];            /* 1 = sliding-window layer */
    unsigned char sparse[MAXL];           /* 1 = MoE layer, 0 = dense MLP */
} Cfg;

/* per-layer dims that depend on the attention type */
#define L_HEADS(c,i) ((c)->local[i] ? (c)->swa_heads : (c)->n_heads)
#define L_KV(c,i)    ((c)->local[i] ? (c)->swa_kv    : (c)->n_kv)
#define L_HD(c,i)    ((c)->local[i] ? (c)->swa_hd    : (c)->head_dim)
#define L_EXT(c,i)   ((c)->local[i] ? (c)->window    : (c)->rel_extent)

/* ---------- resident weights ----------
 * Large matmul weights keep their on-disk dtype in RAM: bf16 for the real
 * 975B checkpoint (f32 residents would need ~172 GB, over sabre's 187),
 * f32 for the tiny oracle (bit-exact validation). Under CUDA, bf16 tensors
 * move to VRAM (dev set, host freed): decode reads ~35 GB of residents per
 * token, so this trades the DDR5 bandwidth wall for VRAM bandwidth AND
 * frees the same RAM for the expert cache. */
typedef struct { float *f; uint16_t *h; void *dev; void *dq8, *dqs; } Wt;  /* dq8/dqs: int8 residents in VRAM */

typedef struct {
    float *in_ln, *post_ln;
    Wt q, k, v, r, o;                     /* projections */
    float *qn, *kn;                       /* per-head rmsnorm [head_dim] */
    float *relp;                          /* [d_rel, ext] bias bank */
    float *k_cw, *v_cw, *a_cw, *m_cw;     /* sconv weights, [C*K] depthwise */
    /* dense layers */
    Wt dg, du, dd; float dgs;
    /* MoE layers */
    float *router, *rbias, rgs;           /* [E+ns, D], [E], scalar */
    Wt sh_g, sh_u, sh_d;                  /* shared experts [ns][I,D] etc. */
} Layer;

/* MTP head module: a full (dense) Inkling block plus the three MTP-specific
 * tensors. Module k predicts token t+k+1 from [hidden_norm(h) ; embed_norm(emb)]
 * projected through in_proj; shares the main final_norm + lm_head. Its KV/conv
 * live at index n_layers+k so attention()/sconv_apply() are reused verbatim. */
typedef struct { Layer L; Wt in_proj; float *embed_norm, *hidden_norm; int inter; } MtpMod;

/* ---------- routed-expert cache: LRU + optional pinned set ----------
 * Container snapshots keep the expert rows PACKED in RAM (int4 stays 4-bit:
 * ~28 MB/expert instead of ~57 unpacked, so the same budget caches twice the
 * experts); the matmul kernels unpack nibbles in-register. */
typedef struct {
    int eid; uint64_t used;
    int pinned;                           /* never evicted (usage-history pin) */
    int filled;                           /* 0 while queued for a parallel fill */
    int pending;                          /* async engine: read chunks still in flight */
    int busy;                             /* referenced by the moe() call in progress:
                                           * a mid-call slot_acquire must not evict what
                                           * pass 3 is about to compute from */
    int transient;                        /* overflow slot outside the cache, freed at the
                                           * end of the moe() call that spilled into it */
    uint8_t *p13, *p2; float *s13, *s2;   /* container: packed rows + row scales */
    int8_t *q13, *q2;                     /* bits>0: runtime-quantized int8 */
    float *f13, *f2;                      /* bits==0: raw f32 (oracle) */
    void *d13, *ds13, *d2, *ds2; int gpu; /* VRAM copies (device ptrs): int4 expert matmul on the GPU */
} Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    Lora lora;                            /* LoRA adapter (lora.on = active) */
    int quant_bits;                       /* 0 = f32 experts (oracle mode) */
    int xq;                               /* experts on disk are a colibri container (U8 + .qs) */
    Wt embed, lm_head;
    float *embed_norm, *final_norm;
    Layer *L;
    MtpMod *mtp; int n_mtp;              /* speculative-draft head (0 = absent) */
    LCache *cache;
    int64_t rb13, rb2;                    /* container row-bytes (0 = not container) */
    uint32_t **eusage;                    /* per-layer expert selection counts */
    uint8_t **seen;                       /* per-layer: expert touched this generation (miss classification) */
    int **prev_topm; int topm;            /* previous decode token's top-M router picks per layer (overfetch probe) */
    int qsim_bits, qsim_all, qsim_hotkeep;/* heat-tiered quant SIM: degrade cold experts to N-bit (quality probe) */
    uint32_t *qsim_thr;                   /* per-layer usage threshold: >= keeps int4, < degrades */
    int npin;                             /* pinned experts per sparse layer */
    uint64_t clock, hits, miss, miss_cold, miss_churn, cold_recover, cold_novel;
    double t_fill, t_expert, t_shared, t_attn, t_route;   /* phase timers */
    float **K, **V; int kv_len, max_t;    /* per-layer [kv][max_t][hd] */
    float **cs[4];                        /* conv states, [n_layers][C*(K-1)] */
    /* speculative verify: per-position conv-state checkpoints so a partially
     * accepted draft batch can roll each of the 4 convs back to the last
     * accepted position. vck[j][li] is [vck_S][C_j*(K-1)]; vck_on gates the
     * checkpoint writes inside sconv_apply. */
    float **vck[4]; int vck_on, vck_S;
    double dense_load_s;
} Model;

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
#endif
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }
static float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
static float siluf(float x) { return x / (1.f + expf(-x)); }

/* y[S,O] = x[S,I] @ W^T, W row-major [O,I] */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

#if defined(__AVX512BF16__) && defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_BF16_DOT 1
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* bf16-weight matmul: activations rounded to bf16 per row (matches the HF
 * bf16 reference numerics), hardware vdpbf16ps dot where available,
 * shift-to-f32 scalar otherwise. */
static void matmul_h(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
#ifdef HAVE_BF16_DOT
    if (I % 32 == 0) {
        uint16_t *xh = malloc((size_t)S * I * sizeof(uint16_t));
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            uint16_t *xd = xh + (int64_t)s * I;
            for (int i = 0; i < I; i += 32) {
                __m512 a = _mm512_loadu_ps(xs + i), b = _mm512_loadu_ps(xs + i + 16);
                _mm512_storeu_si512(xd + i, (__m512i)_mm512_cvtne2ps_pbh(b, a));
            }
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint16_t *w = W + (int64_t)o * I;
            for (int s = 0; s < S; s++) {
                const uint16_t *xs = xh + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                for (int i = 0; i < I; i += 32)
                    acc = _mm512_dpbf16_ps(acc, (__m512bh)_mm512_loadu_si512(xs + i),
                                                (__m512bh)_mm512_loadu_si512(w + i));
                y[(int64_t)s * O + o] = _mm512_reduce_add_ps(acc);
            }
        }
        free(xh);
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) {
                union { uint32_t u; float f; } v = { (uint32_t)w[i] << 16 };
                acc += xs[i] * v.f;
            }
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* dispatch on where the weight lives */
static void matmul_w(float *y, const float *x, Wt W, int S, int I, int O) {
#ifdef COLI_CUDA
    if (W.dq8) {   /* int8 resident in VRAM */
        if (ink_cuda_matmul_q8(y, x, W.dq8, W.dqs, S, I, O) == 0) return;
        fprintf(stderr, "cuda int8 matmul failed and host copy was freed\n"); exit(1);
    }
    if (W.dev) {
        if (ink_cuda_matmul_bf16(y, x, W.dev, S, I, O) == 0) return;
        fprintf(stderr, "cuda matmul failed and host copy was freed\n"); exit(1);
    }
#endif
    if (W.f) matmul(y, x, W.f, S, I, O);
    else     matmul_h(y, x, W.h, S, I, O);
}

/* y[1,O] = x @ q^T, int8 weights + per-row scale. Fast path: activations
 * quantized Q8 per 32-block, VNNI (or maddubs) int8 dot — same family as
 * glm.c's IDOT kernels; IDOT=0 falls back to the byte-exact scalar route. */
#if defined(__AVX2__)
static inline __m256i i8dot_block(__m256i acc, __m256i a, __m256i b) {
    __m256i ax = _mm256_sign_epi8(a, a);        /* |a| as u8 */
    __m256i sy = _mm256_sign_epi8(b, a);        /* b * sign(a) */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#else
    __m256i p = _mm256_maddubs_epi16(ax, sy);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
#endif
}
#endif
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__AVX2__)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 32 == 0 && I <= 8192) {
        int nb = I / 32;
        int8_t xi[8192]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*32;
            float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 32; i++) xi[b*32+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                           _mm256_loadu_si256((const __m256i*)(xi + b*32)),
                                           _mm256_loadu_si256((const __m256i*)(w + b*32)));
                __m128i lo = _mm256_castsi256_si128(vacc), hi = _mm256_extracti128_si256(vacc, 1);
                __m128i s4 = _mm_add_epi32(lo, hi);
                s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                acc += xs[b] * (float)_mm_cvtsi128_si32(s4);
            }
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* y[1,O] = x @ W^T with W kept PACKED int4 (low nibble = even column, +8
 * offset, per-row scale — the on-disk container layout, cached as-is).
 * Nibbles unpack in-register: same numeric result as unpack-to-int8 +
 * matmul_q, half the cache footprint. IDOT=0 keeps the byte-exact scalar. */
static void matmul_q4(float *y, const float *x, const uint8_t *p, const float *scale, int I, int O) {
#if defined(__AVX2__)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 32 == 0 && I <= 8192) {
        int nb = I / 32;
        int8_t xi[8192]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*32;
            float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 32; i++) xi[b*32+i] = (int8_t)lrintf(xb[i]*inv);
        }
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m256i b8 = _mm256_set1_epi8(8);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w = p + (int64_t)o * (I/2);
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                __m128i by = _mm_loadu_si128((const __m128i*)(w + b*16));  /* 16 B = 32 nibbles */
                __m128i lo = _mm_and_si128(by, m4);                        /* even columns */
                __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);     /* odd columns  */
                __m256i nib = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),  /* cols 16..31 */
                                               _mm_unpacklo_epi8(lo, hi)); /* cols  0..15 */
                nib = _mm256_sub_epi8(nib, b8);
                __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                           _mm256_loadu_si256((const __m256i*)(xi + b*32)), nib);
                __m128i l = _mm256_castsi256_si128(vacc), h = _mm256_extracti128_si256(vacc, 1);
                __m128i s4 = _mm_add_epi32(l, h);
                s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                acc += xs[b] * (float)_mm_cvtsi128_si32(s4);
            }
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = p + (int64_t)o * (I/2);
        float acc = 0.f;
        for (int i = 0; i < I; i += 2) {
            uint8_t byte = w[i/2];
            acc += x[i]   * (float)((int)(byte & 0xF) - 8);
            acc += x[i+1] * (float)((int)(byte >> 4)  - 8);
        }
        y[o] = acc * scale[o];
    }
}

static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / qmax; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v >  qmax) v =  qmax;
            if (v < -qmax-1) v = -qmax-1;
            qr[i] = (int8_t)v;
        }
    }
}

/* rmsnorm computed in f64 accumulate like the f32->f32 reference */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- depthwise causal short conv, residual inside (fp32) ----------
 * seq[S,C] in-place: out[t] = sum_j w[c,j]*in[t+j-(K-1)] + in[t], history from
 * state[C*(K-1)] (raw pre-conv inputs), which is updated to the new tail. */
static void sconv_apply(float *seq, int S, int C, const float *w, float *state, int K, float *ckpt) {
    int P = K - 1;
    #pragma omp parallel
    {
        float *col = malloc((P + S) * sizeof(float));
        #pragma omp for schedule(static)
        for (int ch = 0; ch < C; ch++) {
            for (int j = 0; j < P; j++) col[j] = state[(int64_t)ch*P + j];
            for (int t = 0; t < S; t++) col[P + t] = seq[(int64_t)t*C + ch];
            const float *wc = w + (int64_t)ch*K;
            for (int t = 0; t < S; t++) {
                float acc = 0.f;
                for (int j = 0; j < K; j++) acc += wc[j] * col[t + j];
                seq[(int64_t)t*C + ch] = acc + col[P + t];
                /* checkpoint the conv state as it would be AFTER position t: the
                 * P raw inputs ending at t (col[t+1..t+P]), pre-forward state
                 * folded in automatically for the first P-1 positions. */
                if (ckpt) for (int j = 0; j < P; j++)
                    ckpt[((int64_t)t*C + ch)*P + j] = col[t + 1 + j];
            }
            for (int j = 0; j < P; j++) state[(int64_t)ch*P + j] = col[S + j];
        }
        free(col);
    }
}

/* ---------- config loading ----------
 * Accepts both the flat text config (tiny oracle via InklingForCausalLM) and
 * the full multimodal config.json (real checkpoint, fields under text_config). */
static double jnum(jval *o, const char *k, double dflt) {
    jval *v = json_get(o, k);
    return (v && v->t == J_NUM) ? v->num : dflt;
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *root = json_parse(buf, &arena);
    jval *r = json_get(root, "text_config"); if (!r) r = root;

    c->hidden      = (int)jnum(r,"hidden_size",6144);
    c->n_layers    = (int)jnum(r,"num_hidden_layers",66);
    c->vocab       = (int)jnum(r,"vocab_size",201024);
    c->unpad_vocab = (int)jnum(r,"unpadded_vocab_size",c->vocab);
    c->n_heads     = (int)jnum(r,"num_attention_heads",64);
    c->n_kv        = (int)jnum(r,"num_key_value_heads",8);
    c->head_dim    = (int)jnum(r,"head_dim",128);
    c->swa_heads   = (int)jnum(r,"swa_num_attention_heads",c->n_heads);
    c->swa_kv      = (int)jnum(r,"swa_num_key_value_heads",16);
    c->swa_hd      = (int)jnum(r,"swa_head_dim",c->head_dim);
    c->window      = (int)jnum(r,"sliding_window_size",512);
    c->d_rel       = (int)jnum(r,"d_rel",16);
    c->rel_extent  = (int)jnum(r,"rel_extent",1024);
    c->log_floor   = jnum(r,"log_scaling_n_floor",0);
    c->log_alpha   = (float)jnum(r,"log_scaling_alpha",0.1);
    c->conv_k      = (int)jnum(r,"sconv_kernel_size", jnum(r,"conv_kernel_size",4));
    c->n_experts   = (int)jnum(r,"n_routed_experts",256);
    c->topk        = (int)jnum(r,"num_experts_per_tok",6);
    c->n_shared    = (int)jnum(r,"n_shared_experts",2);
    c->eps         = (float)jnum(r,"rms_norm_eps",1e-6);
    c->route_scale = (float)jnum(r,"route_scale",8.0);
    c->mup         = (float)jnum(r,"logits_mup_width_multiplier",24.0);
    /* eos lives at the top level in the real multimodal config, in the text
     * config for a flat snapshot; may be null (tiny oracle) */
    jval *eo = json_get(root,"eos_token_id");
    if (!eo || eo->t != J_NUM) eo = json_get(r,"eos_token_id");
    c->eos = (eo && eo->t == J_NUM) ? (int)eo->num : -1;
    /* multimodal placeholder tokens (InklingConfig defaults) — text-only engine
     * rejects prompts containing them rather than embedding a meaningless row */
    c->img_tok = (int)jnum(root, "image_token_id", 200054);
    c->aud_tok = (int)jnum(root, "audio_token_id", 200053);
    /* serve KV bound: model_max_length is 1M but per-request KV can't be; cap to
     * a memory-safe default, overridable via CTX_MAX */
    { const char *e = getenv("CTX_MAX"); c->ctx_max = e ? atoi(e) : 8192; }
    /* real config.json: intermediate_size = MoE, dense_intermediate_size = dense.
     * HF-saved config (post_init applied): intermediate_size = dense, moe_intermediate_size = MoE. */
    jval *dis = json_get(r,"dense_intermediate_size");
    if (dis && dis->t == J_NUM) {
        c->dense_inter = (int)dis->num;
        c->moe_inter   = (int)jnum(r,"intermediate_size",3072);
    } else {
        c->dense_inter = (int)jnum(r,"intermediate_size",24576);
        c->moe_inter   = (int)jnum(r,"moe_intermediate_size",3072);
    }
    if (c->n_layers > MAXL) { fprintf(stderr,"n_layers %d > MAXL\n", c->n_layers); exit(1); }

    /* attention layer types: explicit layer_types[] > local_layer_ids[] > (i+1)%6 rule */
    jval *lt = json_get(r,"layer_types");
    jval *ll = json_get(r,"local_layer_ids");
    for (int i = 0; i < c->n_layers; i++) {
        if (lt && lt->t == J_ARR) c->local[i] = (strcmp(lt->kids[i]->str,"hybrid_sliding")==0);
        else if (ll && ll->t == J_ARR) {
            c->local[i] = 0;
            for (int j = 0; j < ll->len; j++) if ((int)ll->kids[j]->num == i) { c->local[i] = 1; break; }
        } else c->local[i] = ((i + 1) % 6) != 0;
    }
    /* MLP types: explicit mlp_layer_types[] > dense_mlp_idx (first k layers dense) */
    jval *mt = json_get(r,"mlp_layer_types");
    int dense_idx = (int)jnum(r,"dense_mlp_idx",0);
    for (int i = 0; i < c->n_layers; i++) {
        if (mt && mt->t == J_ARR) c->sparse[i] = (strcmp(mt->kids[i]->str,"sparse")==0);
        else c->sparse[i] = (i >= dense_idx);
    }
    free(buf); free(arena);
}

/* ---------- weight loading ---------- */
static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}
static float load_scalar(Model *m, const char *name, float dflt) {
    if (!st_has(&m->S, name)) return dflt;
    float v; st_read_f32(&m->S, name, &v, 0); return v;
}

/* chunked pread: a single pread caps at ~2.1 GB on Linux, and the bf16
 * embed/lm_head tensors are 2.47 GB — loop in 1 GB slices */
static void pread_all(int fd, void *buf, int64_t nb, int64_t off) {
    char *p = buf;
    while (nb > 0) {
        int64_t chunk = nb < (1<<30) ? nb : (1<<30);
        ssize_t got = pread(fd, p, (size_t)chunk, off);
        if (got <= 0) { perror("pread chunk"); exit(1); }
        p += got; off += got; nb -= got;
    }
}

/* big matmul weights keep their on-disk dtype resident: BF16 raw (real
 * checkpoint, halves RAM), anything else as f32 (tiny oracle: bit-exact).
 * gpu_ok: bf16 tensors move to VRAM while budget lasts (embed stays host —
 * it's a row lookup, not a matmul). */
static Wt load_w(Model *m, const char *name, int gpu_ok, int cols) {
    Wt w = {0};
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "missing %s\n", name); exit(1); }
    if (t->dtype == 0) {
        w.h = malloc(t->nbytes); if (!w.h) { fprintf(stderr,"OOM %s\n",name); exit(1); }
        pread_all(t->fd, w.h, t->nbytes, t->off);
        lora_merge_resident(&m->lora, name, w.h, NULL, t->numel);
#ifdef COLI_CUDA
        /* int8 residents (DEFAULT under CUDA; Q8=0 opts out): quantize this
         * resident to per-row int8 and keep it in VRAM at half the bf16 footprint,
         * freeing VRAM for the expert tier (measured 402 → 780 experts, lossless:
         * ppl 7.35 vs 8.01 bf16, the int8 path uses full-f32 activations). `cols` =
         * the input dim, so rows = numel/cols is always the TRUE tensor row count
         * (correct even for lm_head, whose matmul O is the unpadded vocab < padded
         * rows). cols=0 skips (fused/sliced shared experts + the embedding lookup). */
        const char *q8e = getenv("Q8");
        int q8_on = !q8e || strcmp(q8e, "0") != 0;   /* on unless Q8=0 */
        if (g_cuda && gpu_ok && cols > 0 && q8_on &&
            ink_cuda_free_bytes() > (size_t)t->numel + (size_t)(t->numel/cols)*4 + (3ULL<<30)) {
            int64_t rows = t->numel / cols;
            int8_t *q8 = malloc(t->numel); float *qs = malloc((size_t)rows*4);
            #pragma omp parallel for schedule(static)
            for (int64_t o = 0; o < rows; o++) {
                float mx = 0.f;
                for (int64_t i = 0; i < cols; i++) { float a = fabsf(bf16_to_f32(w.h[o*cols+i])); if (a>mx) mx=a; }
                float sc = mx/127.f; if (sc < 1e-12f) sc = 1e-12f;
                qs[o] = sc; float inv = 1.f/sc;
                for (int64_t i = 0; i < cols; i++) q8[o*cols+i] = (int8_t)lrintf(bf16_to_f32(w.h[o*cols+i])*inv);
            }
            w.dq8 = ink_cuda_upload(q8, t->numel); w.dqs = ink_cuda_upload(qs, (size_t)rows*4);
            free(q8); free(qs);
            if (w.dq8 && w.dqs) { free(w.h); w.h = NULL; return w; }
            w.dq8 = w.dqs = NULL;   /* upload failed: fall through to bf16 */
        }
        /* keep 3 GB VRAM headroom for the activation buffers + future tiers */
        if (g_cuda && gpu_ok && ink_cuda_free_bytes() > (size_t)t->nbytes + (3ULL<<30)) {
            w.dev = ink_cuda_upload(w.h, t->nbytes);
            if (w.dev) { free(w.h); w.h = NULL; }
        }
#else
        (void)gpu_ok; (void)cols;
#endif
    } else {
        w.f = falloc(t->numel);
        st_read_f32(&m->S, name, w.f, 0);
        lora_merge_resident(&m->lora, name, NULL, w.f, t->numel);
    }
    return w;
}
static Wt wt_off(Wt w, int64_t off) {
    Wt r = { w.f ? w.f + off : NULL, w.h ? w.h + off : NULL,
             w.dev ? (char*)w.dev + off*2 : NULL, NULL, NULL };   /* bf16 only (shared experts) */
    return r;
}
static void wt_row_f32(Wt w, int64_t off, float *out, int n) {
    if (w.f) memcpy(out, w.f + off, n * sizeof(float));
    else for (int i = 0; i < n; i++) { union { uint32_t u; float f; } v = { (uint32_t)w.h[off + i] << 16 }; out[i] = v.f; }
}

/* f32 slice of a (possibly bf16/f16) tensor: element offset + count.
 * Needed to stream one expert out of the fused [E,2I,D]/[E,D,I] tensors. */
static void read_f32_slice(shards *S, const char *name, float *out, int64_t off, int64_t cnt) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->dtype == 3) { fprintf(stderr, "%s: U8 container has no f32 view\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    void *raw = malloc((size_t)cnt * esz);
    if (!raw) { fprintf(stderr,"OOM slice %s\n",name); exit(1); }
    if (pread(t->fd, raw, (size_t)cnt*esz, t->off + off*esz) != (ssize_t)(cnt*esz)) { perror("pread slice"); exit(1); }
    if (t->dtype == 2) memcpy(out, raw, (size_t)cnt*4);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < cnt; i++) out[i] = bf16_to_f32(p[i]); }
    else                    { uint16_t *p = raw; for (int64_t i = 0; i < cnt; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    posix_fadvise(t->fd, t->off + off*esz, cnt*esz, POSIX_FADV_DONTNEED);
}

/* raw byte slice of a U8 container tensor */
static void read_u8_slice(shards *S, const char *name, uint8_t *out, int64_t boff, int64_t nb) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (pread(t->fd, out, (size_t)nb, t->off + boff) != (ssize_t)nb) { perror("pread u8 slice"); exit(1); }
    posix_fadvise(t->fd, t->off + boff, nb, POSIX_FADV_DONTNEED);
}

/* container rows -> int8: rowb==cols is int8 verbatim; rowb==cols/2 is packed
 * int4 (low nibble = even column, offset +8 — convert_inkling_int4.py / glm.c) */
static void unpack_rows(const uint8_t *raw, int8_t *q, int64_t rows, int64_t cols, int64_t rowb) {
    if (rowb == cols) { memcpy(q, raw, (size_t)(rows*cols)); return; }
    if (rowb*2 != cols) { fprintf(stderr, "container row size %ld vs cols %ld unsupported\n", (long)rowb, (long)cols); exit(1); }
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *b = raw + r*rowb;
        int8_t *qr = q + r*cols;
        for (int64_t j = 0; j < rowb; j++) {
            qr[2*j]   = (int8_t)((b[j] & 0xF) - 8);
            qr[2*j+1] = (int8_t)((b[j] >> 4) - 8);
        }
    }
}

static double mem_avail_bytes(void);

static void mtp_load(Model *m);
static void fe_init(Model *m);

static void model_init(Model *m, const char *snap, int cap, int bits, const char *lora_dir) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    int D = c->hidden, K = c->conv_k;
    double t0 = now_s();
    /* open the adapter before residents load: load_w merges resident deltas
     * in place as each tensor is read (and before any CUDA upload) */
    if (lora_dir) lora_open(&m->lora, lora_dir, c->n_layers, D, c->moe_inter,
                            c->n_shared, c->n_experts, c->sparse);
#ifdef COLI_CUDA
    if (!getenv("NOGPU")) {
        int dev = getenv("GPU_DEV") ? atoi(getenv("GPU_DEV")) : 0;
        if (ink_cuda_init(dev) == 0) {
            g_cuda = 1;
            fprintf(stderr, "[cuda] device %d ready, %.1f GB free — bf16 residents to VRAM\n",
                    dev, ink_cuda_free_bytes()/1e9);
        } else fprintf(stderr, "[cuda] init failed, running on CPU\n");
    }
#endif
    m->embed      = load_w(m, "model.embed_tokens.weight", 0, 0);
    m->embed_norm = st_has(&m->S,"model.embed_norm.weight") ? load_t(m,"model.embed_norm.weight") : NULL;
    m->final_norm = load_t(m, "model.norm.weight");
    m->lm_head    = load_w(m, "lm_head.weight", 1, c->hidden);
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[320];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        int H = L_HEADS(c,i), hd = L_HD(c,i);   /* o_proj input dim = H*hd, for per-row int8 quant */
        #define LD(field, suffix)  snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        #define LDW(field, suffix, O) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_w(m,nm,1,O)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LDW(q, "self_attn.q_proj.weight", D);  LDW(k, "self_attn.k_proj.weight", D);
        LDW(v, "self_attn.v_proj.weight", D);  LDW(r, "self_attn.r_proj.weight", D);
        LDW(o, "self_attn.o_proj.weight", H*hd);
        LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        LD(relp, "self_attn.rel_logits_proj.proj");
        LD(k_cw, "self_attn.k_sconv.conv1d.weight");
        LD(v_cw, "self_attn.v_sconv.conv1d.weight");
        LD(a_cw, "attn_sconv.conv1d.weight");
        LD(m_cw, "mlp_sconv.conv1d.weight");
        if (!c->sparse[i]) {
            LDW(dg, "mlp.gate_proj.weight", D); LDW(du, "mlp.up_proj.weight", D);
            LDW(dd, "mlp.down_proj.weight", c->dense_inter);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.global_scale",i); l->dgs = load_scalar(m,nm,1.f);
        } else {
            LD(router, "mlp.gate.weight");
            LD(rbias,  "mlp.gate.e_score_correction_bias");
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.global_scale",i); l->rgs = load_scalar(m,nm,1.f);
            LDW(sh_g, "mlp.shared_experts.gate_proj", 0);   /* fused/sliced (wt_off) -> keep bf16 */
            LDW(sh_u, "mlp.shared_experts.up_proj", 0);
            LDW(sh_d, "mlp.shared_experts.down_proj", 0);
        }
        #undef LD
        #undef LDW
        /* conv states: raw inputs of the previous K-1 steps, zero-init */
        int kvdim = L_KV(c,i) * L_HD(c,i);
        for (int j = 0; j < 4; j++) {
            if (!m->cs[j]) m->cs[j] = calloc(c->n_layers, sizeof(float*));
            int C = (j < 2) ? kvdim : D;
            m->cs[j][i] = calloc((int64_t)C * (K-1), sizeof(float));
        }
    }
    /* routed-expert LoRA tensors go resident now, BEFORE the cache auto-cap
     * reads MemAvailable, so the budget accounts for them */
    lora_load_experts(&m->lora);
    /* container detection: converted snapshots store experts as U8 + .qs.
     * rb13/rb2 = bytes per packed row (D/2|D and I/2|I for int4|int8) */
    int64_t I = c->moe_inter, E = c->n_experts;
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",i);
        st_tensor *t = st_find(&m->S, nm);
        if (t && t->dtype == 3) {
            m->xq = 1;
            m->rb13 = t->nbytes / (E * 2*I);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",i);
            st_tensor *t2 = st_find(&m->S, nm);
            m->rb2 = t2->nbytes / (E * (int64_t)D);
            if (m->rb13 != D && m->rb13*2 != D) { fprintf(stderr,"unsupported container row size %lld\n",(long long)m->rb13); exit(1); }
        }
        break;
    }
    int nsp = 0; for (int i = 0; i < c->n_layers; i++) nsp += c->sparse[i];
    int64_t slotb = m->xq ? m->rb13*2*I + m->rb2*D + (2*I+D)*4
                  : m->quant_bits ? 3*I*D + (2*I+D)*4 : 3*I*D*4;
    if (cap <= 0) {   /* auto: fit the LRU in available RAM, 20% + 4 GB headroom */
        double avail = mem_avail_bytes();
        cap = avail > 0 ? (int)((avail*0.80 - 4e9) / ((double)slotb * (nsp ? nsp : 1))) : 16;
        if (cap < 4) cap = 4;
        if (cap > c->n_experts) cap = c->n_experts;
        fprintf(stderr, "[cap auto] %d experts/layer (%.1f GB cache budget)\n",
                cap, (double)cap*slotb*nsp/1e9);
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) { m->cache[i].cap = cap; m->cache[i].slots = calloc(cap, sizeof(Slot)); }
    fe_init(m);   /* async chunked expert-fill engine (FILL_ASYNC=0 disables) */
    /* usage counters; seeded from a previous run's history when present */
    m->eusage = calloc(c->n_layers, sizeof(uint32_t*));
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) m->eusage[i] = calloc(E, 4);
    /* per-generation "touched" set: classifies a miss as cold-first-touch
     * (never seen this generation → unpredictable) vs churn (seen then evicted
     * → prefetchable). Decides whether prefetch is the right lever. */
    m->seen = calloc(c->n_layers, sizeof(uint8_t*));
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) m->seen[i] = calloc(E, 1);
    /* overfetch probe: track prev decode token's top-M router ranking per layer,
     * to see if cold-first-touch experts were recent near-misses (i.e. catchable
     * by over-fetching the router's runners-up — a lossless prefetch). */
    m->topm = getenv("OVERFETCH_M") ? atoi(getenv("OVERFETCH_M")) : 16;
    if (m->topm > E) m->topm = E;
    m->prev_topm = calloc(c->n_layers, sizeof(int*));
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) {
        m->prev_topm[i] = malloc(m->topm * sizeof(int));
        for (int j = 0; j < m->topm; j++) m->prev_topm[i][j] = -1;
    }
    /* opt-in: the 10.5 GB draft head only loads when speculation is requested
     * (MTP=1, or the --mtp-* / spec paths force-load it). Normal serve/decode
     * pays nothing until the spec serve loop is wired up. */
    if (getenv("MTP") && strcmp(getenv("MTP"), "0")) mtp_load(m);
    m->dense_load_s = now_s() - t0;
}

/* Load the MTP speculative-draft head (out-mtp.safetensors, indexed by the same
 * shards as the main model). Each module reuses the Layer struct + the dense
 * block primitives; its KV/conv live at slot n_layers+k so attention() and
 * sconv_apply() are reused verbatim. MTP=0 disables. */
static void mtp_load(Model *m) {
    Cfg *c = &m->c; int D = c->hidden, K = c->conv_k;
    char nm[320];
    int nmax = 0;
    for (int k = 0; c->n_layers + k < MAXL; k++) {
        snprintf(nm, sizeof(nm), "model.mtp.%d.input_proj.weight", k);
        if (!st_has(&m->S, nm)) break;
        nmax = k + 1;
    }
    if (!nmax) return;
    m->n_mtp = nmax;
    m->mtp = calloc(nmax, sizeof(MtpMod));
    for (int j = 0; j < 4; j++)
        m->cs[j] = realloc(m->cs[j], (int64_t)(c->n_layers + nmax) * sizeof(float*));
    for (int k = 0; k < nmax; k++) {
        MtpMod *mm = &m->mtp[k]; Layer *l = &mm->L; int li = c->n_layers + k;
        /* gpu_ok=0: keep the draft head resident in RAM. It is small and its
         * forward is cheap, and on the real model VRAM is already ~full with the
         * main bf16 residents — uploading 10 GB more would OOM the A6000. */
        #define LD(field, suffix)  snprintf(nm,sizeof(nm),"model.mtp.%d." suffix,k); l->field = load_t(m,nm)
        #define LDW(field, suffix) snprintf(nm,sizeof(nm),"model.mtp.%d." suffix,k); l->field = load_w(m,nm,0,0)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LDW(q, "self_attn.q_proj.weight"); LDW(k, "self_attn.k_proj.weight");
        LDW(v, "self_attn.v_proj.weight"); LDW(r, "self_attn.r_proj.weight");
        LDW(o, "self_attn.o_proj.weight");
        LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        LD(relp, "self_attn.rel_logits_proj.proj");
        LD(k_cw, "self_attn.k_sconv.conv1d.weight");
        LD(v_cw, "self_attn.v_sconv.conv1d.weight");
        LD(a_cw, "attn_sconv.conv1d.weight");
        LD(m_cw, "mlp_sconv.conv1d.weight");
        LDW(dg, "mlp.gate_proj.weight"); LDW(du, "mlp.up_proj.weight"); LDW(dd, "mlp.down_proj.weight");
        snprintf(nm,sizeof(nm),"model.mtp.%d.mlp.global_scale",k); l->dgs = load_scalar(m,nm,1.f);
        #undef LD
        #undef LDW
        snprintf(nm,sizeof(nm),"model.mtp.%d.input_proj.weight",k);  mm->in_proj     = load_w(m,nm,0,0);
        snprintf(nm,sizeof(nm),"model.mtp.%d.embed_norm.weight",k);  mm->embed_norm  = load_t(m,nm);
        snprintf(nm,sizeof(nm),"model.mtp.%d.hidden_norm.weight",k); mm->hidden_norm = load_t(m,nm);
        snprintf(nm,sizeof(nm),"model.mtp.%d.mlp.gate_proj.weight",k);
        mm->inter = (int)(st_numel(&m->S, nm) / D);
        snprintf(nm,sizeof(nm),"model.mtp.%d.self_attn.rel_logits_proj.proj",k);
        int ext = (int)(st_numel(&m->S, nm) / c->d_rel);
        c->local[li] = (ext == c->window) ? 1 : 0;      /* sliding vs global */
        int kvdim = L_KV(c,li) * L_HD(c,li);
        for (int j = 0; j < 4; j++) {
            int Cc = (j < 2) ? kvdim : D;
            m->cs[j][li] = calloc((int64_t)Cc * (K-1), sizeof(float));
        }
    }
    fprintf(stderr, "[mtp] %d draft modules loaded\n", nmax);
}

static double mem_avail_bytes(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char ln[256]; double kb = 0;
    while (fgets(ln, sizeof(ln), f)) if (sscanf(ln, "MemAvailable: %lf", &kb) == 1) break;
    fclose(f);
    return kb * 1024.0;
#else
    return 0;
#endif
}

/* ---------- routed-expert slots: serial bookkeeping, parallel fills ---------- */
static void fill_wait(Slot *s);

static Slot *slot_find(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        lc->slots[i].used = ++m->clock;
        return &lc->slots[i];
    }
    return NULL;
}

/* residency check without touching the LRU clock (speculative overfetch) */
static Slot *slot_peek(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) return &lc->slots[i];
    return NULL;
}

/* per-container-mode slot buffer allocation, shared by cache and transient slots */
static void slot_alloc_bufs(Model *m, Slot *s) {
    Cfg *c = &m->c;
    int64_t D = c->hidden, I = c->moe_inter, n13 = 2*I*D, n2 = D*I;
    if (m->xq)              { s->p13 = malloc((size_t)(m->rb13*2*I)); s->p2 = malloc((size_t)(m->rb2*D));
                              s->s13 = falloc(2*I); s->s2 = falloc(D);
                              if (!s->p13 || !s->p2) { fprintf(stderr,"OOM expert slot\n"); exit(1); } }
    else if (m->quant_bits) { s->q13 = malloc(n13); s->q2 = malloc(n2);
                              s->s13 = falloc(2*I); s->s2 = falloc(D);
                              if (!s->q13 || !s->q2) { fprintf(stderr,"OOM expert slot\n"); exit(1); } }
    else                    { s->f13 = falloc(n13); s->f2 = falloc(n2); }
}

/* allocate a slot (or evict the LRU non-pinned one); serial callers only.
 * Never evicts a slot the current moe() call still references (busy) or one
 * with async fill chunks in flight (workers are writing into its buffers) —
 * the pre-existing behavior of evicting a busy slot silently computed earlier
 * positions of an S>1 batch with the WRONG expert's weights whenever a layer
 * call's working set exceeded cap. When no victim exists: spill=1 falls back
 * to a transient slot (freed at the end of the call), spill=0 returns NULL
 * (speculative callers just skip). LFRU=1 switches the victim policy to
 * tier.h's frequency-first score over the accumulated usage counts. */
static Slot *slot_acquire(Model *m, int layer, int eid, int spill) {
    LCache *lc = &m->cache[layer];
    Slot *s = NULL;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        slot_alloc_bufs(m, s);
    } else {
        static int lfru = -1;
        if (lfru < 0) { const char *ev = getenv("LFRU"); lfru = ev && atoi(ev) ? 1 : 0; }
        const uint32_t *heat = m->eusage[layer];
        for (;;) {
            int lru = -1;
            for (int i = 0; i < lc->n; i++) {
                Slot *si = &lc->slots[i];
                if (si->pinned || si->busy || __atomic_load_n(&si->pending, __ATOMIC_ACQUIRE)) continue;
                if (lru < 0) { lru = i; continue; }
                Slot *sl = &lc->slots[lru];
                if (lfru && heat) {
                    if (tier_lfru_score(heat[si->eid], (uint32_t)si->used, (uint32_t)m->clock) <
                        tier_lfru_score(heat[sl->eid], (uint32_t)sl->used, (uint32_t)m->clock)) lru = i;
                } else if (si->used < sl->used) lru = i;
            }
            if (lru >= 0) { s = &lc->slots[lru]; break; }
            /* no victim yet: an idle in-flight fill becomes evictable once it
             * lands — wait for one and rescan before giving up */
            Slot *w = NULL;
            for (int i = 0; i < lc->n; i++) {
                Slot *si = &lc->slots[i];
                if (!si->pinned && !si->busy && __atomic_load_n(&si->pending, __ATOMIC_ACQUIRE)) { w = si; break; }
            }
            if (!w) break;
            fill_wait(w);
        }
        if (!s) {
            if (!spill) return NULL;
            s = calloc(1, sizeof(Slot));
            if (!s) { fprintf(stderr, "OOM transient slot\n"); exit(1); }
            slot_alloc_bufs(m, s);
            s->transient = 1;
        }
    }
    s->eid = eid; s->used = ++m->clock; s->filled = 0; s->pending = 0; s->pinned = 0; s->busy = 0;
    return s;
}

/* heat-tiered quant SIM (quality probe only): degrade a packed-int4 weight matrix
 * to `bits`-bit effective precision, per row, in place — re-quantize the int4
 * integer part (nibble-8) to a 2^bits symmetric grid sharing the same row scale.
 * Lets us measure sub-int4 perplexity with no new format/kernel. */
static void qsim_row(uint8_t *p, int64_t rows, int64_t nb /*bytes/row = cols/2*/, int bits) {
    int lv = 1 << (bits - 1);   /* qi in [-lv, lv-1]; bits=2 -> 4 levels */
    for (int64_t r = 0; r < rows; r++) {
        uint8_t *row = p + r*nb;
        int mx = 0;
        for (int64_t b = 0; b < nb; b++) {
            int lo = (row[b]&0xF)-8, hi = (row[b]>>4)-8;
            if (abs(lo) > mx) mx = abs(lo); if (abs(hi) > mx) mx = abs(hi);
        }
        if (mx == 0) continue;
        float step = (float)mx / lv;
        for (int64_t b = 0; b < nb; b++) {
            int lo = (row[b]&0xF)-8, hi = (row[b]>>4)-8;
            int qlo = lrintf(lo/step); if (qlo < -lv) qlo = -lv; if (qlo > lv-1) qlo = lv-1;
            int qhi = lrintf(hi/step); if (qhi < -lv) qhi = -lv; if (qhi > lv-1) qhi = lv-1;
            int vlo = lrintf(qlo*step)+8; if (vlo < 0) vlo = 0; if (vlo > 15) vlo = 15;
            int vhi = lrintf(qhi*step)+8; if (vhi < 0) vhi = 0; if (vhi > 15) vhi = 15;
            row[b] = (uint8_t)(vlo | (vhi<<4));
        }
    }
}
/* is this expert "cold" for the heat-tiered probe? (below the per-layer keep threshold) */
static int qsim_cold(Model *m, int layer, int eid) {
    if (!m->qsim_bits) return 0;
    if (m->qsim_all) return 1;
    return m->eusage[layer] && m->qsim_thr && m->eusage[layer][eid] < m->qsim_thr[layer];
}

/* ---------- async chunked expert fills ----------
 * The legacy miss path ran one synchronous 4-pread slot_fill per OMP task:
 * during decode that is 1-2 misses in flight, i.e. queue depth ~1-2 to the
 * NVMe, ~35 ms per ~28 MB expert on a drive that does several GB/s when fed.
 * Here every miss becomes ~FILL_CHUNK_KB-sized positioned reads spread over a
 * pool of I/O worker threads, so even a SINGLE miss keeps the drive at depth,
 * and moe() overlaps the wait with the cached experts' matmuls.
 * FILL_ASYNC=0 restores the legacy path (A/B baseline); the engine only
 * serves the U8 container + f32 .qs layout (everything else falls back). */
typedef struct {
    int fd; int64_t off, len; void *dst;
    Model *m; Slot *s; int layer;
} FillChunk;

static struct {
    int on, nth; int64_t chunk;
    pthread_t *th;
    pthread_mutex_t mu;
    pthread_cond_t more, space, done;     /* work available / queue space / a slot completed */
    FillChunk *q; int qcap, qhead, qtail, qn;
} g_fe;
static int g_overfetch = 0;               /* OVERFETCH=n speculative fills per layer per token */

static int fe_on(Model *m) { return g_fe.on && m->xq; }

/* block until a slot's fill has fully landed (no-op when already filled) */
static void fill_wait(Slot *s) {
    if (!g_fe.on || __atomic_load_n(&s->filled, __ATOMIC_ACQUIRE)) return;
    pthread_mutex_lock(&g_fe.mu);
    while (!__atomic_load_n(&s->filled, __ATOMIC_ACQUIRE))
        pthread_cond_wait(&g_fe.done, &g_fe.mu);
    pthread_mutex_unlock(&g_fe.mu);
}

static void *fe_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_fe.mu);
        while (!g_fe.qn) pthread_cond_wait(&g_fe.more, &g_fe.mu);
        FillChunk c = g_fe.q[g_fe.qhead];
        g_fe.qhead = (g_fe.qhead + 1) % g_fe.qcap; g_fe.qn--;
        pthread_cond_signal(&g_fe.space);
        pthread_mutex_unlock(&g_fe.mu);
        char *p = c.dst; int64_t got = 0;
        while (got < c.len) {           /* POSIX allows short reads on regular files */
            ssize_t r = pread(c.fd, p + got, (size_t)(c.len - got), c.off + got);
            if (r <= 0) { perror("pread expert chunk"); exit(1); }
            got += r;
        }
        posix_fadvise(c.fd, c.off, c.len, POSIX_FADV_DONTNEED);
        if (__atomic_sub_fetch(&c.s->pending, 1, __ATOMIC_ACQ_REL) == 0) {
            /* last chunk of this expert: run the qsim requant (needs the whole
             * weight in place), publish, and wake any waiter */
            Model *m = c.m; Slot *s = c.s;
            if (qsim_cold(m, c.layer, s->eid)) {
                qsim_row(s->p13, 2*(int64_t)m->c.moe_inter, m->rb13, m->qsim_bits);
                qsim_row(s->p2,  m->c.hidden,               m->rb2,  m->qsim_bits);
            }
            __atomic_store_n(&s->filled, 1, __ATOMIC_RELEASE);
            pthread_mutex_lock(&g_fe.mu);
            pthread_cond_broadcast(&g_fe.done);
            pthread_mutex_unlock(&g_fe.mu);
        }
    }
    return NULL;
}

/* push one read split into <=ch-sized chunks; caller holds g_fe.mu */
static void fe_push_locked(Model *m, int layer, Slot *s, int fd,
                           int64_t off, int64_t len, void *dst, int64_t ch) {
    for (int64_t o = 0; o < len; o += ch) {
        while (g_fe.qn == g_fe.qcap) pthread_cond_wait(&g_fe.space, &g_fe.mu);
        FillChunk *q = &g_fe.q[g_fe.qtail];
        g_fe.qtail = (g_fe.qtail + 1) % g_fe.qcap; g_fe.qn++;
        q->fd = fd; q->off = off + o; q->len = len - o < ch ? len - o : ch;
        q->dst = (char*)dst + o; q->m = m; q->s = s; q->layer = layer;
        pthread_cond_signal(&g_fe.more);
    }
}

/* queue one expert's four reads; pending counts chunks, the last worker to
 * finish publishes filled=1. Returns 0 when the engine is unavailable. */
static int fill_submit(Model *m, int layer, Slot *s) {
    if (!fe_on(m)) return 0;
    Cfg *c = &m->c; int64_t I = c->moe_inter, D = c->hidden, eid = s->eid;
    char nm[320], qs[340];
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.gate_up_proj", layer);
    snprintf(qs, sizeof(qs), "%s.qs", nm);
    st_tensor *t13 = st_find(&m->S, nm), *tq13 = st_find(&m->S, qs);
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.down_proj", layer);
    snprintf(qs, sizeof(qs), "%s.qs", nm);
    st_tensor *t2 = st_find(&m->S, nm), *tq2 = st_find(&m->S, qs);
    if (!t13 || !tq13 || !t2 || !tq2) { fprintf(stderr, "missing expert tensors, layer %d\n", layer); exit(1); }
    int64_t nb13 = 2*I*m->rb13, nb2 = D*m->rb2, CH = g_fe.chunk;
    int nch = (int)((nb13 + CH-1)/CH + (nb2 + CH-1)/CH) + 2;
    __atomic_store_n(&s->filled, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&s->pending, nch, __ATOMIC_RELEASE);
    pthread_mutex_lock(&g_fe.mu);
    fe_push_locked(m, layer, s, t13->fd,  t13->off  + eid*nb13,  nb13,  s->p13, CH);
    fe_push_locked(m, layer, s, t2->fd,   t2->off   + eid*nb2,   nb2,   s->p2,  CH);
    fe_push_locked(m, layer, s, tq13->fd, tq13->off + eid*2*I*4, 2*I*4, s->s13, INT64_MAX);
    fe_push_locked(m, layer, s, tq2->fd,  tq2->off  + eid*D*4,   D*4,   s->s2,  INT64_MAX);
    pthread_mutex_unlock(&g_fe.mu);
    return 1;
}

/* speculative runner-up fills: this decode token's router ranking [0..topm)
 * at this layer was just recorded in prev_topm; ranks beyond top-K are the
 * "recent near-miss" class the overfetch probe reports as catchable. The cap
 * headroom guard keeps speculation from evicting this token's working set. */
static void overfetch_submit(Model *m, int layer) {
    LCache *lc = &m->cache[layer];
    int K = m->c.topk;
    if (!m->prev_topm[layer]) return;
    if (lc->cap - m->npin < K + g_overfetch + 4) return;
    int nnew = 0;
    for (int r = 0; r < m->topm && nnew < g_overfetch; r++) {
        int eid = m->prev_topm[layer][r];
        if (eid < 0) break;
        if (slot_peek(m, layer, eid)) continue;   /* resident or already in flight */
        Slot *s = slot_acquire(m, layer, eid, 0);  /* never spill for speculation */
        if (!s) break;
        fill_submit(m, layer, s);
        nnew++;
    }
}

static void fe_init(Model *m) {
    if (!m->xq) return;
    const char *e = getenv("FILL_ASYNC");
    if (e && !atoi(e)) { fprintf(stderr, "[fill] async I/O disabled (FILL_ASYNC=0)\n"); return; }
    /* the workers read the .qs scales raw, so they must be f32 on disk (they
     * are in convert_inkling_int4.py containers; anything else falls back) */
    Cfg *c = &m->c; char nm[336];
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) {
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.gate_up_proj.qs", i);
        st_tensor *t = st_find(&m->S, nm);
        if (!t || t->dtype != 2) { fprintf(stderr, "[fill] non-f32 .qs scales, async I/O off\n"); return; }
        break;
    }
    g_fe.nth = getenv("FILL_THREADS") ? atoi(getenv("FILL_THREADS")) : 16;
    if (g_fe.nth < 1) g_fe.nth = 1;
    if (g_fe.nth > 128) g_fe.nth = 128;
    int64_t kb = getenv("FILL_CHUNK_KB") ? atoll(getenv("FILL_CHUNK_KB")) : 2048;
    if (kb < 1) kb = 1;
    g_fe.chunk = kb * 1024;
    g_overfetch = getenv("OVERFETCH") ? atoi(getenv("OVERFETCH")) : 0;
    if (g_overfetch < 0) g_overfetch = 0;
    g_fe.qcap = 8192;
    g_fe.q = malloc((size_t)g_fe.qcap * sizeof(FillChunk));
    g_fe.th = malloc((size_t)g_fe.nth * sizeof(pthread_t));
    if (!g_fe.q || !g_fe.th) { fprintf(stderr, "OOM fill queue\n"); exit(1); }
    pthread_mutex_init(&g_fe.mu, NULL);
    pthread_cond_init(&g_fe.more, NULL);
    pthread_cond_init(&g_fe.space, NULL);
    pthread_cond_init(&g_fe.done, NULL);
    for (int i = 0; i < g_fe.nth; i++)
        if (pthread_create(&g_fe.th[i], NULL, fe_worker, NULL)) { perror("pthread_create"); exit(1); }
    g_fe.on = 1;
    fprintf(stderr, "[fill] async chunked expert I/O: %d workers x %lld KB chunks%s\n",
            g_fe.nth, (long long)kb, g_overfetch ? " + overfetch" : "");
}

/* pure I/O (+ optional requant): safe to run in parallel across slots */
static void slot_fill(Model *m, int layer, Slot *s) {
    Cfg *c = &m->c;
    int64_t D = c->hidden, I = c->moe_inter, n13 = 2*I*D, n2 = D*I;
    int64_t eid = s->eid;
    char nm[320], qs[340];
    if (fill_submit(m, layer, s)) { fill_wait(s); return; }
    if (m->xq) {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        read_u8_slice(&m->S, nm, s->p13, eid*2*I*m->rb13, 2*I*m->rb13);
        snprintf(qs,sizeof(qs),"%s.qs",nm);
        read_f32_slice(&m->S, qs, s->s13, eid*2*I, 2*I);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        read_u8_slice(&m->S, nm, s->p2, eid*D*m->rb2, D*m->rb2);
        snprintf(qs,sizeof(qs),"%s.qs",nm);
        read_f32_slice(&m->S, qs, s->s2, eid*D, D);
        if (qsim_cold(m, layer, (int)eid)) {   /* heat-tiered quant probe */
            qsim_row(s->p13, 2*I, m->rb13, m->qsim_bits);
            qsim_row(s->p2,  D,   m->rb2,  m->qsim_bits);
        }
    } else if (m->quant_bits) {
        float *tmp = falloc(n13 > n2 ? n13 : n2);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        read_f32_slice(&m->S, nm, tmp, eid*n13, n13);
        quantize_rows(tmp, s->q13, s->s13, 2*I, D, m->quant_bits);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        read_f32_slice(&m->S, nm, tmp, eid*n2, n2);
        quantize_rows(tmp, s->q2, s->s2, D, I, m->quant_bits);
        free(tmp);
    } else {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        read_f32_slice(&m->S, nm, s->f13, eid*n13, n13);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        read_f32_slice(&m->S, nm, s->f2, eid*n2, n2);
    }
    s->filled = 1;
}

/* pin the top-N experts per sparse layer from a usage-history file (colibri
 * .coli_usage convention: one uint32 count per expert per layer). Pins are
 * regular cache slots flagged non-evictable, filled in parallel at startup.
 * Toggles: PIN=off (or PIN=0) skips cache warming entirely (no seeding, no
 * pins, cold LRU start); PIN_N=0 seeds the ranking from the history but pins
 * nothing; PIN=<path> uses an alternate history file; PIN_N=<n> pin depth. */
/* sort pinned-expert indices by usage descending (for VRAM upload order) */
static const uint32_t *g_pin_use;
static int u32_desc(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t*)a, ub = *(const uint32_t*)b;
    return ua < ub ? 1 : ua > ub ? -1 : 0;
}
static int pin_use_desc(const void *a, const void *b) {
    uint32_t ua = g_pin_use[*(const int*)a], ub = g_pin_use[*(const int*)b];
    return ua < ub ? 1 : ua > ub ? -1 : 0;
}

static void pins_load(Model *m, const char *snap) {
    Cfg *c = &m->c; int E = c->n_experts;
    char up[2048];
    const char *env = getenv("PIN");
    if (env && (!strcmp(env, "off") || !strcmp(env, "0"))) {
        fprintf(stderr, "[pin] cache warming disabled (PIN=%s)\n", env);
        return;
    }
    if (env) snprintf(up, sizeof(up), "%s", env);
    else snprintf(up, sizeof(up), "%s/.coli_usage", snap);
    FILE *f = fopen(up, "rb");
    if (!f) return;
    uint32_t hdr[3];
    if (fread(hdr, 4, 3, f) != 3 || hdr[0] != 0x31554B49u ||
        (int)hdr[1] != c->n_layers || (int)hdr[2] != E) {
        fprintf(stderr, "[pin] %s: not an inkling usage file, ignoring\n", up);
        fclose(f); return;
    }
    int cap = m->cache[0].cap;
    /* default: pin half the cap. Measured on the 975B: cap/4 (19/layer) gave
     * 83.6% hit / 0.32 tok/s; 40/layer gave 95.6% / 0.80 tok/s — decode fills
     * run at queue depth ~1, so every pinned expert removes a ~35ms stall. */
    m->npin = getenv("PIN_N") ? atoi(getenv("PIN_N")) : cap/2;
    if (m->npin > cap - 8) m->npin = cap - 8;
    if (m->npin < 0) m->npin = 0;
    /* heat-tiered quant probe: QSIM_BITS=2|3 degrades cold experts to N-bit;
     * QSIM_ALL=1 degrades all (uniform control); QSIM_HOT_KEEP=N experts/layer
     * kept at int4 (default = npin, i.e. the pinned hot set). */
    m->qsim_bits = getenv("QSIM_BITS") ? atoi(getenv("QSIM_BITS")) : 0;
    m->qsim_all  = getenv("QSIM_ALL") ? 1 : 0;
    m->qsim_hotkeep = getenv("QSIM_HOT_KEEP") ? atoi(getenv("QSIM_HOT_KEEP")) : m->npin;
    if (m->qsim_bits && !m->qsim_all) m->qsim_thr = calloc(c->n_layers, 4);
    uint32_t *tmp = malloc((size_t)E * 4);
    Slot **ps = malloc((size_t)c->n_layers * m->npin * sizeof(Slot*));
    int *pl = malloc((size_t)c->n_layers * m->npin * sizeof(int));
    int np = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (fread(tmp, 4, E, f) != (size_t)E) break;
        if (!c->sparse[i] || !m->npin) continue;
        memcpy(m->eusage[i], tmp, (size_t)E * 4);          /* seed the ranking */
        if (m->qsim_thr) {   /* per-layer keep threshold = hotkeep-th largest count */
            uint32_t *cp = malloc((size_t)E*4); memcpy(cp, tmp, (size_t)E*4);
            qsort(cp, E, 4, u32_desc);
            int k = m->qsim_hotkeep; if (k < 1) k = 1; if (k > E) k = E;
            m->qsim_thr[i] = cp[k-1];
            free(cp);
        }
        for (int r = 0; r < m->npin; r++) {                /* top-N selection */
            int best = -1; uint32_t bv = 0;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int z = 0; z < r; z++) if (ps[np-r+z]->eid == e) { taken = 1; break; }
                if (!taken && tmp[e] >= bv && tmp[e] > 0) { bv = tmp[e]; best = e; }
            }
            if (best < 0) break;
            Slot *s = slot_acquire(m, i, best, 0);
            s->pinned = 1;
            ps[np] = s; pl[np] = i; np++;
        }
    }
    fclose(f);
    if (np) {
        double t0 = now_s();
        #pragma omp parallel for schedule(dynamic,1)
        for (int j = 0; j < np; j++) slot_fill(m, pl[j], ps[j]);
        fprintf(stderr, "[pin] %d experts pinned (%d/layer) from %s in %.1fs\n",
                np, m->npin, up, now_s()-t0);
#ifdef COLI_CUDA
        /* GPU expert tier: upload the hottest pinned experts (int4 container) into
         * VRAM so their matmul runs on-device — the weight read is the bottleneck
         * and VRAM feeds it ~12x DDR5. Budget = free VRAM after residents, less a
         * 3 GB headroom for activation staging. CUDA_EXPERT_GB caps it. */
        if (g_cuda && m->xq && m->rb13*2 == c->hidden && !(getenv("NOGPU"))) {
            int64_t I = c->moe_inter, D = c->hidden;
            size_t per = (size_t)m->rb13*2*I + (size_t)2*I*4 + (size_t)m->rb2*D + (size_t)D*4;
            size_t budget = ink_cuda_free_bytes();
            budget = budget > (3ULL<<30) ? budget - (3ULL<<30) : 0;
            if (getenv("CUDA_EXPERT_GB")) {
                size_t cap_b = (size_t)(atof(getenv("CUDA_EXPERT_GB"))*1e9);
                if (cap_b < budget) budget = cap_b;
            }
            /* upload globally-hottest experts first, so every layer gets its top
             * experts on-device instead of filling the first few layers */
            int *ord = malloc((size_t)np*sizeof(int));
            uint32_t *pu = malloc((size_t)np*sizeof(uint32_t));
            for (int j = 0; j < np; j++) { ord[j] = j; pu[j] = m->eusage[pl[j]][ps[j]->eid]; }
            g_pin_use = pu; qsort(ord, np, sizeof(int), pin_use_desc);
            size_t used = 0; int nv = 0; double tv = now_s();
            for (int oi = 0; oi < np && used + per <= budget; oi++) {
                Slot *s = ps[ord[oi]];
                s->d13  = ink_cuda_upload(s->p13, (size_t)m->rb13*2*I);
                s->ds13 = ink_cuda_upload(s->s13, (size_t)2*I*4);
                s->d2   = ink_cuda_upload(s->p2,  (size_t)m->rb2*D);
                s->ds2  = ink_cuda_upload(s->s2,  (size_t)D*4);
                if (s->d13 && s->ds13 && s->d2 && s->ds2) { s->gpu = 1; used += per; nv++; }
            }
            free(ord); free(pu);
            fprintf(stderr, "[cuda] %d experts resident in VRAM (%.1f GB) in %.1fs\n",
                    nv, used/1e9, now_s()-tv);
        }
#endif
    }
    free(tmp); free(ps); free(pl);
}

/* usage snapshot: rewritten after every generation run (same contract as
 * glm's .coli_usage — copy it aside if you need a stable ranking).
 * USAGE_SAVE=0 skips the rewrite (e.g. benchmark loops that would skew the
 * ranking); PIN=off also implies no save (that run never seeded counts). */
static int usage_save(Model *m, const char *snap) {
    Cfg *c = &m->c; int E = c->n_experts;
    char up[2048], tp[2060];
    const char *env = getenv("PIN");
    const char *sv = getenv("USAGE_SAVE");
    if (sv && *sv == '0') return 0;
    /* FORCE_EXPERTS routes to a fixed set (diagnostic) — never let that garbage
     * routing overwrite the real usage ranking, or it poisons the pins. */
    if (getenv("FORCE_EXPERTS")) return 0;
    if (env && (!strcmp(env, "off") || !strcmp(env, "0"))) return 0;
    if (env) snprintf(up, sizeof(up), "%s", env);
    else snprintf(up, sizeof(up), "%s/.coli_usage", snap);
    snprintf(tp, sizeof(tp), "%s.tmp", up);
    FILE *f = fopen(tp, "wb");
    if (!f) return 0;
    uint32_t hdr[3] = { 0x31554B49u, (uint32_t)c->n_layers, (uint32_t)E };
    fwrite(hdr, 4, 3, f);
    uint32_t *zero = calloc(E, 4);
    for (int i = 0; i < c->n_layers; i++)
        fwrite(m->eusage[i] ? m->eusage[i] : zero, 4, E, f);
    free(zero); fclose(f);
    return rename(tp, up) == 0;
}

/* ---------- attention (GQA + sliding/global + relative bias + K/V sconv) ---------- */
static void attention(Model *m, Layer *l, int li, float *x, int S, int pos0, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, H = L_HEADS(c,li), KV = L_KV(c,li), hd = L_HD(c,li), ext = L_EXT(c,li);
    int local = c->local[li];
    int qdim = H*hd, kvdim = KV*hd, group = H/KV;
    float *q  = falloc((int64_t)S*qdim);
    float *k  = falloc((int64_t)S*kvdim);
    float *vv = falloc((int64_t)S*kvdim);
    float *rr = falloc((int64_t)S*H*c->d_rel);
    matmul_w(q,  x, l->q, S, D, qdim);
    matmul_w(k,  x, l->k, S, D, kvdim);
    matmul_w(vv, x, l->v, S, D, kvdim);
    matmul_w(rr, x, l->r, S, D, H*c->d_rel);
    /* short convs on K and V (sequence-wise, over the raw projections) */
    sconv_apply(k,  S, kvdim, l->k_cw, m->cs[0][li], c->conv_k, m->vck_on ? m->vck[0][li] : NULL);
    sconv_apply(vv, S, kvdim, l->v_cw, m->cs[1][li], c->conv_k, m->vck_on ? m->vck[1][li] : NULL);
    /* per-head q/k rmsnorm (scaling below is 1/hd, not 1/sqrt(hd), because of this) */
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H;  h++) rmsnorm_row(q + (int64_t)s*qdim  + h*hd, q + (int64_t)s*qdim  + h*hd, l->qn, hd, c->eps);
        for (int h = 0; h < KV; h++) rmsnorm_row(k + (int64_t)s*kvdim + h*hd, k + (int64_t)s*kvdim + h*hd, l->kn, hd, c->eps);
    }
    /* append K,V to the cache */
    for (int s = 0; s < S; s++) for (int h = 0; h < KV; h++) {
        int t = pos0 + s;
        memcpy(m->K[li] + ((int64_t)h*m->max_t + t)*hd, k  + (int64_t)s*kvdim + h*hd, hd*sizeof(float));
        memcpy(m->V[li] + ((int64_t)h*m->max_t + t)*hd, vv + (int64_t)s*kvdim + h*hd, hd*sizeof(float));
    }
    float scale = 1.f / (float)hd;
    float *ctx = falloc((int64_t)S*qdim);
    #pragma omp parallel
    {
        float *rl = malloc(ext * sizeof(float));
        float *sc = malloc((size_t)m->max_t * sizeof(float));
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < S; s++) {
                int qpos = pos0 + s;
                int t0 = local && qpos - c->window + 1 > 0 ? qpos - c->window + 1 : 0;
                /* mix the relative-bias bank for this (token, head): rl[dist] */
                const float *rv = rr + (int64_t)s*H*c->d_rel + h*c->d_rel;
                for (int e = 0; e < ext; e++) {
                    float acc = 0.f;
                    for (int d = 0; d < c->d_rel; d++) acc += rv[d] * l->relp[(int64_t)d*ext + e];
                    rl[e] = acc;
                }
                /* tau: log-length scaling on global layers (f32, per query pos) */
                float tau = 1.f;
                if (!local && c->log_floor > 0) {
                    double en = (double)(qpos + 1) / c->log_floor;
                    if (en > 1.0) tau = 1.f + c->log_alpha * (float)log(en);
                }
                const float *qv = q + (int64_t)s*qdim + h*hd;
                const float *Kh = m->K[li] + ((int64_t)(h/group)*m->max_t)*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *kv = Kh + (int64_t)t*hd;
                    float acc = 0.f;
                    for (int d = 0; d < hd; d++) acc += qv[d]*kv[d];
                    int dist = qpos - t;
                    sc[t - t0] = tau * (acc*scale + (dist < ext ? rl[dist] : 0.f));
                }
                int n = qpos - t0 + 1;
                softmax_row(sc, n);
                float *cx = ctx + (int64_t)s*qdim + h*hd;
                for (int d = 0; d < hd; d++) cx[d] = 0.f;
                const float *Vh = m->V[li] + ((int64_t)(h/group)*m->max_t)*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *vrow = Vh + (int64_t)t*hd;
                    float a = sc[t - t0];
                    for (int d = 0; d < hd; d++) cx[d] += a * vrow[d];
                }
            }
        }
        free(rl); free(sc);
    }
    matmul_w(out, ctx, l->o, S, qdim, D);
    free(q); free(k); free(vv); free(rr); free(ctx);
}

/* ---------- dense MLP ---------- */
static void dense_mlp_i(Model *m, Layer *l, float *x, int S, float *out, int I) {
    Cfg *c = &m->c; int D = c->hidden;
    float *g = falloc((int64_t)S*I), *u = falloc((int64_t)S*I);
    matmul_w(g, x, l->dg, S, D, I);
    matmul_w(u, x, l->du, S, D, I);
    for (int64_t i = 0; i < (int64_t)S*I; i++) g[i] = siluf(g[i]) * u[i];
    matmul_w(out, g, l->dd, S, I, D);
    for (int64_t i = 0; i < (int64_t)S*D; i++) out[i] *= l->dgs;
    free(g); free(u);
}
static void dense_mlp(Model *m, Layer *l, float *x, int S, float *out) {
    dense_mlp_i(m, l, x, S, out, m->c.dense_inter);
}

/* one routed expert's forward (gate+up -> silu -> down) into dst[D]. g is the
 * caller's 2I scratch (u = g+I); after return g holds the post-silu gate — the
 * LoRA down_acc in the caller reads it. Factored out so the reordered decode
 * path and the in-order path run the IDENTICAL FP op sequence per expert. */
static void expert_compute(Model *m, Slot *e, int layer, const float *xs,
                           float *g, float *dst, int q4, int fuse_ok,
                           int lact, const float *lt1, const float *lt3) {
    Cfg *c = &m->c; int D = c->hidden, I = c->moe_inter;
    float *u = g + I;
    (void)fuse_ok;   /* only read by the CUDA fused path */
#ifdef COLI_CUDA
    /* fused GPU expert: gate+up→silu→down in one launch/sync, no host bounce.
     * Decode only and no LoRA (LoRA needs the host g mid-expert). */
    if (fuse_ok && q4 && e->gpu &&
        ink_cuda_expert_q4(dst, xs, e->d13, e->ds13, e->d2, e->ds2, I, D) == 0) return;
#endif
    /* gate+up (fused rows: gate then up) */
    if (m->xq) {
#ifdef COLI_CUDA
        if (q4 && e->gpu) ink_cuda_matmul_q4(g, xs, e->d13, e->ds13, 1, D, 2*I); else
#endif
        if (q4) matmul_q4(g, xs, e->p13, e->s13, D, 2*I);
        else    matmul_q(g, xs, (int8_t*)e->p13, e->s13, D, 2*I);
    } else if (m->quant_bits) matmul_q(g, xs, e->q13, e->s13, D, 2*I);
    else                      matmul(g, xs, e->f13, 1, D, 2*I);
    if (lact) lora_moe_gu(&m->lora, layer, e->eid, lt1, lt3, g);
    for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
    /* down */
    if (m->xq) {
#ifdef COLI_CUDA
        if (q4 && e->gpu) ink_cuda_matmul_q4(dst, g, e->d2, e->ds2, 1, I, D); else
#endif
        if (q4) matmul_q4(dst, g, e->p2, e->s2, I, D);
        else    matmul_q(dst, g, (int8_t*)e->p2, e->s2, I, D);
    } else if (m->quant_bits) matmul_q(dst, g, e->q2, e->s2, I, D);
    else                      matmul(dst, g, e->f2, 1, I, D);
}

/* ---------- MoE: sigmoid router + bias top-k, joint routed+shared weights ----------
 * Three passes per layer call: (1) route every position and acquire slots,
 * (2) fill ALL missing experts in one parallel burst (the NVMe wants queue
 * depth — during prefill this batches the whole sequence's misses), then
 * (3) compute. */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk, I = c->moe_inter, ns = c->n_shared;
    int ET = E + ns;
    float *logits = falloc((int64_t)S*ET);
    matmul(logits, x, l->router, S, D, ET);
    memset(out, 0, (int64_t)S*D*sizeof(float));
    int   *idx  = malloc((size_t)S*K*sizeof(int));
    float *wgt  = malloc((size_t)S*(K+ns)*sizeof(float));
    Slot **use  = malloc((size_t)S*K*sizeof(Slot*));
    Slot **fill = malloc((size_t)S*K*sizeof(Slot*));
    int  *fl    = malloc((size_t)S*K*sizeof(int));
    int nfill = 0;
    /* pass 1: routing + slot bookkeeping (serial) */
    for (int s = 0; s < S; s++) {
        float *lg = logits + (int64_t)s*ET;
        int *si = idx + (int64_t)s*K;
        /* selection: sigmoid(routed) + correction bias, top-K */
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (si[j]==e){taken=1;break;}
                float ch = sigmoidf(lg[e]) + l->rbias[e];
                if (!taken && ch > bv) { bv = ch; best = e; }
            }
            si[kk] = best;
        }
        /* FORCE_EXPERTS=1: pin routing to a fixed expert set (0..K-1) so every
         * token hits the same cached experts — zero misses, isolates the compute
         * ceiling from disk. Output is garbage (wrong experts); timing is real. */
        static int fce = -1;
        if (fce < 0) fce = getenv("FORCE_EXPERTS") ? 1 : 0;
        if (fce) for (int kk = 0; kk < K; kk++) si[kk] = kk % E;
        /* combine weights: sigmoids of the raw logits of (topK routed + shared),
         * normalized to sum 1 over all K+ns, x route_scale x gate.global_scale */
        float *w = wgt + (int64_t)s*(K+ns); float sum = 0.f;
        for (int kk = 0; kk < K; kk++)  { w[kk]   = sigmoidf(lg[si[kk]]); sum += w[kk]; }
        for (int j = 0; j < ns; j++)    { w[K+j]  = sigmoidf(lg[E+j]);    sum += w[K+j]; }
        for (int kk = 0; kk < K+ns; kk++) w[kk] *= c->route_scale * l->rgs / sum;
        for (int kk = 0; kk < K; kk++) {
            int eid = si[kk];
            if (m->eusage[layer]) m->eusage[layer][eid]++;
            Slot *e = slot_find(m, layer, eid);
            if (e) m->hits++;
            else {
                m->miss++;
                if (m->seen[layer]) {   /* cold first-touch vs evicted-and-back */
                    if (m->seen[layer][eid]) m->miss_churn++;
                    else {
                        m->miss_cold++; m->seen[layer][eid] = 1;
                        if (S == 1) {   /* was this cold expert a recent near-miss? */
                            int in = 0;
                            for (int j = 0; j < m->topm; j++) if (m->prev_topm[layer][j] == eid) { in = 1; break; }
                            if (in) m->cold_recover++; else m->cold_novel++;
                        }
                    }
                }
                e = slot_acquire(m, layer, eid, 1);
                fill[nfill] = e; fl[nfill] = layer; nfill++;
            }
            e->busy = 1;
            use[(int64_t)s*K + kk] = e;
        }
        /* record this decode token's top-M router ranking for next token's probe */
        if (S == 1 && m->prev_topm[layer]) {
            int *tm = m->prev_topm[layer];
            for (int r = 0; r < m->topm; r++) {
                int best = -1; float bv = -1e30f;
                for (int e = 0; e < E; e++) {
                    int taken = 0; for (int j = 0; j < r; j++) if (tm[j]==e){taken=1;break;}
                    float ch = sigmoidf(lg[e]) + l->rbias[e];
                    if (!taken && ch > bv) { bv = ch; best = e; }
                }
                tm[r] = best;
            }
        }
    }
    /* pass 2: kick every miss in this layer call. Engine on: chunked async
     * reads — the NVMe sees a FILL_THREADS-deep queue even for a single decode
     * miss, and the waits move into pass 3 behind the cached experts' compute.
     * Engine off (FILL_ASYNC=0): the original synchronous parallel burst. */
    int async = fe_on(m);
    if (nfill) {
        double tf = now_s();
        if (async) for (int j = 0; j < nfill; j++) fill_submit(m, fl[j], fill[j]);
        else {
            #pragma omp parallel for schedule(dynamic,1)
            for (int j = 0; j < nfill; j++) slot_fill(m, fl[j], fill[j]);
        }
        m->t_fill += now_s() - tf;
    }
    /* OVERFETCH=n: also start fills for the router's runners-up (this token's
     * ranking beyond top-K) — the "recent near-miss" class the overfetch probe
     * measures. They stream in behind later layers' compute and turn the next
     * token's cold misses into hits or partial waits. Decode only. */
    if (async && S == 1 && g_overfetch > 0) overfetch_submit(m, layer);
    /* pass 3: compute (per-expert math lives in expert_compute). */
    float *g = falloc(2*I), *u = g + I, *hh = falloc(D);
    int q4 = m->xq && m->rb13*2 == D;   /* packed int4 vs int8 container */
    /* LoRA runtime path: shared-A projections t1/t3 computed once per token;
     * the per-expert down deltas accumulate as weighted rank-r vectors in
     * acc2 so the shared B2 applies once per token, not once per expert */
    Lora *lr = &m->lora;
    int lact = lr->experts_on && lr->b1[layer];
    float *lt = lact ? falloc(3*lr->r) : NULL;
    float *lt1 = lt, *lt3 = lact ? lt + lr->r : NULL, *acc2 = lact ? lt + 2*lr->r : NULL;
    /* async decode (no LoRA): compute each expert into its own hb row in the
     * order the fills LAND — cached experts first, so their matmuls hide the
     * misses' remaining I/O — then accumulate into os in kk order. Per-expert
     * math and the accumulation sequence are identical to the in-order path,
     * so the reorder cannot change the emitted tokens. */
    int reorder = async && S == 1 && !lact && K <= 64;
    float *hb = reorder ? falloc((int64_t)K*D) : NULL;
    if (async && !reorder) {
        /* prefill / LoRA / MTP batches: settle every slot up front (a fill_wait
         * on a filled slot is a load and a branch), then compute as before */
        double tw = now_s();
        for (int64_t t = 0; t < (int64_t)S*K; t++) fill_wait(use[t]);
        m->t_fill += now_s() - tw;
    }
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s*D;
        float *os = out + (int64_t)s*D;
        float *w = wgt + (int64_t)s*(K+ns);
        double te = now_s();
        if (lact) { lora_moe_pre(lr, layer, xs, lt1, lt3); memset(acc2, 0, lr->r*sizeof(float)); }
        if (reorder) {
            unsigned char dn[64] = {0};
            int left = K;
            for (int pass = 0; pass < 2 && left; pass++)
                for (int kk = 0; kk < K && left; kk++) {
                    Slot *e = use[kk];
                    if (dn[kk]) continue;
                    if (!pass && !__atomic_load_n(&e->filled, __ATOMIC_ACQUIRE)) continue;
                    if (pass) {   /* exposed stall: I/O the hits' compute couldn't hide */
                        double tw = now_s(); fill_wait(e);
                        double t1 = now_s(); m->t_fill += t1 - tw; te += t1 - tw;
                    }
                    expert_compute(m, e, layer, xs, g, hb + (int64_t)kk*D, q4, 1, 0, NULL, NULL);
                    dn[kk] = 1; left--;
                }
            for (int kk = 0; kk < K; kk++) {
                const float *hk = hb + (int64_t)kk*D;
                for (int d = 0; d < D; d++) os[d] += w[kk] * hk[d];
            }
        } else for (int kk = 0; kk < K; kk++) {
            Slot *e = use[(int64_t)s*K + kk];
            expert_compute(m, e, layer, xs, g, hh, q4, S == 1 && !lact, lact, lt1, lt3);
            if (lact) lora_moe_down_acc(lr, layer, e->eid, g, w[kk], acc2);
            for (int d = 0; d < D; d++) os[d] += w[kk] * hh[d];
        }
        if (lact) lora_moe_down_apply(lr, layer, acc2, os);
        double ts = now_s(); m->t_expert += ts - te;
        /* shared experts: gamma inside (before down_proj is linear, so applied at the end) */
        for (int j = 0; j < ns; j++) {
            matmul_w(g, xs, wt_off(l->sh_g, (int64_t)j*I*D), 1, D, I);
            matmul_w(u, xs, wt_off(l->sh_u, (int64_t)j*I*D), 1, D, I);
            for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
            matmul_w(hh, g, wt_off(l->sh_d, (int64_t)j*D*I), 1, I, D);
            for (int d = 0; d < D; d++) os[d] += w[K+j] * hh[d];
        }
        m->t_shared += now_s() - ts;
    }
    for (int64_t t = 0; t < (int64_t)S*K; t++) {
        Slot *e = use[t];
        e->busy = 0;
        if (e->transient) {   /* cap-overflow spill: lives for this call only */
            free(e->p13); free(e->p2); free(e->q13); free(e->q2);
            free(e->s13); free(e->s2); free(e->f13); free(e->f2); free(e);
        }
    }
    free(logits); free(idx); free(wgt); free(use); free(fill); free(fl);
    free(g); free(hh); free(lt); free(hb);    /* u aliases g+I */
}

/* ---------- one forward pass over S new tokens ----------
 * Returns malloc'd logits of the last token (unpadded vocab). If tf_out is
 * non-NULL also writes the per-position argmax (teacher-forcing check). */
static float *step(Model *m, const int *ids, int S, int pos0, int *tf_out, double *nll_sum, float *hid_out) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) {
        wt_row_f32(m->embed, (int64_t)ids[s]*D, x + (int64_t)s*D, D);
        if (m->embed_norm) rmsnorm_row(x + (int64_t)s*D, x + (int64_t)s*D, m->embed_norm, D, c->eps);
    }
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        double ta = now_s();
        attention(m, l, i, nrm, S, pos0, tmp);
        m->t_attn += now_s() - ta;
        sconv_apply(tmp, S, D, l->a_cw, m->cs[2][i], c->conv_k, m->vck_on ? m->vck[2][i] : NULL);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        if (c->sparse[i]) moe(m, l, i, nrm, S, tmp);
        else dense_mlp(m, l, nrm, S, tmp);
        sconv_apply(tmp, S, D, l->m_cw, m->cs[3][i], c->conv_k, m->vck_on ? m->vck[3][i] : NULL);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->kv_len = pos0 + S;
    /* MTP module 0 consumes the POST-final-norm hidden (vLLM: the base forward
     * returns self.norm(hidden), and that same tensor is the drafter's input). */
    if (hid_out) for (int s = 0; s < S; s++)
        rmsnorm_row(hid_out + (int64_t)s*D, x + (int64_t)s*D, m->final_norm, D, c->eps);
    float *last = falloc(D);
    float *logit = falloc(c->unpad_vocab);
    if (tf_out || nll_sum) {
        for (int s = 0; s < S; s++) {
            rmsnorm_row(last, x + (int64_t)s*D, m->final_norm, D, c->eps);
            for (int d = 0; d < D; d++) last[d] /= c->mup;
            matmul_w(logit, last, m->lm_head, 1, D, c->unpad_vocab);
            if (tf_out) {
                int best = 0; for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > logit[best]) best = i;
                tf_out[pos0 + s] = best;
            }
            /* teacher-forced NLL of the true next token: -log softmax(logit)[next] */
            if (nll_sum && s < S-1) {
                int nx = ids[s+1];
                if (nx >= 0 && nx < c->unpad_vocab) {
                    float mx = logit[0]; for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > mx) mx = logit[i];
                    double se = 0; for (int i = 0; i < c->unpad_vocab; i++) se += exp((double)(logit[i]-mx));
                    *nll_sum += (mx + log(se)) - logit[nx];
                }
            }
        }
    }
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    for (int d = 0; d < D; d++) last[d] /= c->mup;
    matmul_w(logit, last, m->lm_head, 1, D, c->unpad_vocab);
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

static void state_reset(Model *m) {
    Cfg *c = &m->c;
    m->kv_len = 0;
    for (int i = 0; i < c->n_layers + m->n_mtp; i++) {   /* MTP slots at n_layers+k too */
        int kvdim = L_KV(c,i) * L_HD(c,i);
        for (int j = 0; j < 4; j++)
            memset(m->cs[j][i], 0, (int64_t)((j < 2) ? kvdim : c->hidden) * (c->conv_k-1) * sizeof(float));
    }
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    int nl = c->n_layers + m->n_mtp;             /* MTP modules keep KV at n_layers+k */
    if (m->K && max_t <= m->max_t) return;   /* reuse across prompts when big enough */
    if (m->K) for (int i = 0; i < nl; i++) { free(m->K[i]); free(m->V[i]); }
    free(m->K); free(m->V);
    m->max_t = max_t;
    m->K = calloc(nl, sizeof(float*)); m->V = calloc(nl, sizeof(float*));
    for (int i = 0; i < nl; i++) {
        m->K[i] = falloc((int64_t)L_KV(c,i) * max_t * L_HD(c,i));
        m->V[i] = falloc((int64_t)L_KV(c,i) * max_t * L_HD(c,i));
    }
}

/* greedy generation, olmoe.c-style */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0, NULL, NULL, NULL);
    int len = np;
    Cfg *c = &m->c;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1, NULL, NULL, NULL);
    }
}

/* ---------- MTP draft head ----------
 * One MTP module = a dense Inkling block (same op sequence as step()'s layer
 * loop) over the module's KV/conv slot li = n_layers+k. */
static void mtp_block(Model *m, MtpMod *mm, int li, float *xin, int S, int pos0, float *xout) {
    Cfg *c = &m->c; int D = c->hidden; Layer *l = &mm->L;
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    memcpy(xout, xin, (int64_t)S*D*sizeof(float));
    for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, xout + (int64_t)s*D, l->in_ln, D, c->eps);
    attention(m, l, li, nrm, S, pos0, tmp);
    sconv_apply(tmp, S, D, l->a_cw, m->cs[2][li], c->conv_k, NULL);
    for (int64_t j = 0; j < (int64_t)S*D; j++) xout[j] += tmp[j];
    for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, xout + (int64_t)s*D, l->post_ln, D, c->eps);
    dense_mlp_i(m, l, nrm, S, tmp, mm->inter);
    sconv_apply(tmp, S, D, l->m_cw, m->cs[3][li], c->conv_k, NULL);
    for (int64_t j = 0; j < (int64_t)S*D; j++) xout[j] += tmp[j];
    free(nrm); free(tmp);
}

/* fuse [hidden_norm(h) ; embed_norm(emb(tok))] -> in_proj -> block input row */
static void mtp_fuse(Model *m, MtpMod *mm, const float *h, int tok, float *xrow) {
    Cfg *c = &m->c; int D = c->hidden;
    float *e = malloc((size_t)D*sizeof(float)), *cat = malloc((size_t)2*D*sizeof(float));
    wt_row_f32(m->embed, (int64_t)tok*D, e, D);
    /* the depth layers were trained on the BACKBONE-normed embedding: apply the
     * main model's (whitening) embed_norm, then the module's own (near-identity)
     * embed_norm — matching vLLM's fused_input_cat. */
    if (m->embed_norm) rmsnorm_row(e, e, m->embed_norm, D, c->eps);
    rmsnorm_row(e, e, mm->embed_norm, D, c->eps);
    rmsnorm_row(cat, h, mm->hidden_norm, D, c->eps);        /* hidden first */
    memcpy(cat + D, e, (size_t)D*sizeof(float));
    matmul_w(xrow, cat, mm->in_proj, 1, 2*D, D);
    free(e); free(cat);
}

/* MTP head logits at each of P positions: the depth output goes STRAIGHT to the
 * shared lm_head (no final_norm — the depth was trained that way), mup-scaled
 * (argmax-invariant, kept for scale parity). vLLM InklingMTP.compute_logits. */
static void mtp_head(Model *m, const float *hp, int P, int *pred) {
    Cfg *c = &m->c; int D = c->hidden;
    float *last = falloc(D), *logit = falloc(c->unpad_vocab);
    for (int i = 0; i < P; i++) {
        for (int d = 0; d < D; d++) last[d] = hp[(int64_t)i*D + d] / c->mup;
        matmul_w(logit, last, m->lm_head, 1, D, c->unpad_vocab);
        int best = 0; for (int v = 1; v < c->unpad_vocab; v++) if (logit[v] > logit[best]) best = v;
        pred[i] = best;
    }
    free(last); free(logit);
}

/* module k teacher-forced: input h^k rows hin[P*D] + tokens toks[P] (t_{i+k+1});
 * writes h^{k+1} to hout[P*D] and per-position argmax to pred[P] (if non-NULL).
 * Resets the module's own conv/KV slot first (fresh teacher-forced pass). */
static void mtp_module_tf(Model *m, int k, const float *hin, const int *toks, int P, float *hout, int *pred) {
    Cfg *c = &m->c; int D = c->hidden;
    MtpMod *mm = &m->mtp[k]; int li = c->n_layers + k;
    float *xin = falloc((int64_t)P*D);
    for (int i = 0; i < P; i++) mtp_fuse(m, mm, hin + (int64_t)i*D, toks[i], xin + (int64_t)i*D);
    int kvdim = L_KV(c,li)*L_HD(c,li);
    for (int j = 0; j < 4; j++) memset(m->cs[j][li], 0, (int64_t)((j<2)?kvdim:D)*(c->conv_k-1)*sizeof(float));
    mtp_block(m, mm, li, xin, P, 0, hout);
    if (pred) mtp_head(m, hout, P, pred);
    free(xin);
}

/* ---------- speculative decode (Phase 1: n-gram draft + batched verify) ----------
 * The verify forward is just step() over [committed, draft...] with tf_out (the
 * per-position argmax = the acceptance signal). Inkling's four causal short-convs
 * are stateful, so a partially accepted batch must roll their state back to the
 * last accepted position — that is what the vck checkpoints (written in
 * sconv_apply while vck_on) and conv_rollback() are for. KV is position-indexed,
 * so rejected rows are simply overwritten by the next forward. */

static void vck_alloc(Model *m, int maxS) {
    Cfg *c = &m->c;
    if (m->vck[0] && maxS <= m->vck_S) return;
    int P = c->conv_k - 1;
    for (int j = 0; j < 4; j++) {
        if (m->vck[j]) { for (int i = 0; i < c->n_layers; i++) free(m->vck[j][i]); free(m->vck[j]); }
        m->vck[j] = calloc(c->n_layers, sizeof(float*));
    }
    for (int i = 0; i < c->n_layers; i++) {
        int64_t kvdim = (int64_t)L_KV(c,i) * L_HD(c,i);
        m->vck[0][i] = falloc((int64_t)maxS * kvdim * P);
        m->vck[1][i] = falloc((int64_t)maxS * kvdim * P);
        m->vck[2][i] = falloc((int64_t)maxS * c->hidden * P);
        m->vck[3][i] = falloc((int64_t)maxS * c->hidden * P);
    }
    m->vck_S = maxS;
}

/* roll every conv's live state back to the state AFTER batch position k
 * (0-based within the just-verified batch). */
static void conv_rollback(Model *m, int k) {
    Cfg *c = &m->c;
    int64_t P = c->conv_k - 1;
    for (int i = 0; i < c->n_layers; i++) {
        int64_t kvdim = (int64_t)L_KV(c,i) * L_HD(c,i);
        memcpy(m->cs[0][i], m->vck[0][i] + (int64_t)k*kvdim*P, kvdim*P*sizeof(float));
        memcpy(m->cs[1][i], m->vck[1][i] + (int64_t)k*kvdim*P, kvdim*P*sizeof(float));
        memcpy(m->cs[2][i], m->vck[2][i] + (int64_t)k*c->hidden*P, (int64_t)c->hidden*P*sizeof(float));
        memcpy(m->cs[3][i], m->vck[3][i] + (int64_t)k*c->hidden*P, (int64_t)c->hidden*P*sizeof(float));
    }
}

/* n-gram drafter: the most recent prior occurrence of the last committed token
 * predicts the tokens that followed it. Zero model cost; cheap correctness
 * exercise for the verify/rollback machinery before the MTP head lands. */
static int ngram_draft(const int *hist, int n, int G, int *draft) {
    if (n < 2) return 0;
    int last = hist[n-1];
    for (int p = n - 2; p >= 0; p--) {
        if (hist[p] == last) {
            int d = 0;
            for (int g = 0; g < G && p + 1 + g < n; g++) draft[d++] = hist[p + 1 + g];
            return d;
        }
    }
    return 0;
}

/* greedy speculative decode. Output is token-identical to generate(); returns
 * tokens-per-verify-forward (>1 = speedup). */
static double generate_spec(Model *m, const int *prompt, int np, int n_new, int *out, int G) {
    Cfg *c = &m->c;
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    vck_alloc(m, 1 + G);
    int *tf = calloc((int64_t)(np + n_new + G + 2), sizeof(int));
    int *batch = malloc((size_t)(1 + G) * sizeof(int));
    int *draft = malloc((size_t)G * sizeof(int));

    float *logit = step(m, prompt, np, 0, NULL, NULL, NULL);
    int best = 0; for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > logit[best]) best = i;
    free(logit);

    int kv = np, gen = 0; long fwd = 0;
    while (gen < n_new) {
        out[kv] = best; gen++;
        if (gen >= n_new) break;
        int Gc = G; if (Gc > n_new - gen) Gc = n_new - gen;
        int g = ngram_draft(out, kv + 1, Gc, draft);
        batch[0] = best; for (int i = 0; i < g; i++) batch[1 + i] = draft[i];
        int S = 1 + g;
        m->vck_on = 1;
        step(m, batch, S, kv, tf, NULL, NULL);   /* verify: fills tf[kv..kv+g], conv ckpts */
        m->vck_on = 0; fwd++;
        int a = 0;
        while (a < g && tf[kv + a] == draft[a]) { out[kv + 1 + a] = draft[a]; a++; }
        gen += a;
        best = tf[kv + a];                    /* next committed = pred after last accepted */
        conv_rollback(m, a);
        m->kv_len = kv + 1 + a;
        kv += 1 + a;
    }
    free(tf); free(batch); free(draft);
    return fwd ? (double)n_new / (double)fwd : 0.0;
}

/* greedy speculative decode with the MTP depth-0 drafter (1 draft/round). Output
 * is token-identical to generate(). Module 0 only ever processes CONFIRMED
 * positions (draft input = [hid_confirmed ; emb(committed)]), so its KV/conv grow
 * monotonically and never need rollback — the draft prediction is a pure read. */
static double generate_spec_mtp(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c; int D = c->hidden;
    if (m->n_mtp < 1) return 0;
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    int cap = np + n_new + 4;
    MtpMod *mm = &m->mtp[0]; int li = c->n_layers;
    float *hid = falloc((int64_t)cap * D);          /* confirmed main post-norm hiddens */
    int *tf = calloc((size_t)cap, sizeof(int));
    float *xin = falloc((int64_t)cap * D), *xout = falloc((int64_t)cap * D);
    int *pr = malloc((size_t)cap * sizeof(int));

    float *logit = step(m, prompt, np, 0, NULL, NULL, hid);     /* hid[0..np-1] */
    int best = 0; for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > logit[best]) best = i;
    free(logit);

    int kv = np, gen = 0, mtp_abs = 0; long fwd = 0;
    while (gen < n_new) {
        out[kv] = best; gen++;
        if (gen >= n_new) break;
        int p = kv - 1;                             /* module-0 draft position */
        int s0 = mtp_abs, S = p - s0 + 1;           /* absorb the gap [mtp_abs..p], all confirmed */
        for (int i = 0; i < S; i++) mtp_fuse(m, mm, hid + (int64_t)(s0+i)*D, out[s0+i+1], xin + (int64_t)i*D);
        mtp_block(m, mm, li, xin, S, s0, xout);
        mtp_head(m, xout, S, pr);
        int draft0 = pr[S-1];                        /* candidate for position kv+1 */
        mtp_abs = p + 1;
        int batch[2] = { best, draft0 };
        step(m, batch, 2, kv, tf, NULL, hid + (int64_t)kv*D);    /* hid[kv],hid[kv+1]; tf[kv],tf[kv+1] */
        fwd++;
        if (tf[kv] == draft0) { out[kv+1] = draft0; gen++; best = tf[kv+1]; kv += 2; }
        else                  { best = tf[kv]; kv += 1; }
    }
    free(hid); free(tf); free(xin); free(xout); free(pr);
    return fwd ? (double)n_new / (double)fwd : 0.0;
}

static const char *prompt_reject(Cfg *c, const int *ids, int np, int want);

/* ---------- interactive prompt mode: greedy, streaming, stop on eos ---------- */
static void generate_stream(Model *m, Tok *T, const char *prompt, int n_new) {
    Cfg *c = &m->c;
    int cap = (int)strlen(prompt) + 16;
    int *ids = malloc(cap * sizeof(int));
    int np = tok_encode(T, prompt, (int)strlen(prompt), ids, cap);
    if (np <= 0) { fprintf(stderr, "empty prompt after tokenization\n"); return; }
    const char *bad = prompt_reject(c, ids, np, n_new);
    if (bad) { fprintf(stderr, "rejected: %s\n", bad); free(ids); return; }
    kv_alloc(m, np + n_new + 8);
    printf("[%d prompt tokens] %s", np, prompt); fflush(stdout);
    double t0 = now_s(), t1 = 0;
    /* phase timers accumulate globally (across prompts in -f mode); snapshot to
     * report THIS prompt's deltas, or 'other' goes negative on later prompts */
    double f0 = m->t_fill, e0 = m->t_expert, s0 = m->t_shared, a0 = m->t_attn;
    /* fresh miss-classification window for this prompt */
    uint64_t mc0 = m->miss_cold, mh0 = m->miss_churn, cr0 = m->cold_recover, cn0 = m->cold_novel;
    for (int i = 0; i < c->n_layers; i++) {
        if (m->seen[i]) memset(m->seen[i], 0, c->n_experts);
        if (m->prev_topm[i]) for (int j = 0; j < m->topm; j++) m->prev_topm[i][j] = -1;
    }
    float *logit = step(m, ids, np, 0, NULL, NULL, NULL);
    int len = np;
    char buf[512];
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->unpad_vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        if (s == 0) t1 = now_s();
        if (best == c->eos) { printf("\n[eos after %d tokens]", s); break; }
        int nb = tok_decode(T, &best, 1, buf, sizeof(buf)-1);
        buf[nb] = 0; fputs(buf, stdout); fflush(stdout);
        int one = best;
        len++;
        if (s == n_new - 1) break;
        logit = step(m, &one, 1, len - 1, NULL, NULL, NULL);
    }
    double dt = now_s() - t1;
    int gen = len - np;
    printf("\n[prefill %.1fs | %d tokens in %.1fs = %.2f tok/s | RSS %.1f GB]\n",
           t1 - t0, gen, dt, gen > 1 ? (gen-1)/dt : 0.0, rss_gb());
    double wall = now_s() - t0;
    double pf = m->t_fill-f0, pe = m->t_expert-e0, ps = m->t_shared-s0, pa = m->t_attn-a0;
    printf("[phases] fill %.1fs | expert-mm %.1fs | shared %.1fs | attn %.1fs | other %.1fs\n",
           pf, pe, ps, pa, wall - pf - pe - ps - pa);
    uint64_t mc = m->miss_cold - mc0, mh = m->miss_churn - mh0, mt = mc + mh;
    printf("[misses] %llu total | cold-first-touch %llu (%.0f%%) | churn/re-evict %llu (%.0f%%)\n",
           (unsigned long long)mt, (unsigned long long)mc, mt?100.0*mc/mt:0.0,
           (unsigned long long)mh, mt?100.0*mh/mt:0.0);
    uint64_t cr = m->cold_recover - cr0, cn = m->cold_novel - cn0, ct = cr + cn;
    printf("[overfetch probe M=%d] decode cold-touches %llu | recent near-miss %llu (%.0f%%, catchable) | truly-novel %llu (%.0f%%)\n",
           m->topm, (unsigned long long)ct, (unsigned long long)cr, ct?100.0*cr/ct:0.0,
           (unsigned long long)cn, ct?100.0*cn/ct:0.0);
    free(ids);
}

/* ---------- serve mode: openai_server.py engine protocol ----------
 * stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n
 *         CANCEL <id>\n
 * stdout: READY sentinel once loaded, then per request a stream of
 *         DATA <id> <size>\n<bytes>\n frames and a final
 *         DONE <id> <tok> <tps> <hit%> <rss> <prompt_tok> <len_limited>\n
 * v1 semantics: requests run one at a time (SUBMITs arriving mid-generation
 * queue up); the KV slot argument is accepted but every request re-prefills. */

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double rng_next(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

/* temperature + top-p nucleus sampling; temp<=0 = greedy */
typedef struct { float p; int i; } PI;
static int pi_desc(const void *a, const void *b) {
    float d = ((const PI*)b)->p - ((const PI*)a)->p;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}
/* reject a prompt that can't be served correctly: multimodal placeholder
 * tokens (text-only engine) or a context that would overrun the KV bound.
 * Returns NULL if ok, else a short reason. */
static const char *prompt_reject(Cfg *c, const int *ids, int np, int want) {
    for (int i = 0; i < np; i++)
        if (ids[i] == c->img_tok || ids[i] == c->aud_tok)
            return "multimodal input not supported (text only)";
    if (np + want > c->ctx_max) return "context exceeds CTX_MAX";
    return NULL;
}

/* CTC-style repetition penalty applied in place: recently emitted tokens get
 * their logit divided (if >0) or multiplied (if <0) by pen>1. pen<=1 = off. */
static void apply_rep_penalty(float *logit, int n, const int *hist, int nhist, float pen) {
    if (pen <= 1.f) return;
    for (int i = 0; i < nhist; i++) {
        int t = hist[i];
        if (t < 0 || t >= n) continue;
        logit[t] = logit[t] > 0 ? logit[t] / pen : logit[t] * pen;
    }
}

static int sample_logits(const float *logit, int n, float temp, float top_p) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logit[i] > logit[best]) best = i;
    if (temp <= 0.f) return best;
    PI *c = malloc((size_t)n * sizeof(PI));
    double sum = 0;
    for (int i = 0; i < n; i++) {
        c[i].p = expf((logit[i] - logit[best]) / temp);
        c[i].i = i; sum += c[i].p;
    }
    qsort(c, n, sizeof(PI), pi_desc);
    double cut = (top_p > 0.f && top_p < 1.f) ? top_p * sum : sum;
    double acc = 0; int k = 0;
    while (k < n && acc < cut) acc += c[k++].p;
    double r = rng_next() * acc, run = 0;
    int pick = c[0].i;
    for (int i = 0; i < k; i++) { run += c[i].p; if (run >= r) { pick = c[i].i; break; } }
    free(c);
    return pick;
}

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } SReq;
#define SRV_QMAX 16
static SReq g_q[SRV_QMAX]; static int g_qn = 0;

static int stdin_readable(void) {
    fd_set r; struct timeval tv = {0, 0};
    FD_ZERO(&r); FD_SET(0, &r);
    return select(1, &r, NULL, NULL, &tv) > 0;
}

/* read one control line (+ payload for SUBMIT). cur_id: request in flight;
 * returns 1 if that request was cancelled, 0 otherwise, -1 on stdin EOF. */
static int serve_read_cmd(const char *cur_id) {
    char ln[512];
    if (!fgets(ln, sizeof(ln), stdin)) return -1;
    char cmd[16], id[64];
    if (sscanf(ln, "%15s %63s", cmd, id) < 2) return 0;
    if (!strcmp(cmd, "CANCEL")) return cur_id && !strcmp(id, cur_id);
    if (!strcmp(cmd, "SUBMIT")) {
        int slot, plen, max_tok; float temp, top_p;
        if (sscanf(ln, "%*s %*s %d %d %d %f %f", &slot, &plen, &max_tok, &temp, &top_p) != 5 ||
            plen < 0 || plen > (1<<22)) { printf("ERROR %s bad submit header\n", id); fflush(stdout); return 0; }
        char *pl = malloc((size_t)plen + 1);
        if (fread(pl, 1, (size_t)plen, stdin) != (size_t)plen) { free(pl); return -1; }
        int nl = fgetc(stdin); (void)nl;
        pl[plen] = 0;
        if (g_qn < SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", id);
            q->max_tok = max_tok; q->temp = temp; q->top_p = top_p;
            q->payload = pl; q->plen = plen;
        } else { printf("ERROR %s queue full\n", id); fflush(stdout); free(pl); }
    }
    return 0;
}

static void serve_one(Model *m, Tok *T, SReq *q) {
    Cfg *c = &m->c;
    int cap = q->plen + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { printf("ERROR %s empty prompt\n", q->id); fflush(stdout); free(ids); return; }
    const char *bad = prompt_reject(c, ids, np, q->max_tok);
    if (bad) { printf("ERROR %s %s\n", q->id, bad); fflush(stdout); free(ids); return; }
    state_reset(m);
    kv_alloc(m, np + q->max_tok + 8);
    double t0 = now_s();
    uint64_t h0 = m->hits, m0 = m->miss;
    /* per-turn phase snapshot for the PROF line (timers accumulate globally) */
    double f0 = m->t_fill, e0 = m->t_expert, s0 = m->t_shared, a0 = m->t_attn;
    float *logit = step(m, ids, np, 0, NULL, NULL, NULL);
    int len = np, gen = 0, limited = 1, cancelled = 0;
    char buf[512];
    /* repetition-penalty history: prompt tail + emitted tokens, ring of 128 */
    float rep = getenv("REP_PEN") ? atof(getenv("REP_PEN")) : 1.1f;
    int hist[128], nhist = 0;
    for (int i = (np > 128 ? np - 128 : 0); i < np; i++) hist[nhist++] = ids[i];
    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        apply_rep_penalty(logit, c->unpad_vocab, hist, nhist, rep);
        int tk = sample_logits(logit, c->unpad_vocab, q->temp, q->top_p);
        free(logit); logit = NULL;
        if (tk == c->eos) { limited = 0; break; }
        if (nhist < 128) hist[nhist++] = tk;
        else { memmove(hist, hist+1, 127*sizeof(int)); hist[127] = tk; }
        int nb = tok_decode(T, &tk, 1, buf, sizeof(buf)-1);
        printf("DATA %s %d\n", q->id, nb);
        fwrite(buf, 1, (size_t)nb, stdout);
        fputc('\n', stdout); fflush(stdout);
        gen++; len++;
        while (stdin_readable()) {
            int r = serve_read_cmd(q->id);
            if (r < 0) { free(ids); return; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
        if (cancelled || s == q->max_tok - 1) break;
        logit = step(m, &tk, 1, len - 1, NULL, NULL, NULL);
    }
    free(logit);
    double dt = now_s() - t0;
    double tot = (double)(m->hits - h0 + m->miss - m0);
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen,
           dt > 0 ? gen/dt : 0.0, tot ? 100.0*(m->hits-h0)/tot : 0.0, rss_gb(), np, limited);
    /* PROF: per-turn phase timings for the dashboard (gateway schema — we map
     * expert_wait -> shared-expert compute, lm_head folded into 0). */
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %d\n", dt, np, gen,
           m->t_fill - f0, m->t_shared - s0, m->t_expert - e0, m->t_attn - a0, 0.0, gen + 1);
    fflush(stdout);
    free(ids);
}

/* ---------- dashboard protocol (HWINFO / TIERS / EMAP) ----------
 * Same stdout lines glm.c emits for the web dashboard; the gateway parses
 * them and the Brain/Profiling pages render live expert-tier state. */
static void serve_hwinfo(Model *m) {
    char cpu[256] = ""; int cores = 0; double rt = 0, ra = 0;
    FILE *ci = fopen("/proc/cpuinfo", "r");
    if (ci) { char ln[256];
        while (fgets(ln, sizeof(ln), ci)) if (!strncmp(ln, "model name", 10)) {
            char *p = strchr(ln, ':'); if (p) { p++; while (*p == ' ') p++;
            int n = (int)strlen(p); if (n > 0 && p[n-1] == '\n') p[--n] = 0;
            snprintf(cpu, sizeof(cpu), "%s", p); } break; }
        fclose(ci); }
#ifdef _SC_NPROCESSORS_ONLN
    cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    FILE *mi = fopen("/proc/meminfo", "r");
    if (mi) { char ln[256]; double v = 0;
        while (fgets(ln, sizeof(ln), mi)) {
            if (sscanf(ln, "MemTotal: %lf", &v) == 1) rt = v/1e6;
            if (sscanf(ln, "MemAvailable: %lf", &v) == 1) ra = v/1e6;
        } fclose(mi); }
    int ngpu = 0; double vram = 0;
    const char *gpu = "";
#ifdef COLI_CUDA
    if (g_cuda) { ngpu = 1; vram = ink_cuda_total_bytes()/1e9; gpu = "CUDA device"; }
#endif
    (void)m;
    printf("HWINFO %d %.1f %.1f %d %.1f %s|%s\n", cores, rt, ra, ngpu, vram, cpu[0]?cpu:"unknown", gpu);
    fflush(stdout);
}

static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts;
    int nsp = 0, filled = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) { nsp++; filled += m->cache[i].n; }
    int64_t I = c->moe_inter, D = c->hidden;
    int64_t slotb = m->xq ? m->rb13*2*I + m->rb2*D + (2*I+D)*4
                  : m->quant_bits ? 3*I*D + (2*I+D)*4 : 3*I*D*4;
    printf("TIERS 0 %d %d 0.00 %.2f\n", filled, nsp*E - filled, filled*(double)slotb/1e9);
    /* EMAP: 1 byte/expert hex — tier(2b: 0=disk 1=RAM)<<6 | heat(6b: log2 usage) */
    char *hex = malloc((size_t)nsp*E*2 + 1); int w = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (!c->sparse[i]) continue;
        LCache *lc = &m->cache[i];
        for (int e = 0; e < E; e++) {
            int tier = 0;
            for (int z = 0; z < lc->n; z++) if (lc->slots[z].eid == e && lc->slots[z].filled) { tier = 1; break; }
            uint32_t u = m->eusage[i] ? m->eusage[i][e] : 0;
            int heat = 0; while (u) { heat++; u >>= 1; } if (heat > 63) heat = 63;
            int b = (tier << 6) | heat;
            hex[w++] = "0123456789abcdef"[b >> 4];
            hex[w++] = "0123456789abcdef"[b & 15];
        }
    }
    hex[w] = 0;
    printf("EMAP %d %d %s\n", nsp, E, hex);
    fflush(stdout); free(hex);
}

static void serve_loop(Model *m, Tok *T) {
    setvbuf(stdin, NULL, _IONBF, 0);
    const char *sd = getenv("SEED");
    if (sd) g_rng ^= (uint64_t)strtoull(sd, NULL, 10);
    else g_rng ^= (uint64_t)time(NULL) * 2654435761u;
    /* the gateway reads a STAT line right after the READY sentinel (glm
     * reports its load stats there) — match the handshake */
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb());
    fflush(stdout);
    serve_hwinfo(m);
    serve_tiers_emap(m);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(NULL) < 0) return;   /* blocks on stdin */
        SReq q = g_q[0];
        memmove(g_q, g_q+1, (size_t)(--g_qn) * sizeof(SReq));
        serve_one(m, T, &q);
        serve_tiers_emap(m);
        free(q.payload);
    }
}

/* ---------- ref_inkling.json harness ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { *n_out = 0; return NULL; }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

int main(int argc, char **argv) {
    /* OpenMP hot-thread tuning, same trick (and rationale) as glm.c: the
     * per-expert matmul regions are tiny and back-to-back; the default passive
     * wait policy parks the team between regions and re-wake latency dominates.
     * libgomp reads OMP_/GOMP_ vars before main(), so seed them and re-exec
     * once (COLI_OMP_TUNED guards the exec; COLI_NO_OMP_TUNE=1 disables).
     * NOT under CUDA — same exception glm.c makes: a spinning 24-thread team
     * starves the CUDA driver during every stream sync. */
#ifndef COLI_CUDA
    if (!getenv("COLI_OMP_TUNED") && !getenv("COLI_NO_OMP_TUNE")) {
        setenv("OMP_WAIT_POLICY","active",0);
        setenv("GOMP_SPINCOUNT","200000",0);
        /* LLVM libomp ignores GOMP_* and turns WAIT_POLICY=active into
         * KMP_BLOCKTIME=infinite — a parked serve-mode engine then spins at
         * 100% x nthreads forever (upstream glm.c #341). 200 ms keeps the team
         * hot across back-to-back expert matmuls, asleep at the prompt. */
        setenv("KMP_BLOCKTIME","200",0);
        setenv("OMP_PROC_BIND","close",0);
        setenv("OMP_DYNAMIC","FALSE",0);
        setenv("COLI_OMP_TUNED","1",1);
#ifdef __linux__
        execv("/proc/self/exe", argv);
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
#endif  /* !COLI_CUDA */
#ifdef _WIN32
    /* CRT text mode turns \n into \r\n, corrupting the byte-framed serve
     * protocol (READY sentinel, DATA frames) — same fix as glm.c */
    _setmode(fileno(stdout), O_BINARY);
#endif
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    /* flags: -p "prompt" [-n N] [-l lora_dir] -> generate mode;
     * positional: [cap] [bits] [ref.json]. LORA=<dir> env is the serve-mode
     * equivalent of -l (the gateway passes it through). */
    const char *prompt = NULL, *pfile = NULL, *refpath = "ref_inkling.json";
    const char *lora_dir = getenv("LORA");
    int cap = -1, bits = 0, n_new = 256, npos = 0, mtp_oracle = 0, mtp_accept = 0, cuda_q4_test = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-f") && i+1 < argc) pfile = argv[++i];
        else if (!strcmp(argv[i], "-l") && i+1 < argc) lora_dir = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1 < argc) n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mtp-oracle") && i+1 < argc) { mtp_oracle = 1; refpath = argv[++i]; }
        else if (!strcmp(argv[i], "--mtp-accept") && i+1 < argc) { mtp_accept = 1; refpath = argv[++i]; }
        else if (!strcmp(argv[i], "--cuda-q4-test")) cuda_q4_test = 1;
        else if (npos == 0) { cap = atoi(argv[i]); npos++; }
        else if (npos == 1) { bits = atoi(argv[i]); npos++; }
        else refpath = argv[i];
    }
    (void)cuda_q4_test;   /* only read under COLI_CUDA */

#ifdef COLI_CUDA
    if (cuda_q4_test) {   /* GPU int4 GEMM vs CPU matmul_q4 (run with IDOT=0 for the scalar ref) */
        if (ink_cuda_init(getenv("GPU_DEV") ? atoi(getenv("GPU_DEV")) : 0) != 0) {
            fprintf(stderr, "[q4-test] CUDA init failed\n"); return 1; }
        srand(1234);
        int I = 6144, O = 6144;
        uint8_t *packed = malloc((size_t)O*(I/2));
        float *scale = malloc((size_t)O*4), *x = malloc((size_t)I*4);
        for (size_t j = 0; j < (size_t)O*(I/2); j++) packed[j] = (uint8_t)(rand() & 0xFF);
        for (int o = 0; o < O; o++) scale[o] = ((rand()%2000)-1000)/100000.f;
        for (int i = 0; i < I; i++) x[i] = ((rand()%2000)-1000)/1000.f;
        float *ycpu = malloc((size_t)O*4), *ygpu = malloc((size_t)O*4);
        matmul_q4(ycpu, x, packed, scale, I, O);
        void *dp = ink_cuda_upload(packed, (size_t)O*(I/2)), *ds = ink_cuda_upload(scale, (size_t)O*4);
        if (!dp || !ds || ink_cuda_matmul_q4(ygpu, x, dp, ds, 1, I, O) != 0) {
            fprintf(stderr, "[q4-test] GPU path failed\n"); return 1; }
        double maxabs = 0, meany = 0;
        for (int o = 0; o < O; o++) {
            double d = fabs((double)ycpu[o] - ygpu[o]);
            if (d > maxabs) maxabs = d;
            meany += fabs((double)ycpu[o]);
        }
        meany /= O;
        /* per-element relative error is meaningless on near-zero outputs; the
         * meaningful metric is max abs error vs the output scale (f32 reorder). */
        double rel = maxabs / (meany + 1e-9);
        int ok = rel < 1e-3;
        printf("q4 GPU vs CPU (I=%d O=%d): max abs %.2e, mean|y| %.3f, scaled-rel %.2e  %s\n",
               I, O, maxabs, meany, rel, ok ? "[OK]" : "[FAIL]");

        /* int8 residents kernel: signed int8 weights + per-row scale vs CPU matmul_q */
        int8_t *q8 = malloc((size_t)O*I);
        float *q8s = malloc((size_t)O*4);
        for (size_t j = 0; j < (size_t)O*I; j++) q8[j] = (int8_t)((rand()%255)-127);
        for (int o = 0; o < O; o++) q8s[o] = ((rand()%2000)-1000)/1000000.f;
        matmul_q(ycpu, x, q8, q8s, I, O);
        void *dq8 = ink_cuda_upload(q8, (size_t)O*I), *dq8s = ink_cuda_upload(q8s, (size_t)O*4);
        int ok8 = 0;
        if (dq8 && dq8s && ink_cuda_matmul_q8(ygpu, x, dq8, dq8s, 1, I, O) == 0) {
            double ma = 0, my = 0;
            for (int o = 0; o < O; o++) { double d = fabs((double)ycpu[o]-ygpu[o]); if (d>ma) ma=d; my += fabs((double)ycpu[o]); }
            my /= O; double r8 = ma/(my+1e-9); ok8 = r8 < 1e-3;
            printf("q8 GPU vs CPU (I=%d O=%d): max abs %.2e, mean|y| %.3f, scaled-rel %.2e  %s\n",
                   I, O, ma, my, r8, ok8 ? "[OK]" : "[FAIL]");
        } else fprintf(stderr, "[q8-test] GPU path failed\n");

        /* fused GPU expert vs CPU (matmul_q4 + siluf + matmul_q4). D=I=6144. */
        int okf = 0;
        {
            int Dd = 6144, Ii = 6144;
            uint8_t *p13 = malloc((size_t)(2*Ii)*(Dd/2)), *p2 = malloc((size_t)Dd*(Ii/2));
            float *s13 = malloc((size_t)(2*Ii)*4), *s2 = malloc((size_t)Dd*4);
            for (size_t j = 0; j < (size_t)(2*Ii)*(Dd/2); j++) p13[j] = (uint8_t)(rand()&0xFF);
            for (size_t j = 0; j < (size_t)Dd*(Ii/2); j++) p2[j] = (uint8_t)(rand()&0xFF);
            for (int o = 0; o < 2*Ii; o++) s13[o] = ((rand()%2000)-1000)/100000.f;
            for (int o = 0; o < Dd; o++)   s2[o]  = ((rand()%2000)-1000)/100000.f;
            float *gc = malloc((size_t)2*Ii*4), *hc = malloc((size_t)Dd*4), *hg = malloc((size_t)Dd*4);
            matmul_q4(gc, x, p13, s13, Dd, 2*Ii);
            for (int i = 0; i < Ii; i++) gc[i] = siluf(gc[i]) * gc[Ii+i];
            matmul_q4(hc, gc, p2, s2, Ii, Dd);
            void *dp13=ink_cuda_upload(p13,(size_t)(2*Ii)*(Dd/2)), *ds13=ink_cuda_upload(s13,(size_t)(2*Ii)*4);
            void *dp2 =ink_cuda_upload(p2,(size_t)Dd*(Ii/2)),      *ds2 =ink_cuda_upload(s2,(size_t)Dd*4);
            if (dp13 && ds13 && dp2 && ds2 &&
                ink_cuda_expert_q4(hg, x, dp13, ds13, dp2, ds2, Ii, Dd) == 0) {
                double ma=0,my=0; for (int o=0;o<Dd;o++){double d=fabs((double)hc[o]-hg[o]);if(d>ma)ma=d;my+=fabs((double)hc[o]);}
                my/=Dd; double rf=ma/(my+1e-9); okf = rf < 1e-3;
                printf("fused-expert GPU vs CPU (D=%d I=%d): max abs %.2e, mean|y| %.3f, scaled-rel %.2e  %s\n",
                       Dd, Ii, ma, my, rf, okf ? "[OK]" : "[FAIL]");
            } else fprintf(stderr, "[fused-expert-test] GPU path failed\n");
            free(p13); free(p2); free(s13); free(s2); free(gc); free(hc); free(hg);
        }
        return (ok && ok8 && okf) ? 0 : 1;
    }
#endif

    int serve = getenv("SERVE") && *getenv("SERVE") == '1';
    if (cap < 0) cap = (prompt || pfile || serve) ? 0 : 16;   /* generate/serve default to RAM-sized auto cap */
    if (bits && (bits < 2 || bits > 8)) { fprintf(stderr, "quant_bits must be 0 (f32) or 2..8\n"); return 1; }

    if (prompt || pfile || serve) {
        Model m; model_init(&m, snap, cap, bits, lora_dir);
        if (!serve)
            printf("== Inkling C engine, %d layers, experts @ %s%s, cache %d/layer ==\n",
                   m.c.n_layers, m.xq ? "container" : bits ? "int" : "f32",
                   m.lora.on ? " + lora" : "", m.cache[0].cap);
        pins_load(&m, snap);
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        if (serve) { serve_loop(&m, &T); usage_save(&m, snap); return 0; }
        if (prompt) generate_stream(&m, &T, prompt, n_new);
        else {   /* -f: one prompt per line, model loaded once, usage accumulates */
            FILE *pf = fopen(pfile, "rb"); if (!pf) { perror(pfile); return 1; }
            char ln[8192]; int np = 0;
            while (fgets(ln, sizeof(ln), pf)) {
                size_t n = strlen(ln); while (n && (ln[n-1]=='\n'||ln[n-1]=='\r')) ln[--n]=0;
                if (!n || ln[0]=='#') continue;
                printf("\n===== prompt %d =====\n", ++np);
                state_reset(&m);
                generate_stream(&m, &T, ln, n_new);
            }
            fclose(pf);
        }
        int saved = usage_save(&m, snap);
        double tot = m.hits + m.miss;
        printf("[cache] hit %.1f%% (%llu hit / %llu load)%s\n",
               tot ? 100.0*m.hits/tot : 0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss,
               saved ? " | usage history saved" : "");
        return 0;
    }

    FILE *f = fopen(refpath, "rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull, ntf;
    int *pids  = read_int_array(ref,"prompt_ids",&np);
    int *full  = read_int_array(ref,"full_ids",&nfull);
    int *tfref = read_int_array(ref,"tf_pred",&ntf);
    int ngen = nfull - np;

    Model m; model_init(&m, snap, cap, bits, lora_dir);
    printf("== Inkling C engine (Stage A), cache = %d experts/layer, experts @ %s%s ==\n",
           cap, m.xq ? "container (int4/int8 + .qs)" : bits ? "int (runtime quant)" : "f32",
           m.lora.on ? " + lora" : "");
    printf("cfg: D=%d L=%d V=%d(%d) heads=%d/%d kv=%d/%d hd=%d win=%d d_rel=%d ext=%d E=%d+%d topk=%d\n",
           m.c.hidden, m.c.n_layers, m.c.vocab, m.c.unpad_vocab, m.c.n_heads, m.c.swa_heads,
           m.c.n_kv, m.c.swa_kv, m.c.head_dim, m.c.window, m.c.d_rel, m.c.rel_extent,
           m.c.n_experts, m.c.n_shared, m.c.topk);
    printf("resident weights loaded in %.1fs | RSS: %.2f GB\n", m.dense_load_s, rss_gb());
    /* opt-in only (preserves the oracle's cold-cache default): seed the heat
     * ranking + pin hot experts. Numerically a no-op — pins only pre-load — but
     * enables the heat-tiered quant probe's per-layer keep threshold. */
    if (getenv("PIN") || getenv("QSIM_BITS")) pins_load(&m, snap);

    /* MTP chain teacher-forced oracle: main forward -> pre-norm hidden -> run the
     * module chain over positions, argmax at each depth vs the transformers ref. */
    if (mtp_oracle) {
        if (!m.n_mtp) mtp_load(&m);   /* these modes force-load the head */
        if (!m.n_mtp) { fprintf(stderr, "[mtp-oracle] no MTP head in %s\n", snap); return 1; }
        jval *mpred = json_get(ref, "mtp_pred");
        int nmods = (mpred && mpred->t == J_ARR) ? mpred->len : 0;
        int D = m.c.hidden;
        kv_alloc(&m, np + 8); state_reset(&m);
        float *hid = falloc((int64_t)np * D);
        float *lg = step(&m, pids, np, 0, NULL, NULL, hid); free(lg);
        float *hk = malloc((int64_t)np * D * sizeof(float));
        memcpy(hk, hid, (int64_t)np * D * sizeof(float));       /* h^0 */
        int *toks = malloc((size_t)np * sizeof(int)), *pred = malloc((size_t)np * sizeof(int));
        int all_ok = 1;
        for (int k = 0; k < nmods && k < m.n_mtp; k++) {
            int P = np - (k + 1);
            if (P <= 0) break;
            for (int i = 0; i < P; i++) toks[i] = pids[i + k + 1];   /* t_{i+k+1} */
            float *hout = malloc((int64_t)P * D * sizeof(float));
            mtp_module_tf(&m, k, hk, toks, P, hout, pred);
            jval *rk = mpred->kids[k];
            int match = 0; for (int i = 0; i < P && i < rk->len; i++) if (pred[i] == (int)rk->kids[i]->num) match++;
            int ok = (match == P); if (!ok) all_ok = 0;
            printf("MTP module-%d: %d/%d match %s\n", k, match, P, ok ? "[OK]" : "[FAIL]");
            memcpy(hk, hout, (int64_t)P * D * sizeof(float));       /* h^{k+1} for next depth */
            free(hout);
        }
        free(hid); free(hk); free(toks); free(pred);
        return all_ok ? 0 : 1;
    }

    /* MTP acceptance on a real token sequence: for each depth k, how often does
     * the MTP draft equal the main model's own greedy token k+2 ahead. Estimates
     * the expected accepted draft length (the speculative speedup) using only the
     * validated teacher-forced chain — no serve loop required. */
    if (mtp_accept) {
        if (!m.n_mtp) mtp_load(&m);   /* force-load the head */
        if (!m.n_mtp) { fprintf(stderr, "[mtp-accept] no MTP head in %s\n", snap); return 1; }
        int nt = 0; int *seq = full;                       /* prefer full_ids; else prompt_ids */
        if (seq) nt = nfull; else seq = read_int_array(ref, "prompt_ids", &nt);
        if (nt < 4) { fprintf(stderr, "[mtp-accept] need >= 4 tokens\n"); return 1; }
        int D = m.c.hidden;
        kv_alloc(&m, nt + 8); state_reset(&m);
        int *mainpred = malloc((size_t)nt * sizeof(int));
        float *hid = falloc((int64_t)nt * D);
        float *lg = step(&m, seq, nt, 0, mainpred, NULL, hid); free(lg);   /* mainpred[i] = greedy t_{i+1} */
        float *hk = malloc((int64_t)nt * D * sizeof(float));
        memcpy(hk, hid, (int64_t)nt * D * sizeof(float));
        int *toks = malloc((size_t)nt * sizeof(int)), *pred = malloc((size_t)nt * sizeof(int));
        double chain = 1.0, exp_len = 1.0;
        int chain_main = getenv("MTP_MAIN_HIDDEN") != NULL;   /* modules off the main hidden vs chained */
        printf("MTP acceptance on %d tokens (%d depths, hidden=%s):\n", nt, m.n_mtp,
               chain_main ? "main" : "chained");
        for (int k = 0; k < m.n_mtp; k++) {
            int P = nt - (k + 1); if (P <= 0) break;
            for (int i = 0; i < P; i++) toks[i] = seq[i + k + 1];
            float *hout = malloc((int64_t)P * D * sizeof(float));
            mtp_module_tf(&m, k, hk, toks, P, hout, pred);
            int match = 0, n = 0;
            for (int i = 0; i < P && i + k + 1 < nt; i++) { if (pred[i] == mainpred[i + k + 1]) match++; n++; }
            double ak = n ? (double)match / n : 0.0;
            chain *= ak; exp_len += chain;
            printf("  depth %d: accept %5.1f%% (%d/%d)  chained %5.1f%%\n", k, 100*ak, match, n, 100*chain);
            if (!chain_main) memcpy(hk, hout, (int64_t)P * D * sizeof(float));   /* else keep h^0 */
            free(hout);
        }
        printf("expected tokens/verify: %.2f (max %d)  -> ~%.2fx decode\n", exp_len, m.n_mtp + 1, exp_len);
        free(hid); free(hk); free(toks); free(pred); free(mainpred);
        return 0;
    }
    kv_alloc(&m, nfull + 8);

    /* pass 1: teacher-forced perplexity (+ argmax match if tf_pred present).
     * PPL validates the WEIGHTS — a broken conversion tanks perplexity even when
     * argmax still looks plausible. Works on any token list (full_ids alone) so
     * it doubles as a real-model health check; compares to ppl_ref if present. */
    if (nfull > 1) {
        int have_tf = tfref && ntf == nfull;
        int *tf = have_tf ? malloc(nfull * sizeof(int)) : NULL;
        double nll = 0;
        float *lg = step(&m, full, nfull, 0, tf, &nll, NULL);
        free(lg);
        double ppl = exp(nll / (nfull - 1));
        if (have_tf) {
            int ok = 0; for (int i = 0; i < nfull; i++) ok += (tf[i] == tfref[i]);
            printf("teacher-forced argmax: %d/%d match | perplexity: %.4f\n", ok, nfull, ppl);
            free(tf);
        } else {
            printf("perplexity (%d tokens): %.4f\n", nfull, ppl);
        }
        jval *pr = json_get(ref, "ppl_ref");
        if (pr && pr->t == J_NUM) {
            double rel = fabs(ppl - pr->num) / pr->num;
            printf("perplexity vs transformers ref %.4f: %.2f%% rel diff %s\n",
                   pr->num, 100.0*rel, rel < 0.02 ? "[OK]" : rel < 0.10 ? "[quant-noise]" : "[FAIL]");
        }
        state_reset(&m);
    }

    /* pass 2: greedy generation, token-for-token vs the oracle */
    int *out = malloc(nfull * sizeof(int));
    double t = now_s();
    generate(&m, pids, np, ngen, out);
    double dt = now_s() - t;
    int match = 0;
    printf("Reference: "); for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, ngen);
    double tot = m.hits + m.miss;
    printf("PEAK RSS: %.2f GB | expert cache hit %.1f%% | %.2f tok/s\n",
           rss_gb(), tot?100.0*m.hits/tot:0.0, ngen/dt);

    /* Phase-1 lossless self-test: speculative decode must reproduce plain greedy
     * token-for-token over a long generation (frequent mispredicts stress the
     * partial-acceptance conv rollback). SPEC_TEST=1; SPEC_N sets length. */
    int spec_ok = 1;
    if (getenv("SPEC_TEST")) {
        int N = getenv("SPEC_N") ? atoi(getenv("SPEC_N")) : 128; if (N < 8) N = 8;
        int maxG = getenv("DRAFT") ? atoi(getenv("DRAFT")) : 8; if (maxG < 1) maxG = 8;
        kv_alloc(&m, np + N + maxG + 8);
        int *gout = malloc((size_t)(np + N) * sizeof(int));
        state_reset(&m); generate(&m, pids, np, N, gout);       /* greedy reference */
        for (int G = 1; G <= maxG; G++) {
            state_reset(&m);
            int *sout = malloc((size_t)(np + N) * sizeof(int));
            double ts = now_s();
            double tpf = generate_spec(&m, pids, np, N, sout, G);
            double dts = now_s() - ts;
            int sm = 0; for (int i = np; i < np + N; i++) if (sout[i] == gout[i]) sm++;
            int ok = (sm == N); if (!ok) spec_ok = 0;
            printf("SPEC G=%d: %d/%d vs greedy | %.2f tok/verify | %.1f tok/s %s\n",
                   G, sm, N, tpf, dts>0?N/dts:0.0, ok ? "[LOSSLESS]" : "[DIVERGED]");
            free(sout);
        }
        if (m.n_mtp >= 1) {                               /* MTP depth-0 drafter path */
            state_reset(&m);
            int *sout = malloc((size_t)(np + N) * sizeof(int));
            double tpf = generate_spec_mtp(&m, pids, np, N, sout);
            int sm = 0; for (int i = np; i < np + N; i++) if (sout[i] == gout[i]) sm++;
            int ok = (sm == N); if (!ok) spec_ok = 0;
            printf("SPEC MTP: %d/%d vs greedy | %.2f tok/verify %s\n",
                   sm, N, tpf, ok ? "[LOSSLESS]" : "[DIVERGED]");
            free(sout);
        }
        free(gout);
    }
    free(buf); free(arena);
    return (match == ngen && spec_ok) ? 0 : 1;
}
