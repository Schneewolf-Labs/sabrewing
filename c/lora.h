/* LoRA adapter serving for inkling.c — Tinker raw adapter format.
 *
 * The adapter (thinkingmachines Tinker, `adapter_model.safetensors` +
 * `adapter_config.json`) targets every linear in the model. Two weight
 * classes get two different treatments:
 *
 *  RESIDENTS (attn q/k/v/r/o, dense MLP, shared experts, lm_head):
 *    merged into the resident weights at load time, W += (alpha/r)*B*A,
 *    before any CUDA upload — zero decode-time cost. bf16 residents are
 *    merged with round-to-nearest-even (same rounding torch uses for an
 *    in-place bf16 += f32, i.e. PEFT merge_and_unload semantics).
 *
 *  ROUTED EXPERTS: never merged — the int4 container stays pristine (a
 *    rank-32 delta is below the int4 quantization step; baking it in would
 *    destroy most of the signal, and would mean requantizing ~470 GB per
 *    adapter). Instead the A/B tensors stay resident in RAM as f32 and a
 *    low-rank correction is applied around each expert's base matmul.
 *    The expert LRU cache and pinned set are untouched by the adapter.
 *
 * Tinker's expert factorization makes the runtime path cheap:
 *    w1/w3 (gate/up): lora_A [1, r, D]  SHARED across experts,
 *                     lora_B [E, I, r]  per-expert
 *    w2 (down):       lora_A [E, r, I]  per-expert,
 *                     lora_B [1, D, r]  SHARED
 * so A1·x / A3·x are computed once per (layer, token) regardless of top-k,
 * and the shared B2 apply is hoisted out of the expert loop entirely:
 *    out += B2 · sum_k( w_k * scale * A2_ek · h_ek )
 * which is exact by linearity (the router weight multiplies a linear map).
 * Net per-expert cost is 3*I*r MACs vs 3*I*D for the base matmuls: <1%.
 *
 * Name mapping (Tinker -> engine), assumptions validated by the tiny oracle
 * (tools/make_tiny_lora.py):
 *    attn.wq_du/wk_dv/wv_dv/wr_du/wo_ud  -> self_attn.{q,k,v,r,o}_proj
 *    mlp.gate_up_proj (dense, fused)     -> mlp.gate_proj rows [0,I),
 *                                           mlp.up_proj   rows [I,2I)
 *    mlp.experts.w1/w3/w2                -> experts gate/up/down
 *    mlp.shared_experts.w1/w3            -> shared gate/up (fused over ns)
 *    mlp.shared_experts.w2               -> shared down: engine keeps ns
 *                                           consecutive [D,I] blocks, the
 *                                           adapter's fused [D, ns*I] view
 *                                           maps to per-block A column slices
 *    lm_head                             -> lm_head (padded vocab rows)
 * Router, norms, convs and embed_tokens are not targeted by Tinker. */
#ifndef LORA_H
#define LORA_H
#include <time.h>
#include "st.h"

#define LORA_MAX_R 256

static double lora_now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct {
    int on;                 /* adapter open (residents merge as they load) */
    int experts_on;         /* expert tensors present + loaded */
    int r;
    float scale;            /* lora_alpha / r * LORA_SCALE */
    shards S;               /* adapter_model.safetensors index */
    /* model dims, copied from Cfg at open */
    int n_layers, D, moe_I, ns, E;
    const unsigned char *sparse;
    /* per-layer routed-expert tensors, f32, NULL on dense layers.
     * Layouts are verbatim from the file — every runtime access below is a
     * contiguous rank-r (or length-I) dot, no transposition needed. */
    float **a1, **b1;       /* [r*D] shared, [E*I*r] per-expert  (gate) */
    float **a3, **b3;       /* [r*D] shared, [E*I*r] per-expert  (up)   */
    float **a2, **b2;       /* [E*r*I] per-expert, [D*r] shared  (down) */
    double expert_gb;
    double load_s;
} Lora;

static inline uint16_t lora_f32_to_bf16_rne(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

/* read one adapter tensor as f32, validating element count */
static float *lora_read(Lora *lr, const char *name, int64_t want) {
    int64_t n = st_numel(&lr->S, name);
    if (n < 0) { fprintf(stderr, "[lora] missing tensor %s\n", name); exit(1); }
    if (want > 0 && n != want) {
        fprintf(stderr, "[lora] %s: %lld elements, expected %lld — adapter/model mismatch\n",
                name, (long long)n, (long long)want); exit(1);
    }
    float *p = malloc((size_t)n * sizeof(float));
    if (!p) { fprintf(stderr, "[lora] OOM %s\n", name); exit(1); }
    st_read_f32(&lr->S, name, p, 1);
    return p;
}

/* index the adapter + read config. Residents merge lazily (lora_merge_resident
 * from the engine's load path); expert tensors load via lora_load_experts. */
static void lora_open(Lora *lr, const char *dir, int n_layers, int D, int moe_I,
                      int ns, int E, const unsigned char *sparse) {
    memset(lr, 0, sizeof(*lr));
    lr->n_layers = n_layers; lr->D = D; lr->moe_I = moe_I;
    lr->ns = ns; lr->E = E; lr->sparse = sparse;

    char path[2048]; snprintf(path, sizeof(path), "%s/adapter_config.json", dir);
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1); if (fread(buf, 1, n, f) != (size_t)n) {} buf[n] = 0; fclose(f);
    char *arena = NULL; jval *root = json_parse(buf, &arena);
    jval *jr = json_get(root, "r"), *ja = json_get(root, "lora_alpha");
    if (!jr || jr->t != J_NUM || !ja || ja->t != J_NUM) {
        fprintf(stderr, "[lora] %s: missing r / lora_alpha\n", path); exit(1); }
    lr->r = (int)jr->num;
    if (lr->r < 1 || lr->r > LORA_MAX_R) { fprintf(stderr, "[lora] rank %d out of range\n", lr->r); exit(1); }
    lr->scale = (float)(ja->num / jr->num);
    const char *sc = getenv("LORA_SCALE");
    if (sc) lr->scale *= (float)atof(sc);
    free(buf); free(arena);

    st_init(&lr->S, dir);
    lr->on = 1;
    fprintf(stderr, "[lora] %s: r=%d scale=%.3f, %d tensors\n", dir, lr->r, lr->scale, lr->S.n);
}

/* W[boff+o][aoff..aoff+I) += scale * B[boff+o] · A[·][aoff..], o in [0,O).
 * A is [r, astr] row-major, B is [*, r] row-major. Exactly one of h/f is set;
 * bf16 merges round-to-nearest-even per element. */
static void lora_merge_block(uint16_t *h, float *f, int64_t O, int64_t I,
                             const float *A, int64_t astr, int64_t aoff,
                             const float *B, int64_t boff, int r, float scale) {
    #pragma omp parallel
    {
        float *row = malloc((size_t)I * sizeof(float));
        #pragma omp for schedule(static)
        for (int64_t o = 0; o < O; o++) {
            memset(row, 0, (size_t)I * sizeof(float));
            const float *br = B + (boff + o) * r;
            for (int k = 0; k < r; k++) {
                float bk = scale * br[k];
                const float *ak = A + (int64_t)k * astr + aoff;
                for (int64_t i = 0; i < I; i++) row[i] += bk * ak[i];
            }
            if (h) { uint16_t *w = h + o * I;
                     for (int64_t i = 0; i < I; i++)
                         w[i] = lora_f32_to_bf16_rne(bf16_to_f32(w[i]) + row[i]); }
            else   { float *w = f + o * I;
                     for (int64_t i = 0; i < I; i++) w[i] += row[i]; }
        }
        free(row);
    }
}

/* If the adapter targets engine tensor `name`, merge its delta in place
 * (h = bf16 buffer or f = f32 buffer, numel elements). Returns 1 if merged.
 * Called from the engine's resident load path, before any CUDA upload. */
static int lora_merge_resident(Lora *lr, const char *name, uint16_t *h, float *f, int64_t numel) {
    if (!lr->on) return 0;
    enum { DIRECT, GATE_HALF, UP_HALF, SHARED_DOWN } mode = DIRECT;
    char ad[256];
    int li;
    char rest[160];
    if (!strcmp(name, "lm_head.weight")) {
        snprintf(ad, sizeof(ad), "language_model.lm_head");
    } else if (sscanf(name, "model.layers.%d.%159s", &li, rest) == 2) {
        size_t rl = strlen(rest);
        if (rl > 7 && !strcmp(rest + rl - 7, ".weight")) rest[rl - 7] = 0;
        const char *map = NULL;
        if      (!strcmp(rest, "self_attn.q_proj")) map = "attn.wq_du";
        else if (!strcmp(rest, "self_attn.k_proj")) map = "attn.wk_dv";
        else if (!strcmp(rest, "self_attn.v_proj")) map = "attn.wv_dv";
        else if (!strcmp(rest, "self_attn.r_proj")) map = "attn.wr_du";
        else if (!strcmp(rest, "self_attn.o_proj")) map = "attn.wo_ud";
        else if (!strcmp(rest, "mlp.gate_proj")) { map = "mlp.gate_up_proj"; mode = GATE_HALF; }
        else if (!strcmp(rest, "mlp.up_proj"))   { map = "mlp.gate_up_proj"; mode = UP_HALF; }
        else if (!strcmp(rest, "mlp.down_proj"))   map = "mlp.down_proj";
        else if (!strcmp(rest, "mlp.shared_experts.gate_proj")) map = "mlp.shared_experts.w1";
        else if (!strcmp(rest, "mlp.shared_experts.up_proj"))   map = "mlp.shared_experts.w3";
        else if (!strcmp(rest, "mlp.shared_experts.down_proj")) { map = "mlp.shared_experts.w2"; mode = SHARED_DOWN; }
        else return 0;
        snprintf(ad, sizeof(ad), "language_model.layers.%d.%s", li, map);
    } else return 0;

    char an[288], bn[288];
    snprintf(an, sizeof(an), "%s.lora_A.weight", ad);
    snprintf(bn, sizeof(bn), "%s.lora_B.weight", ad);
    if (!st_has(&lr->S, an)) return 0;   /* adapter doesn't target this tensor */
    float *A = lora_read(lr, an, 0), *B = lora_read(lr, bn, 0);
    int r = lr->r;
    int64_t acols = st_numel(&lr->S, an) / r;   /* A: [r, acols] */
    int64_t brows = st_numel(&lr->S, bn) / r;   /* B: [brows, r] */
    switch (mode) {
    case DIRECT:
        if (brows * acols != numel) { fprintf(stderr, "[lora] %s: shape mismatch\n", name); exit(1); }
        lora_merge_block(h, f, brows, acols, A, acols, 0, B, 0, r, lr->scale);
        break;
    case GATE_HALF: case UP_HALF: {
        /* fused gate_up B: gate rows then up rows (w1/w3 = gate/up convention) */
        int64_t O = numel / acols;
        if (brows != 2 * O || O * acols != numel) { fprintf(stderr, "[lora] %s: fused shape mismatch\n", name); exit(1); }
        lora_merge_block(h, f, O, acols, A, acols, 0, B, mode == UP_HALF ? O : 0, r, lr->scale);
        break;
    }
    case SHARED_DOWN: {
        /* engine: ns consecutive [D, I] blocks; adapter A [r, ns*I], B [D, r].
         * Block j takes A columns [j*I, (j+1)*I). */
        int64_t D = brows, nsI = acols, I = nsI / lr->ns;
        if (D * nsI != numel || I * lr->ns != nsI) { fprintf(stderr, "[lora] %s: shared-down shape mismatch\n", name); exit(1); }
        for (int j = 0; j < lr->ns; j++)
            lora_merge_block(h ? h + (int64_t)j * D * I : NULL,
                             f ? f + (int64_t)j * D * I : NULL,
                             D, I, A, nsI, (int64_t)j * I, B, 0, r, lr->scale);
        break;
    }
    }
    free(A); free(B);
    return 1;
}

/* load the routed-expert A/B tensors resident (f32, verbatim layout).
 * ~19 GB for the 975B rank-32 adapter — do this BEFORE the expert-cache
 * auto-cap reads MemAvailable so the budget accounts for it. */
static void lora_load_experts(Lora *lr) {
    if (!lr->on) return;
    double t0 = lora_now_s();
    int L = lr->n_layers, r = lr->r;
    int64_t D = lr->D, I = lr->moe_I, E = lr->E;
    lr->a1 = calloc(L, sizeof(float*)); lr->b1 = calloc(L, sizeof(float*));
    lr->a3 = calloc(L, sizeof(float*)); lr->b3 = calloc(L, sizeof(float*));
    lr->a2 = calloc(L, sizeof(float*)); lr->b2 = calloc(L, sizeof(float*));
    int64_t bytes = 0;
    for (int i = 0; i < L; i++) {
        if (!lr->sparse[i]) continue;
        char nm[256];
        snprintf(nm, sizeof(nm), "language_model.layers.%d.mlp.experts.w1.lora_A.weight", i);
        if (!st_has(&lr->S, nm)) continue;   /* residents-only adapter */
        #define ERD(dst, w, part, want) \
            snprintf(nm, sizeof(nm), "language_model.layers.%d.mlp.experts." w ".lora_" part ".weight", i); \
            lr->dst[i] = lora_read(lr, nm, want); bytes += (int64_t)(want) * 4
        ERD(a1, "w1", "A", r*D);    ERD(b1, "w1", "B", E*I*r);
        ERD(a3, "w3", "A", r*D);    ERD(b3, "w3", "B", E*I*r);
        ERD(a2, "w2", "A", E*r*I);  ERD(b2, "w2", "B", D*r);
        #undef ERD
        lr->experts_on = 1;
    }
    lr->expert_gb = bytes / 1e9;
    lr->load_s = lora_now_s() - t0;
    fprintf(stderr, "[lora] expert tensors: %.1f GB resident (f32) in %.1fs%s\n",
            lr->expert_gb, lr->load_s, lr->experts_on ? "" : " — residents-only adapter");
}

/* ---------- runtime path, called from moe() ----------
 * Per (layer, token):  lora_moe_pre        t1 = A1·x, t3 = A3·x   (shared A)
 * Per selected expert: lora_moe_gu         g += s·B1e·t1, u += s·B3e·t3
 *                      lora_moe_down_acc   acc2 += w_k·s·(A2e·h)
 * Per (layer, token):  lora_moe_down_apply out += B2·acc2         (shared B) */

static void lora_moe_pre(const Lora *lr, int layer, const float *x, float *t1, float *t3) {
    int r = lr->r; int64_t D = lr->D;
    const float *a1 = lr->a1[layer], *a3 = lr->a3[layer];
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < 2*r; k++) {
        const float *ak = (k < r ? a1 + (int64_t)k*D : a3 + (int64_t)(k-r)*D);
        float acc = 0.f;
        for (int64_t i = 0; i < D; i++) acc += ak[i] * x[i];
        (k < r ? t1 : t3)[k < r ? k : k-r] = acc;
    }
}

/* g[0..I) gate rows, g[I..2I) up rows — matches the fused base matmul layout */
static void lora_moe_gu(const Lora *lr, int layer, int eid, const float *t1, const float *t3, float *g) {
    int r = lr->r; int64_t I = lr->moe_I;
    const float *b1 = lr->b1[layer] + (int64_t)eid*I*r;
    const float *b3 = lr->b3[layer] + (int64_t)eid*I*r;
    float s = lr->scale;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < 2*I; i++) {
        const float *br = (i < I ? b1 + i*r : b3 + (i-I)*r);
        const float *t  = (i < I ? t1 : t3);
        float acc = 0.f;
        for (int k = 0; k < r; k++) acc += br[k] * t[k];
        g[i] += s * acc;
    }
}

static void lora_moe_down_acc(const Lora *lr, int layer, int eid, const float *h, float wk, float *acc2) {
    int r = lr->r; int64_t I = lr->moe_I;
    const float *a2 = lr->a2[layer] + (int64_t)eid*r*I;
    float t2[LORA_MAX_R];
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < r; k++) {
        const float *ak = a2 + (int64_t)k*I;
        float acc = 0.f;
        for (int64_t i = 0; i < I; i++) acc += ak[i] * h[i];
        t2[k] = acc;
    }
    float ws = wk * lr->scale;
    for (int k = 0; k < r; k++) acc2[k] += ws * t2[k];
}

static void lora_moe_down_apply(const Lora *lr, int layer, const float *acc2, float *out) {
    int r = lr->r; int64_t D = lr->D;
    const float *b2 = lr->b2[layer];
    #pragma omp parallel for schedule(static)
    for (int64_t d = 0; d < D; d++) {
        const float *br = b2 + d*r;
        float acc = 0.f;
        for (int k = 0; k < r; k++) acc += br[k] * acc2[k];
        out[d] += acc;
    }
}

#endif
