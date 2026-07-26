/* linattn_check.c — validate the gated delta net against the transformers reference.
 *
 * The linear-attention layer is new machinery with an easy-to-get-wrong recurrence (decay
 * before the memory read, update before the output read), and an engine-level oracle cannot
 * localize a bug in it. So this runs the C implementation over the fixture written by
 * tools/gdn_ref.py and checks three things:
 *
 *   1. the recurrence output on its own, against transformers
 *   2. the full layer output, against transformers
 *      (1 failing is the delta rule; only 2 failing is projections / conv / gated norm)
 *   3. span-schedule invariance: feeding the sequence as 1-token steps, as uneven spans, and
 *      as one shot must give *bit-identical* results, because a chunked prefill will do all
 *      three and the recurrent state plus the conv window have to carry correctly across the
 *      boundary. This is the same failure mode as the sliding-KV ring off-by-N — silent, and
 *      invisible to a single-span test.
 *
 *   python3 tools/gdn_ref.py gdn_ref.npz
 *   cc -O2 -lm tools/linattn_check.c -o linattn-check && ./linattn-check gdn_ref.npz
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../moe_linattn.h"

/* ---- minimal .npz reader: stored (uncompressed) float32/int32 arrays only ---- */
typedef struct { char name[64]; float *f; int32_t i; int is_int; long n; } Arr;
static Arr g_arr[64];
static int g_narr = 0;

static uint32_t rd32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* npz is a zip of .npy members; we only handle STORED entries, which np.savez produces. */
static void load_npz(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);

    for (long i = 0; i + 30 < sz; ) {
        if (rd32(buf + i) != 0x04034b50) { i++; continue; }        /* local file header */
        uint32_t csize = rd32(buf + i + 18);
        int nlen = buf[i + 26] | (buf[i + 27] << 8);
        int elen = buf[i + 28] | (buf[i + 29] << 8);
        char name[128] = {0};
        int keep = nlen < 127 ? nlen : 127;
        memcpy(name, buf + i + 30, keep);
        char *dot = strstr(name, ".npy"); if (dot) *dot = 0;
        unsigned char *npy = buf + i + 30 + nlen + elen;

        /* .npy: magic(6) ver(2) hlen(2) header */
        int hlen = npy[8] | (npy[9] << 8);
        char *hdr = (char *)npy + 10;
        int is_int = strstr(hdr, "<i4") != NULL;
        unsigned char *data = npy + 10 + hlen;
        long nbytes = (long)csize - (10 + hlen);
        long n = nbytes / 4;

        Arr *a = &g_arr[g_narr++];
        snprintf(a->name, sizeof(a->name), "%s", name);
        a->is_int = is_int; a->n = n;
        if (is_int) { memcpy(&a->i, data, 4); a->f = NULL; }
        else { a->f = malloc((size_t)n * 4); memcpy(a->f, data, (size_t)n * 4); }
        i += 30 + nlen + elen + csize;
    }
}

static const float *A(const char *name) {
    for (int i = 0; i < g_narr; i++) if (!strcmp(g_arr[i].name, name) && g_arr[i].f) return g_arr[i].f;
    fprintf(stderr, "missing array %s\n", name); exit(1);
}
static int I(const char *name) {
    for (int i = 0; i < g_narr; i++) if (!strcmp(g_arr[i].name, name) && g_arr[i].is_int) return g_arr[i].i;
    fprintf(stderr, "missing scalar %s\n", name); exit(1);
}

/* y[o] = sum_i x[i] * W[o][i] — HF Linear, row-major [out, in] */
static void mm(float *y, const float *x, const float *W, int in, int out) {
    for (int o = 0; o < out; o++) {
        double s = 0;
        const float *w = W + (size_t)o * in;
        for (int i = 0; i < in; i++) s += (double)x[i] * w[i];
        y[o] = (float)s;
    }
}

static int report(const char *what, const float *got, const float *ref, long n, double tol) {
    double num = 0, den = 0, md = 0;
    for (long i = 0; i < n; i++) {
        double d = (double)got[i] - ref[i];
        num += d * d; den += (double)ref[i] * ref[i];
        if (fabs(d) > md) md = fabs(d);
    }
    double l2 = sqrt(num / (den + 1e-30));
    int ok = l2 < tol;
    printf("  %-34s max|abs|=%.3e  L2rel=%.3e  %s\n", what, md, l2, ok ? "OK" : "FAIL");
    return ok;
}

/* ---- geometry, filled from the fixture ---- */
static int hidden, seq, nk, nv, kd, vd, ck, key_dim, value_dim, conv_dim;
static float g_eps;

/* Run the whole layer over the sequence using the given span schedule. */
static void run_layer(const int *spans, int nspans, float *core, float *y, int mode) {
    const float *x = A("x"), *w_qkv = A("w_qkv"), *w_z = A("w_z"), *w_b = A("w_b"), *w_a = A("w_a");
    const float *conv_w = A("conv_w"), *conv_b = A("conv_b"), *dt_bias = A("dt_bias");
    const float *A_log = A("A_log"), *norm_w = A("norm_w"), *w_out = A("w_out");

    LinAttnState st;
    st.state = calloc((size_t)nv * kd * vd, sizeof(float));
    st.conv  = calloc((size_t)conv_dim * ck, sizeof(float));

    int maxS = 0;
    for (int i = 0; i < nspans; i++) if (spans[i] > maxS) maxS = spans[i];
    float *mixed = malloc((size_t)maxS * conv_dim * 4);
    float *zbuf  = malloc((size_t)maxS * value_dim * 4);
    float *gbuf  = malloc((size_t)maxS * nv * 4);
    float *bbuf  = malloc((size_t)maxS * nv * 4);
    float *qkv_a = malloc((size_t)maxS * nv * 4);

    int t0 = 0;
    for (int s = 0; s < nspans; s++) {
        int S = spans[s];
        for (int t = 0; t < S; t++) {
            const float *xt = x + (size_t)(t0 + t) * hidden;
            mm(mixed + (size_t)t * conv_dim, xt, w_qkv, hidden, conv_dim);
            mm(zbuf + (size_t)t * value_dim, xt, w_z, hidden, value_dim);
            mm(bbuf + (size_t)t * nv, xt, w_b, hidden, nv);
            mm(qkv_a + (size_t)t * nv, xt, w_a, hidden, nv);
        }
        linattn_conv_span(st.conv, mixed, conv_w, conv_b, mixed, S, conv_dim, ck);

        /* beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias) */
        for (int t = 0; t < S; t++)
            for (int h = 0; h < nv; h++) {
                float b = bbuf[(size_t)t * nv + h];
                bbuf[(size_t)t * nv + h] = 1.f / (1.f + expf(-b));
                gbuf[(size_t)t * nv + h] =
                    -expf(A_log[h]) * linattn_softplus(qkv_a[(size_t)t * nv + h] + dt_bias[h]);
            }

        /* conv output is [q | k | v] concatenated along the channel axis */
        float *core_span = core + (size_t)t0 * value_dim;
        {
            /* q/k/v are interleaved per token inside `mixed`, so hand linattn_scan strided
             * views by copying into contiguous per-quantity buffers. */
            float *qb = malloc((size_t)S * key_dim * 4);
            float *kb = malloc((size_t)S * key_dim * 4);
            float *vb = malloc((size_t)S * value_dim * 4);
            for (int t = 0; t < S; t++) {
                const float *m = mixed + (size_t)t * conv_dim;
                memcpy(qb + (size_t)t * key_dim, m, (size_t)key_dim * 4);
                memcpy(kb + (size_t)t * key_dim, m + key_dim, (size_t)key_dim * 4);
                memcpy(vb + (size_t)t * value_dim, m + 2 * key_dim, (size_t)value_dim * 4);
            }
            linattn_scan_mode(&st, qb, kb, vb, gbuf, bbuf, core_span, S, nk, nv, kd, vd, mode);
            free(qb); free(kb); free(vb);
        }

        for (int t = 0; t < S; t++) {
            float tmp[4096];
            memcpy(tmp, core_span + (size_t)t * value_dim, (size_t)value_dim * 4);
            for (int h = 0; h < nv; h++)
                linattn_gated_norm(tmp + (size_t)h * vd, zbuf + (size_t)t * value_dim + (size_t)h * vd,
                                   norm_w, vd, g_eps);
            mm(y + (size_t)(t0 + t) * hidden, tmp, w_out, value_dim, hidden);
        }
        t0 += S;
    }
    free(st.state); free(st.conv);
    free(mixed); free(zbuf); free(gbuf); free(bbuf); free(qkv_a);
}

/* ---- throughput at Qwen3.5-35B-A3B's real geometry ----
 *
 * The correctness fixture is deliberately tiny; it says nothing about cost. This runs the
 * recurrence at the shipping shapes (32 value heads / 16 key heads, 128x128 state, 30 linear
 * layers) to answer the question that decides whether a chunked kernel is needed: how many
 * ms per token does the whole linear-attention stack cost?
 *
 * Honesty notes, learned the hard way from the int4 GEMM bench in kernel_check.c:
 *  - It rotates through 30 *separate* states (60 MB, ~L3-sized) rather than hammering one
 *    hot 2 MB buffer, because the real model walks a different layer's state every step.
 *    Still optimistic: the real model also streams ~500 MB of expert weights per token,
 *    which evicts all of it. Treat these as a floor on cost, not a prediction.
 *  - min-of-5 timed batches, because a single sample here inverted a ranking once. */
static double mono(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define BN_LAYERS 30
#define BN_KH 16
#define BN_VH 32
#define BN_D  128

static double bench_mode(int S, int iters, int mode, LinAttnState *sts,
                         const float *q, const float *k, const float *v,
                         const float *g, const float *b, float *out) {
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        double t = mono();
        for (int it = 0; it < iters; it++)
            for (int l = 0; l < BN_LAYERS; l++)
                linattn_scan_mode(&sts[l], q, k, v, g, b, out, S, BN_KH, BN_VH, BN_D, BN_D, mode);
        t = mono() - t;
        if (t < best) best = t;
    }
    return best / (double)iters;      /* seconds per token-group of S, all 30 layers */
}

static void bench_real_geometry(void) {
    int kdim = BN_KH * BN_D, vdim = BN_VH * BN_D;
    int Smax = 128;
    LinAttnState *sts = malloc(sizeof(LinAttnState) * BN_LAYERS);
    for (int l = 0; l < BN_LAYERS; l++) {
        sts[l].state = calloc((size_t)BN_VH * BN_D * BN_D, sizeof(float));
        sts[l].conv = NULL;                       /* conv is exercised by the fixture, not here */
    }
    float *q = malloc((size_t)Smax * kdim * 4), *k = malloc((size_t)Smax * kdim * 4);
    float *v = malloc((size_t)Smax * vdim * 4), *out = malloc((size_t)Smax * vdim * 4);
    float *g = malloc((size_t)Smax * BN_VH * 4), *b = malloc((size_t)Smax * BN_VH * 4);
    unsigned r = 12345;
#define RND (((r = r * 1664525u + 1013904223u) >> 8) * (1.f / 8388608.f) - 1.f)
    for (long i = 0; i < (long)Smax * kdim; i++) { q[i] = RND; k[i] = RND; }
    for (long i = 0; i < (long)Smax * vdim; i++) v[i] = RND;
    for (long i = 0; i < (long)Smax * BN_VH; i++) { g[i] = -0.05f * (1.f + RND); b[i] = RND; }

    printf("\nreal geometry: %d linear layers, %d v-heads / %d k-heads, %dx%d state "
           "(%.1f MB/layer, %.0f MB total)\n",
           BN_LAYERS, BN_VH, BN_KH, BN_D, BN_D,
           (double)BN_VH * BN_D * BN_D * 4 / 1e6,
           (double)BN_LAYERS * BN_VH * BN_D * BN_D * 4 / 1e6);
#ifdef _OPENMP
    printf("  (head-parallel via OpenMP)\n");
#endif
    struct { const char *name; int mode; } modes[] = {
        { "naive (3 sweeps)", LINATTN_NAIVE }, { "fused (2 sweeps)", LINATTN_FUSED } };
    for (int m = 0; m < 2; m++) {
        double d1 = bench_mode(1, 40, modes[m].mode, sts, q, k, v, g, b, out);
        double d128 = bench_mode(128, 3, modes[m].mode, sts, q, k, v, g, b, out);
        printf("  %-18s decode %6.2f ms/tok    prefill(S=128) %6.3f ms/tok  (%.0f tok/s)\n",
               modes[m].name, d1 * 1e3, d128 * 1e3 / 128.0, 128.0 / d128);
    }
    for (int l = 0; l < BN_LAYERS; l++) free(sts[l].state);
    free(sts); free(q); free(k); free(v); free(out); free(g); free(b);
#undef RND
}

int main(int argc, char **argv) {
    load_npz(argc > 1 ? argv[1] : "gdn_ref.npz");
    hidden = I("hidden"); seq = I("seq");
    nk = I("n_k_heads"); nv = I("n_v_heads");
    kd = I("k_dim"); vd = I("v_dim"); ck = I("conv_k");
    for (int i = 0; i < g_narr; i++) if (!strcmp(g_arr[i].name, "eps")) g_eps = g_arr[i].f[0];
    if (g_eps == 0) g_eps = 1e-6f;

    key_dim = kd * nk; value_dim = vd * nv; conv_dim = key_dim * 2 + value_dim;
    printf("gated delta net [hidden=%d seq=%d k_heads=%d v_heads=%d kd=%d vd=%d conv_k=%d]\n",
           hidden, seq, nk, nv, kd, vd, ck);
    printf("  per-sequence state: %.1f KB (constant in context length)\n",
           linattn_state_bytes(nv, kd, vd, conv_dim, ck) / 1024.0);

    const float *core_ref = A("core_ref"), *y_ref = A("y_ref");
    long ncore = (long)seq * value_dim, ny = (long)seq * hidden;
    int ok = 1;

    /* Both kernels are checked against transformers, not against each other: the fused form
     * has a different summation order, so "matches the naive form" is the wrong bar. */
    int *ones = malloc((size_t)seq * sizeof(int));
    for (int i = 0; i < seq; i++) ones[i] = 1;
    float *core_n = malloc((size_t)ncore * 4), *y_n = malloc((size_t)ny * 4);
    run_layer(ones, seq, core_n, y_n, LINATTN_NAIVE);
    printf("naive (3-sweep, literal reference form):\n");
    ok &= report("recurrence (pre gated-norm)", core_n, core_ref, ncore, 1e-5);
    ok &= report("full layer output", y_n, y_ref, ny, 1e-5);

    /* schedule 1: every token its own span — the decode path */
    float *core_a = malloc((size_t)ncore * 4), *y_a = malloc((size_t)ny * 4);
    run_layer(ones, seq, core_a, y_a, LINATTN_FUSED);
    printf("fused (2-sweep, the default):\n");
    ok &= report("recurrence (pre gated-norm)", core_a, core_ref, ncore, 1e-5);
    ok &= report("full layer output", y_a, y_ref, ny, 1e-5);

    /* schedule 2: one shot — the single-span prefill path */
    int one[1] = { seq };
    float *core_b = malloc((size_t)ncore * 4), *y_b = malloc((size_t)ny * 4);
    run_layer(one, 1, core_b, y_b, LINATTN_FUSED);

    /* schedule 3: uneven spans — the chunked prefill path, where state + conv window must
     * carry across the boundary. Uneven on purpose: an equal split can hide an error that
     * happens to be symmetric, and 5 < conv_k+1 exercises a window that is still filling. */
    int uneven[3] = { 5, 1, seq - 6 };
    float *core_c = malloc((size_t)ncore * 4), *y_c = malloc((size_t)ny * 4);
    run_layer(uneven, 3, core_c, y_c, LINATTN_FUSED);

    /* These must be bit-identical, not merely close: same ops, same order, only the span
     * boundaries differ. Any drift means state or window is being rebuilt, not carried. */
    long bad_b = 0, bad_c = 0;
    for (long i = 0; i < ny; i++) {
        if (memcmp(&y_b[i], &y_a[i], 4)) bad_b++;
        if (memcmp(&y_c[i], &y_a[i], 4)) bad_c++;
    }
    printf("  %-34s %ld/%ld differing floats  %s\n", "one-shot span == 1-tok steps",
           bad_b, ny, bad_b ? "FAIL" : "OK");
    printf("  %-34s %ld/%ld differing floats  %s\n", "spans {5,1,rest} == 1-tok steps",
           bad_c, ny, bad_c ? "FAIL" : "OK");
    ok &= (bad_b == 0) && (bad_c == 0);

    bench_real_geometry();

    printf("%s\n", ok ? "LINATTN-CHECK PASS" : "LINATTN-CHECK FAIL");
    return !ok;
}
