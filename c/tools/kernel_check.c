/* kernel_check.c — validation harness for the shared quant kernels (moe_quant.h)
 * that the token-exact oracle can't reach (it runs f32 only). Generates random
 * weights, quantizes them to the int4 container, and checks each kernel path
 * against a double-precision dequant reference.
 *
 *   cc -O3 -march=native -fopenmp tools/kernel_check.c -o kernel_check && ./kernel_check
 *
 * Expectations: MOE_Q4_F32 exact=1 is bit-identical to the reference (both are the
 * double-accumulate dequant dot); MOE_Q4_F32 exact=0 (AVX-512 FMA) agrees to
 * float-noise (~1e-6 rel); MOE_Q4_IDOT (int8 activation quant) is deliberately
 * lossy but bounded (~1e-2 rel — activation quantization, not a bug). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "../moe_quant.h"
#include "../moe_matmul.h"
#include "../moe_sample.h"   /* sample_logits: head path vs exhaustive sort */
#include "../moe_attn.h"     /* sdpa_head: SIMD generation path vs the exact contract */

/* round f32 -> bf16 (round-to-nearest-even), as bf16 weights are stored */
static uint16_t f2bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

/* quantize one row to int4 (nibble-8, per-row scale) — matches the converter's
 * quant_int4_rows (convert_laguna_int4.py). */
static void quant_row_i4(const float *w, int I, uint8_t *packed, float *scale) {
    float mx = 0; for (int i = 0; i < I; i++) { float a = fabsf(w[i]); if (a > mx) mx = a; }
    float s = mx / 7.f; if (s < 1e-12f) s = 1e-12f; *scale = s;
    for (int c = 0; c < I / 2; c++) {
        int lo = (int)lrintf(w[2 * c] / s);     if (lo < -8) lo = -8; if (lo > 7) lo = 7;
        int hi = (int)lrintf(w[2 * c + 1] / s); if (hi < -8) hi = -8; if (hi > 7) hi = 7;
        packed[c] = (uint8_t)((lo + 8) | ((hi + 8) << 4));
    }
}

/* L2-relative error ||y-ref|| / ||ref|| — robust to near-zero outputs (unlike a
 * per-element max-rel, which explodes when an output happens to be ~0). This is
 * the right measure of whole-vector kernel fidelity. */
static int report(const char *name, const float *y, const float *ref, int O, double tol) {
    double num = 0, den = 0, md = 0; for (int o = 0; o < O; o++) {
        double d = (double)y[o] - ref[o]; num += d * d; den += (double)ref[o] * ref[o];
        if (fabs(d) > md) md = fabs(d);
    }
    double l2 = sqrt(num / (den + 1e-30));
    int ok = l2 < tol;
    printf("  %-22s max|abs|=%.3e  L2rel=%.3e  %s\n", name, md, l2, ok ? "OK" : "FAIL");
    return ok;
}

int main(void) {
    srand(1234);
    int I = 2048, O = 1024;
    float *W = malloc((size_t)O * I * 4), *x = malloc((size_t)I * 4);
    uint8_t *packed = malloc((size_t)O * (I / 2)); float *scale = malloc((size_t)O * 4);
    float *yref = malloc((size_t)O * 4), *yf = malloc((size_t)O * 4);
    float *yi = malloc((size_t)O * 4), *ye = malloc((size_t)O * 4);
    for (int i = 0; i < I; i++) x[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    for (int o = 0; o < O; o++) {
        for (int i = 0; i < I; i++) W[(size_t)o * I + i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
        quant_row_i4(W + (size_t)o * I, I, packed + (size_t)o * (I / 2), scale + o);
    }
    /* double-precision reference over the DEQUANTIZED weights */
    for (int o = 0; o < O; o++) {
        const uint8_t *p = packed + (size_t)o * (I / 2);
        double s = 0;
        for (int c = 0; c < I / 2; c++) { uint8_t b = p[c]; s += (double)x[2*c]*((int)(b&0xF)-8) + (double)x[2*c+1]*((int)(b>>4)-8); }
        yref[o] = (float)(s * scale[o]);
    }
    matmul_q4_k(ye, x, packed, scale, I, O, MOE_Q4_F32, 1);   /* exact double */
    matmul_q4_k(yf, x, packed, scale, I, O, MOE_Q4_F32, 0);   /* AVX-512 f32 FMA (laguna) */
    matmul_q4_k(yi, x, packed, scale, I, O, MOE_Q4_IDOT, 0);  /* int8-act VNNI (inkling) */
    printf("int4 matmul_q4_k [%d->%d] vs double dequant reference:\n", I, O);
    int ok = 1;
    ok &= report("MOE_Q4_F32 exact",   ye, yref, O, 1e-9);   /* bit-identical: double vs double */
    ok &= report("MOE_Q4_F32 simd",    yf, yref, O, 1e-3);   /* AVX-512 FMA float noise */
    ok &= report("MOE_Q4_IDOT (int8a)", yi, yref, O, 3e-2);  /* int8 activation quant loss */
    int bitexact = 1;
    for (int o = 0; o < O; o++) if (ye[o] != yref[o]) { bitexact = 0; break; }
    if (!bitexact) { printf("  FAIL: exact path not bit-identical to reference\n"); ok = 0; }

    /* --- bf16 matmul_bf16_k: same random weights rounded to bf16 --- */
    uint16_t *Wb = malloc((size_t)O * I * sizeof(uint16_t));
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) Wb[(size_t)o * I + i] = f2bf16(W[(size_t)o * I + i]);
    float *br = malloc((size_t)O * 4), *b0 = malloc((size_t)O * 4), *b1 = malloc((size_t)O * 4);
    for (int o = 0; o < O; o++) {   /* reference: x(f32) . dequant(bf16 W), double accum */
        const uint16_t *w = Wb + (size_t)o * I;
        double s = 0; for (int i = 0; i < I; i++) { uint32_t u = (uint32_t)w[i] << 16; float wf; memcpy(&wf, &u, 4); s += (double)x[i] * wf; }
        br[o] = (float)s;
    }
    matmul_bf16_k(b0, x, Wb, 1, I, O, 0, 0);   /* round_x=0: activations f32 (laguna) */
    matmul_bf16_k(b1, x, Wb, 1, I, O, 1, 0);   /* round_x=1: activations bf16 (inkling) */
    printf("bf16 matmul_bf16_k [%d->%d] vs double dequant reference:\n", I, O);
    ok &= report("round_x=0 (f32 act)",  b0, br, O, 1e-3);   /* weight-only bf16, f32 acts */
    ok &= report("round_x=1 (bf16 act)", b1, br, O, 1e-2);   /* + activation bf16 rounding */

    /* --- int8 matmul_q8_k: per-row int8 quant of the same weights --- */
    int8_t *q8 = malloc((size_t)O * I); float *q8s = malloc((size_t)O * 4);
    for (int o = 0; o < O; o++) {
        const float *w = W + (size_t)o * I;
        float mx = 0; for (int i = 0; i < I; i++) { float a = fabsf(w[i]); if (a > mx) mx = a; }
        float s = mx / 127.f; if (s < 1e-12f) s = 1e-12f; q8s[o] = s;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(w[i] / s); q8[(size_t)o*I+i] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v); }
    }
    float *q8r = malloc((size_t)O * 4), *q8f = malloc((size_t)O * 4), *q8i = malloc((size_t)O * 4);
    for (int o = 0; o < O; o++) {   /* reference: x(f32) . dequant(int8), double accum */
        const int8_t *w = q8 + (size_t)o * I;
        double s = 0; for (int i = 0; i < I; i++) s += (double)x[i] * w[i];
        q8r[o] = (float)(s * q8s[o]);
    }
    matmul_q8_k(q8f, x, q8, q8s, I, O, MOE_Q8_F32, 0);   /* f32 act (laguna) */
    matmul_q8_k(q8i, x, q8, q8s, I, O, MOE_Q8_IDOT, 0);  /* int8 act (inkling) */
    printf("int8 matmul_q8_k [%d->%d] vs double dequant reference:\n", I, O);
    ok &= report("MOE_Q8_F32 simd",    q8f, q8r, O, 1e-3);
    ok &= report("MOE_Q8_IDOT (int8a)", q8i, q8r, O, 3e-2);

    /* ---- attention: the SIMD generation path vs the double-accumulate contract ----
     * laguna's oracle runs MOE_QK_DBL, so the fast path used for generation needs its
     * own check: same scores, same weighted sum, to float noise. Long windows are the
     * interesting case (more accumulation), so sweep the window length. */
    {
        int hd = 128, kvs_stride = 8 * 128;
        int wins[] = {1, 17, 512, 2187};
        printf("attention sdpa_head [hd=%d] MOE_QK_SIMD vs MOE_QK_DBL:\n", hd);
        for (size_t wi = 0; wi < sizeof(wins) / sizeof(wins[0]); wi++) {
            int npos = wins[wi];
            float *q = malloc((size_t)hd * 4);
            float *K = malloc((size_t)npos * kvs_stride * 4), *V = malloc((size_t)npos * kvs_stride * 4);
            float *o1 = malloc((size_t)hd * 4), *o2 = malloc((size_t)hd * 4);
            float *sc = malloc((size_t)npos * 4);
            for (int i = 0; i < hd; i++) q[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
            for (int64_t i = 0; i < (int64_t)npos * kvs_stride; i++) {
                K[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
                V[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
            }
            sdpa_head(q, K, V, kvs_stride, hd, 0, npos - 1, 1.f / sqrtf((float)hd), 1.f,
                      NULL, 0, MOE_QK_DBL, o1, sc, 0);
            sdpa_head(q, K, V, kvs_stride, hd, 0, npos - 1, 1.f / sqrtf((float)hd), 1.f,
                      NULL, 0, MOE_QK_SIMD, o2, sc, 0);
            char name[64]; snprintf(name, sizeof(name), "window %-5d simd", npos);
            ok &= report(name, o2, o1, hd, 1e-5);
            /* online softmax: one pass, rescaled accumulator — same softmax, different
             * rounding, so it gets its own (slightly looser) tolerance */
            float *o3 = malloc((size_t)hd * 4);
            sdpa_head(q, K, V, kvs_stride, hd, 0, npos - 1, 1.f / sqrtf((float)hd), 1.f,
                      NULL, 0, MOE_QK_ONLINE, o3, sc, 0);
            snprintf(name, sizeof(name), "window %-5d online", npos);
            ok &= report(name, o3, o1, hd, 1e-4);
            free(o3);
            free(q); free(K); free(V); free(o1); free(o2); free(sc);
        }
    }

    /* ---- int8 KV cache: quantization error of the whole attention output ----
     * Storing K/V at 1.03 bytes/value instead of 4 is lossy, so bound the damage: the
     * attention output from an int8 cache vs the same data in f32 with the exact
     * double-accumulate contract. Per-row symmetric int8 over hd=128 should land near
     * 1e-3 relative; the real gate is model perplexity, this just catches a broken
     * scale or a layout mistake. */
    {
        int hd = 128, nkvh = 8, stride = nkvh * hd;
        int wins[] = {17, 512, 2187};
        printf("int8 KV sdpa_head_q8 [hd=%d] vs f32 cache + MOE_QK_DBL:\n", hd);
        for (size_t wi = 0; wi < sizeof(wins) / sizeof(wins[0]); wi++) {
            int npos = wins[wi];
            float *q = malloc((size_t)hd * 4);
            float *K = malloc((size_t)npos * stride * 4), *V = malloc((size_t)npos * stride * 4);
            int8_t *Kq = malloc((size_t)npos * stride), *Vq = malloc((size_t)npos * stride);
            float *Ks = malloc((size_t)npos * nkvh * 4), *Vs = malloc((size_t)npos * nkvh * 4);
            float *o1 = malloc((size_t)hd * 4), *o2 = malloc((size_t)hd * 4), *sc = malloc((size_t)npos * 4);
            for (int i = 0; i < hd; i++) q[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
            for (int64_t i = 0; i < (int64_t)npos * stride; i++) {
                K[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
                V[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
            }
            for (int t = 0; t < npos; t++) for (int h = 0; h < nkvh; h++) {  /* quantize as the engine does */
                int64_t off = (int64_t)t * nkvh + h;
                kv_quant_row(K + (int64_t)t * stride + h * hd, hd, Kq + off * hd, Ks + off);
                kv_quant_row(V + (int64_t)t * stride + h * hd, hd, Vq + off * hd, Vs + off);
            }
            sdpa_head(q, K, V, stride, hd, 0, npos - 1, 1.f / sqrtf((float)hd), 1.f,
                      NULL, 0, MOE_QK_DBL, o1, sc, 0);
            sdpa_head_q8(q, Kq, Ks, Vq, Vs, stride, nkvh, hd, 0, npos - 1,
                         1.f / sqrtf((float)hd), 1.f, o2, sc, 0);
            char name[64]; snprintf(name, sizeof(name), "window %-5d int8 KV", npos);
            ok &= report(name, o2, o1, hd, 2e-2);
            free(q); free(K); free(V); free(Kq); free(Vq); free(Ks); free(Vs); free(o1); free(o2); free(sc);
        }
    }

    /* ---- sampler: the top-k head path must pick EXACTLY what the full sort picks ----
     * Same RNG state in, same token out, across distribution shapes: peaked (the head
     * covers top_p immediately), realistic, near-uniform (forces the head to widen),
     * and dead-flat (forces the exhaustive fallback). Also checks top_p=1 (the whole
     * distribution is the nucleus) and that ties don't crash the selection. */
    {
        int V = 100352, trials = 60, bad = 0, nflat = 0;
        float *lg = malloc((size_t)V * 4);
        const float spread[] = {8.f, 2.f, 0.35f, 0.0f};    /* logit scale: peaked -> flat */
        const float tps[] = {0.95f, 0.5f, 1.0f};
        for (int t = 0; t < trials; t++) {
            float sp = spread[t % 4], tp = tps[(t / 4) % 3], temp = 0.4f + 0.3f * (t % 3);
            for (int i = 0; i < V; i++)
                lg[i] = sp * ((rand() / (float)RAND_MAX) * 2.f - 1.f);
            if (sp == 0.f) { nflat++; for (int i = 0; i < V; i++) lg[i] = 1.25f; }  /* all tied */
            /* the two paths must consume the same RNG state, so replay it */
            uint64_t saved = g_rng;
            int a = sample_logits(lg, V, temp, tp);
            g_rng = saved;
            int best = 0; for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
            double sum = 0;
            for (int i = 0; i < V; i++) sum += expf((lg[i] - lg[best]) / temp);
            double cut = (tp > 0.f && tp < 1.f) ? tp * sum : sum;
            int b = sample_logits_full(lg, V, temp, cut, best);
            if (a != b && !(sp == 0.f)) bad++;   /* flat = every logit tied, order is arbitrary */
        }
        printf("sampler sample_logits [V=%d, %d trials incl. %d flat] head-path vs full sort:\n",
               V, trials, nflat);
        printf("  %-22s %d/%d picks differ  %s\n", "top-k head", bad, trials - nflat,
               bad ? "FAIL" : "OK");
        if (bad) ok = 0;
        free(lg);
    }

    /* ---- int4 expert GEMM: which activation contract wins at which group size? ----
     * Group-by-expert changed this question. The f32 path dequantizes a weight row
     * ONCE per call and reuses it across the group's rows (dequant costs several times
     * the multiply), while MOE_Q4_IDOT pays a horizontal reduction per 32-element
     * block but skips dequant entirely. So the answer depends on rows-per-group, and
     * guessing it from end-to-end runs on a 57 GB model produced two wrong claims —
     * hence this direct measurement. Timing only; it does not gate PASS. */
    {
        struct { int I, O; const char *what; } shapes[] = {
            {3072, 2048, "gate_up (D->2I)"}, {1024, 3072, "down (I->D)"},
        };
        int rows[] = {1, 2, 4, 8, 16};
        printf("int4 expert GEMM throughput (GMAC/s), f32 vs idot by rows-per-group:\n");
        for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
            int I = shapes[si].I, O = shapes[si].O;
            uint8_t *pk = malloc((size_t)O * (I / 2));
            float *scl = malloc((size_t)O * 4);
            for (int64_t i = 0; i < (int64_t)O * (I / 2); i++) pk[i] = rand() & 0xFF;
            for (int o = 0; o < O; o++) scl[o] = 0.01f + (rand() / (float)RAND_MAX) * 0.03f;
            printf("  %-16s", shapes[si].what);
            for (size_t ri = 0; ri < sizeof(rows) / sizeof(rows[0]); ri++) {
                int S = rows[ri];
                float *xb = malloc((size_t)S * I * 4), *yb = malloc((size_t)S * O * 4);
                for (int64_t i = 0; i < (int64_t)S * I; i++) xb[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
                /* MIN of several timed batches, not one batch: a single measurement
                 * of this kernel swings ~2x with clock/thread state, which is enough
                 * to invert the f32-vs-idot ranking run to run. The minimum is the
                 * least-contended sample, so it is the stable comparator. */
                double mac = (double)S * I * O, best[2] = {0, 0};
                for (int mode = 0; mode < 2; mode++) {
                    int m = mode ? MOE_Q4_IDOT : MOE_Q4_F32;
                    matmul_q4_kb(yb, xb, pk, scl, S, I, O, m, 0);          /* warm */
                    int reps = (int)(4e9 / mac); if (reps < 5) reps = 5;
                    double tbest = 1e30;
                    for (int trial = 0; trial < 5; trial++) {
                        double t0 = now_s();
                        for (int r = 0; r < reps; r++) matmul_q4_kb(yb, xb, pk, scl, S, I, O, m, 0);
                        double dt = now_s() - t0;
                        if (dt < tbest) tbest = dt;
                    }
                    best[mode] = mac * reps / tbest / 1e9;
                }
                printf("  S=%-2d f32 %5.1f idot %5.1f", S, best[0], best[1]);
                free(xb); free(yb);
            }
            printf("\n");
            free(pk); free(scl);
        }
    }

    printf("%s\n", ok ? "KERNEL-CHECK PASS" : "KERNEL-CHECK FAIL");
    return !ok;
}
