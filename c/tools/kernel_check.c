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
#include <stdint.h>
#include "../moe_quant.h"

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
    printf("%s\n", ok ? "KERNEL-CHECK PASS" : "KERNEL-CHECK FAIL");
    return !ok;
}
