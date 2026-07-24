/* moe_math.h — scalar math + timing helpers shared by the MoE engines
 * (colibri/inkling/laguna/olmoe). Phase 1 of the MoE-runtime refactor
 * (docs/moe-runtime-plan.md): these were byte-identical copy-paste across the
 * per-model engines; this is the first extraction, reconciling the trivial
 * drift (rss_gb's platform assumption; the rmsnorm/rmsnorm_row naming).
 *
 * Each engine is its own translation unit / binary, so the `static` definitions
 * here carry no cross-TU symbol risk; unused ones are tolerated (the build sets
 * -Wno-unused-function). Numerics are unchanged from the originals, so every
 * engine's token-exact oracle stays green.
 *
 * NOT here (yet): softmax — laguna accumulates the denom in double and multiplies
 * by the reciprocal, inkling/olmoe accumulate in float and divide; those are not
 * bit-identical, so softmax stays per-engine until a deliberate reconciliation. */
#ifndef MOE_MATH_H
#define MOE_MATH_H
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <sys/resource.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* Peak RSS in GB. ru_maxrss is KB on Linux/BSD, bytes on macOS — the per-engine
 * copies disagreed on this (laguna assumed KB, inkling/olmoe assumed bytes); this
 * is correct on both. Cosmetic (a reported stat), never on the oracle path. */
static double rss_gb(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
#if defined(__APPLE__)
    return r.ru_maxrss / (1024.0 * 1024.0 * 1024.0);   /* bytes -> GB */
#else
    return r.ru_maxrss / (1024.0 * 1024.0);            /* KB -> GB */
#endif
}

/* bf16 (top 16 bits of an f32) -> f32, exact. */
static float bf16_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static float siluf(float x) { return x / (1.f + expf(-x)); }
static float softplusf(float x) { return x > 20.f ? x : log1pf(expf(x)); }   /* numerically stable */

/* RMSNorm over a length-D row: out = x / rms(x) * w, with the sum-of-squares
 * accumulated in double (the shared contract; both source copies did this). */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ss = 0; for (int i = 0; i < D; i++) ss += (double)x[i] * x[i];
    float inv = 1.f / sqrtf((float)(ss / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * inv * w[i];
}

#endif
