/* moe_sample.h — shared token sampling for the MoE engines.
 * Phase 1 of the MoE-runtime refactor (docs/moe-runtime-plan.md). Oracle-neutral:
 * greedy decode (temp<=0) is argmax and the token-exact oracles never exercise
 * this; it only shapes temperature/top-p generation in serve mode.
 *
 * xorshift64 RNG (g_rng is seeded per-run in the serve loop), plus temperature +
 * top-p nucleus sampling. Byte-identical to the copies laguna/inkling carried. */
#ifndef MOE_SAMPLE_H
#define MOE_SAMPLE_H
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double rng_next(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

typedef struct { float p; int i; } PI;
static int pi_desc(const void *a, const void *b) {
    float d = ((const PI*)b)->p - ((const PI*)a)->p;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

/* temperature + top-p nucleus sampling over n logits; temp<=0 = greedy argmax. */
static int sample_logits(const float *logit, int n, float temp, float top_p) {
    int best = 0; for (int i = 1; i < n; i++) if (logit[i] > logit[best]) best = i;
    if (temp <= 0.f) return best;
    PI *c = malloc((size_t)n * sizeof(PI)); double sum = 0;
    for (int i = 0; i < n; i++) { c[i].p = expf((logit[i] - logit[best]) / temp); c[i].i = i; sum += c[i].p; }
    qsort(c, n, sizeof(PI), pi_desc);
    double cut = (top_p > 0.f && top_p < 1.f) ? top_p * sum : sum, acc = 0; int k = 0;
    while (k < n && acc < cut) acc += c[k++].p;
    double r = rng_next() * acc, run = 0; int pick = c[0].i;
    for (int i = 0; i < k; i++) { run += c[i].p; if (run >= r) { pick = c[i].i; break; } }
    free(c); return pick;
}

#endif
