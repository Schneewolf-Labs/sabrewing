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

/* Exhaustive nucleus sampling: sort the whole vocabulary. Correct for any
 * distribution and used as the fallback (and the reference) for the fast path. */
static int sample_logits_full(const float *logit, int n, float temp, double cut, int best) {
    PI *c = malloc((size_t)n * sizeof(PI));
    for (int i = 0; i < n; i++) { c[i].p = expf((logit[i] - logit[best]) / temp); c[i].i = i; }
    qsort(c, n, sizeof(PI), pi_desc);
    double acc = 0; int k = 0;
    while (k < n && acc < cut) acc += c[k++].p;
    double r = rng_next() * acc, run = 0; int pick = c[0].i;
    for (int i = 0; i < k; i++) { run += c[i].p; if (run >= r) { pick = c[i].i; break; } }
    free(c); return pick;
}

/* temperature + top-p nucleus sampling over n logits; temp<=0 = greedy argmax.
 *
 * The nucleus is a PREFIX of the descending order, so sorting the whole vocabulary
 * is wasted work: take a head of candidates by partial selection, and only if that
 * head fails to cover top_p's mass widen it (finally deferring to the full sort, so
 * flat distributions stay exact). Laguna's vocabulary is 100k and serve mode samples
 * once per stream per step — the full qsort measured 13% of aggregate throughput at
 * 8 concurrent slots (8x96 tokens: 22.8 tok/s greedy vs 19.7 at temp 0.7).
 *
 * Same draw as the exhaustive path: the total mass is still summed over every logit
 * in index order, the head accumulates in descending order, and exactly one
 * rng_next() is consumed. (Exact ties order by lower index here vs qsort's arbitrary
 * order there — reachable only when two logits are bit-equal AND the draw lands in
 * their span.) */
#define SAMPLE_HEAD     64      /* first head size tried */
#define SAMPLE_HEAD_MAX 4096    /* widen to here, then sort the whole vocabulary */
static int sample_logits(const float *logit, int n, float temp, float top_p) {
    int best = 0; for (int i = 1; i < n; i++) if (logit[i] > logit[best]) best = i;
    if (temp <= 0.f) return best;
    double sum = 0;                              /* full mass, index order (as above) */
    for (int i = 0; i < n; i++) sum += expf((logit[i] - logit[best]) / temp);
    double cut = (top_p > 0.f && top_p < 1.f) ? top_p * sum : sum;

    float pv[SAMPLE_HEAD_MAX], tv[SAMPLE_HEAD_MAX];   /* head: probs, logits (descending) */
    int idx[SAMPLE_HEAD_MAX];                         /* ~48 KB of stack, no statics: this is
                                                       * called per stream per step, keep it
                                                       * re-entrant */
    for (int K = SAMPLE_HEAD; K <= SAMPLE_HEAD_MAX && K < n; K *= 8) {
        int k = 0;
        for (int i = 0; i < n; i++) {            /* selection: O(1) reject for most i */
            if (k == K && logit[i] <= tv[k - 1]) continue;
            int j = k < K ? k++ : k - 1;
            while (j > 0 && tv[j - 1] < logit[i]) { tv[j] = tv[j - 1]; idx[j] = idx[j - 1]; j--; }
            tv[j] = logit[i]; idx[j] = i;
        }
        double acc = 0; int m = 0;
        while (m < k && acc < cut) { pv[m] = expf((tv[m] - logit[best]) / temp); acc += pv[m]; m++; }
        if (acc < cut) continue;                 /* head too small — widen */
        double r = rng_next() * acc, run = 0;
        for (int a = 0; a < m; a++) { run += pv[a]; if (run >= r) return idx[a]; }
        return idx[0];
    }
    return sample_logits_full(logit, n, temp, cut, best);
}

#endif
