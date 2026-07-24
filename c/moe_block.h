/* moe_block.h — the shared MoE routing+combine algorithm (moe_arch.h descriptor
 * + engine hooks). The template method: route -> activation -> top-k selection
 * (correction-bias for selection, unbiased score for the combine weight) ->
 * optional renormalize -> routed sum * scale -> optional shared expert.
 *
 * The expert/shared/router matmuls (which touch each engine's own weight layout)
 * are hooks; everything here is layout-agnostic. Op order matches the hand-written
 * per-engine loops so the token-exact oracles stay green. */
#ifndef MOE_BLOCK_H
#define MOE_BLOCK_H
#include <stdlib.h>
#include <math.h>
#include "moe_arch.h"
#include "moe_util.h"   /* falloc */
#include "moe_math.h"   /* softmax_row */

/* out[D] += MoE(x[D]) for one token. */
static void moe_block(const MoeDesc *d, const MoeHooks *h, const float *x, float *out, int D) {
    int E = d->n_experts, K = d->topk;
    float *rl = falloc(E), *sc = falloc(E), *w = falloc(K);
    int *sel = malloc((size_t)K * sizeof(int));
    float *scr = falloc(D), *acc = falloc(D);

    h->router(h->ctx, x, rl);
    if (d->route_act == MOE_ROUTE_SOFTMAX) { softmax_row(rl, E); for (int e = 0; e < E; e++) sc[e] = rl[e]; }
    else for (int e = 0; e < E; e++) sc[e] = 1.f / (1.f + expf(-rl[e]));           /* sigmoid loss-free */

    /* top-K by (score + correction bias); combine weight is the UNBIASED score */
    moe_topk(sc, h->corr_bias, d->use_corr_bias, E, K, sel);
    for (int a = 0; a < K; a++) w[a] = sc[sel[a]];
    if (d->norm_topk) { float sm = 0; for (int a = 0; a < K; a++) sm += w[a]; for (int a = 0; a < K; a++) w[a] /= sm; }

    for (int i = 0; i < D; i++) acc[i] = 0;
    for (int a = 0; a < K; a++) {
        h->expert(h->ctx, sel[a], x, scr);
        for (int i = 0; i < D; i++) acc[i] += w[a] * scr[i];
    }
    for (int i = 0; i < D; i++) acc[i] *= d->route_scale;

    if (d->shared_mode != MOE_SHARED_NONE) {
        h->shared(h->ctx, x, scr);
        /* UNSCALED (laguna): routed*scale + shared, added together into the residual.
         * SCALED/GATED will be refined when glm/inkling migrate — only UNSCALED and
         * NONE are wired and oracle-verified today. */
        for (int i = 0; i < D; i++) acc[i] += scr[i];
    }
    for (int i = 0; i < D; i++) out[i] += acc[i];

    free(rl); free(sc); free(w); free(sel); free(scr); free(acc);
}

#endif
