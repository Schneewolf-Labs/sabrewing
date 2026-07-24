/* moe_arch.h — arch descriptors + engine hooks for the shared MoE-native core.
 * Phase 1 finale of the MoE-runtime refactor (docs/moe-runtime-plan.md,
 * docs/moe-arch-survey.md). The forward pass is where the engines genuinely
 * diverge, so it is abstracted as strategy + template-method: a *descriptor*
 * carries the behavioral config (enums/scalars), engine-specific weight access
 * goes through *hooks* (function pointers over the engine's own layout), and the
 * shared core (moe_block.h, and later moe_attn.h) runs the algorithm.
 *
 * Each shared component takes the descriptor IT needs — not one god-struct — so
 * the coupling stays local and readable.
 *
 * Survey mapping (docs/moe-arch-survey.md):
 *   laguna: sigmoid, corr-bias, norm_topk, scale 2.5, shared UNSCALED (1 shared).
 *   inkling: sigmoid, corr-bias, joint-normalized => shared GATED, scale 8*gs.
 *   olmoe:  softmax, no bias, no shared.
 *   glm:    sigmoid, corr-bias, shared SCALED. */
#ifndef MOE_ARCH_H
#define MOE_ARCH_H

/* router score activation applied to the gate logits before top-k selection */
typedef enum { MOE_ROUTE_SIGMOID = 0, MOE_ROUTE_SOFTMAX = 1 } MoeRouteAct;

/* how the always-computed shared expert combines with the routed sum */
typedef enum {
    MOE_SHARED_NONE = 0,       /* no shared expert (olmoe) */
    MOE_SHARED_UNSCALED = 1,   /* routed*scale + shared           (laguna) */
    MOE_SHARED_SCALED = 2,     /* (routed + shared) then *scale?  (glm — see block) */
    MOE_SHARED_GATED = 3       /* shared shares the routed top-k weight pool (inkling) */
} MoeSharedMode;

/* behavioral config of one MoE block (from the engine's Cfg) */
typedef struct {
    int   n_experts;           /* routed expert count */
    int   topk;                /* experts selected per token */
    MoeRouteAct route_act;     /* sigmoid loss-free vs softmax */
    int   use_corr_bias;       /* add e_score_correction_bias for SELECTION only */
    int   norm_topk;           /* renormalize the top-k combine weights to sum 1 */
    float route_scale;         /* routed_scaling_factor applied to the routed sum */
    MoeSharedMode shared_mode; /* shared-expert combine policy */
} MoeDesc;

/* engine-provided operations over its own weight layout. ctx is opaque engine
 * state (a per-layer/token context). out buffers are length D (hidden). */
typedef struct {
    void *ctx;
    void (*router)(void *ctx, const float *x, float *logits);     /* -> logits[n_experts] */
    const float *corr_bias;                                       /* [n_experts], NULL if unused */
    void (*expert)(void *ctx, int e, const float *x, float *out); /* routed expert e -> out[D] */
    void (*shared)(void *ctx, const float *x, float *out);        /* shared expert -> out[D] (mode!=NONE) */
    /* Optional: compute the whole weighted routed sum acc[D] = sum_k w[k]*expert(sel[k],x)
     * in one shot. Lets an engine batch the K experts (e.g. one GPU submission instead
     * of K per-expert syncs). NULL -> the block loops `expert` per selection. */
    void (*expert_batch)(void *ctx, const int *sel, const float *w, int K, const float *x, float *acc);
} MoeHooks;

/* Top-K distinct expert selection: pick the K highest (score[e] + optional
 * correction bias), writing expert ids to sel[K]. This selection rule is
 * genuinely identical across the engines (loss-free sigmoid or softmax scores,
 * bias added for SELECTION only), so it is shared here even where the rest of the
 * MoE block is not — e.g. inkling's batched, streaming-cache MoE calls this for
 * routing but keeps its own gated-weight + expert-dispatch machinery. Ties break
 * toward the lower expert id (ascending scan, strict >), matching the hand loops. */
static void moe_topk(const float *score, const float *corr_bias, int use_bias, int E, int K, int *sel) {
    for (int a = 0; a < K; a++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int used = 0; for (int b = 0; b < a; b++) if (sel[b] == e) { used = 1; break; }
            if (used) continue;
            float s = score[e] + (use_bias && corr_bias ? corr_bias[e] : 0.f);
            if (s > bv) { bv = s; best = e; }
        }
        sel[a] = best;
    }
}

#endif
