/* moe_hc.h — Manifold-Constrained Hyper-Connections (mHC), shared by the engines that
 * carry several residual streams per token (deepseek.c: DeepSeek-V4, glm53.c: GLM-5.3).
 *
 * Every token carries `hc` parallel residual streams. At each sublayer site a small
 * learned map turns the (RMS-normed, flattened) streams into
 *   pre  [hc]     sigmoid + eps        — collapses the streams into the sublayer input
 *   post [hc]     2 * sigmoid          — places the sublayer output back across streams
 *   comb [hc,hc]  softmax rows + eps, then Sinkhorn-Knopp onto the doubly-stochastic
 *                 manifold (first COLUMNS, then iters-1 x (rows, columns))
 * and the streams advance as  streams[k] = post[k] * y + sum_j comb[j][k] * streams[j].
 * `comb` is consumed TRANSPOSED (the sum runs over the first axis); Sinkhorn output is
 * doubly stochastic but not symmetric, so the direction is load-bearing and silent when
 * wrong. Both engines validate this against their transformers oracle. */
#ifndef MOE_HC_H
#define MOE_HC_H
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "moe_math.h"          /* softmax_row */
#include "moe_matmul.h"        /* matmul_f32 */

#define MOE_HC_MAX 8

typedef struct {
    int hc, D, iters;          /* streams, hidden size, Sinkhorn iterations */
    float eps;                 /* hc_eps: the pre/comb floor and Sinkhorn denominators */
    float norm_eps;            /* the unweighted RMSNorm in front of the fn projection */
} MoeHC;

/* unweighted RMSNorm (double sum of squares) */
static void moe_rmsnorm_plain(float *out, const float *x, int D, float eps) {
    double ss = 0; for (int i = 0; i < D; i++) ss += (double)x[i] * x[i];
    float inv = 1.f / sqrtf((float)(ss / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * inv;
}

/* (pre, post, comb) from one token's mix logits, then collapse its streams with pre.
 * `mix` = fn . rmsnorm_plain(streams), computed for the whole chunk by moe_hc_sites. */
static void moe_hc_site(const MoeHC *h, const float *base, const float *scale,
                        const float *streams, float *post, float *comb, float *collapsed,
                        const float *mix) {
    int hc = h->hc, D = h->D;
    float eps = h->eps, pre[MOE_HC_MAX] = {0};
    for (int k = 0; k < hc; k++)
        pre[k] = 1.f / (1.f + expf(-(mix[k] * scale[0] + base[k]))) + eps;
    for (int k = 0; k < hc; k++)
        post[k] = 2.f / (1.f + expf(-(mix[hc + k] * scale[1] + base[hc + k])));
    for (int j = 0; j < hc; j++) {
        float *row = comb + j * hc;
        for (int k = 0; k < hc; k++)
            row[k] = mix[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
        softmax_row(row, hc);
        for (int k = 0; k < hc; k++) row[k] += eps;
    }
    for (int k = 0; k < hc; k++) {
        float s = 0; for (int j = 0; j < hc; j++) s += comb[j * hc + k];
        s += eps; for (int j = 0; j < hc; j++) comb[j * hc + k] /= s;
    }
    for (int it = 0; it < h->iters - 1; it++) {
        for (int j = 0; j < hc; j++) {
            float s = 0; for (int k = 0; k < hc; k++) s += comb[j * hc + k];
            s += eps; for (int k = 0; k < hc; k++) comb[j * hc + k] /= s;
        }
        for (int k = 0; k < hc; k++) {
            float s = 0; for (int j = 0; j < hc; j++) s += comb[j * hc + k];
            s += eps; for (int j = 0; j < hc; j++) comb[j * hc + k] /= s;
        }
    }
    for (int i = 0; i < D; i++) {
        float a = 0; for (int k = 0; k < hc; k++) a += pre[k] * streams[k * D + i];
        collapsed[i] = a;
    }
}

/* One mHC site for n tokens. The fn projection ([mixn, hc*D]; 24 x 16384 on the real
 * models) runs as ONE S=n GEMM instead of n single-row ones — bit-identical per row
 * (moe_matmul.h), and single-threaded at S=1 (24 outputs is under the OpenMP threshold).
 * Scratch: flat [n, hc*D], mix [n, (2+hc)*hc]. */
static void moe_hc_sites(const MoeHC *h, const float *fn, const float *base, const float *scale,
                         const float *hs, int n, float *post, float *comb, float *coll,
                         float *flat, float *mix, int exact) {
    int hc = h->hc, D = h->D, mixn = (2 + hc) * hc;
    #pragma omp parallel for schedule(static) if(n > 1)
    for (int t = 0; t < n; t++)
        moe_rmsnorm_plain(flat + (int64_t)t * hc * D, hs + (int64_t)t * hc * D, hc * D, h->norm_eps);
    matmul_f32(mix, flat, fn, n, hc * D, mixn, exact);
    #pragma omp parallel for schedule(static) if(n > 1)
    for (int t = 0; t < n; t++)
        moe_hc_site(h, base, scale, hs + (int64_t)t * hc * D, post + (int64_t)t * hc,
                    comb + (int64_t)t * hc * hc, coll + (int64_t)t * D, mix + (int64_t)t * mixn);
}

/* streams[k] = post[k]*y + sum_j comb[j][k]*streams[j]   (comb consumed TRANSPOSED).
 * tmp: [hc*D] scratch. */
static void moe_hc_merge(const MoeHC *h, float *streams, const float *post, const float *comb,
                         const float *y, float *tmp) {
    int hc = h->hc, D = h->D;
    memcpy(tmp, streams, (size_t)hc * D * sizeof(float));
    for (int k = 0; k < hc; k++) {
        float *dst = streams + (int64_t)k * D;
        for (int i = 0; i < D; i++) {
            float a = post[k] * y[i];
            for (int j = 0; j < hc; j++) a += comb[j * hc + k] * tmp[j * D + i];
            dst[i] = a;
        }
    }
}

#endif
