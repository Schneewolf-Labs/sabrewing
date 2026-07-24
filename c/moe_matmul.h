/* moe_matmul.h — shared f32 GEMV/GEMM for the MoE engines.
 * Phase 1 of the MoE-runtime refactor (docs/moe-runtime-plan.md).
 *
 * y[S,O] = x[S,I] @ W[O,I]^T  (W row-major [out,in], HF Linear convention).
 * Batched over S rows — the general/prefill-friendly form (inkling was already
 * batched; laguna called the S=1 special case). One canonical implementation:
 * AVX-512 FMA (f32 accumulate) for speed, with an `exact` double-accumulate
 * fallback used by the token-exact oracle. For S=1 this is bit-identical to
 * laguna's previous single-row matmul (same per-output reduction order); the
 * batched path reduces each (o,s) independently, so S>1 is just the same dot
 * repeated — no cross-row reduction that could perturb numerics.
 *
 * This is the ONLY oracle-covered kernel (the tiny oracle runs bits=0 = f32).
 * The bf16/int8/int4 kernels have real per-engine contract differences (laguna
 * keeps activations f32; inkling rounds them to bf16) and are NOT unified here —
 * they need a dedicated kernel test harness first. */
#ifndef MOE_MATMUL_H
#define MOE_MATMUL_H
#include <stdint.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

static void matmul_f32(float *y, const float *x, const float *W, int S, int I, int O, int exact) {
#if defined(__AVX512F__)
    if (!exact) {
        #pragma omp parallel for schedule(static) if(O >= 512)
        for (int o = 0; o < O; o++) {
            const float *w = W + (int64_t)o * I;
            for (int s = 0; s < S; s++) {
                const float *xs = x + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                int i = 0;
                for (; i + 16 <= I; i += 16)
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(xs + i), _mm512_loadu_ps(w + i), acc);
                float sm = _mm512_reduce_add_ps(acc);
                for (; i < I; i++) sm += xs[i] * w[i];
                y[(int64_t)s * O + o] = sm;
            }
        }
        return;
    }
#endif
    /* exact double-accumulate reference (the oracle path) */
    #pragma omp parallel for schedule(static) if(O >= 512)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            double sm = 0; for (int i = 0; i < I; i++) sm += (double)xs[i] * w[i];
            y[(int64_t)s * O + o] = (float)sm;
        }
    }
}

#endif
