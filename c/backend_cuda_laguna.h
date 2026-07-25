/* Minimal CUDA backend for laguna.c: bf16 resident weights live in VRAM,
 * their matmuls run on-device. Motivation is bandwidth, not FLOPs: decode reads
 * ~8 GB of bf16 residents per token, which caps CPU decode at the DDR5
 * bandwidth wall (~2.3 tok/s on this box); the A6000 feeds the same reads from
 * VRAM at ~15x. The routed experts (int4, ~53 GB) do NOT fit in 48 GB VRAM
 * alongside the residents, so they stay on the CPU int4 path in this phase —
 * attention, routing and everything else stay on the CPU too.
 *
 * Numeric contract mirrors laguna.c's CPU kernels so GPU and CPU runs stay
 * closely comparable (float-noise apart, from a different reduction order):
 *   - matmul_bf16: weight rounded bf16->f32 EXACT, activation kept full f32
 *     (laguna.c line ~124 loads x as f32 — it does NOT round x to bf16, unlike
 *     inkling's backend; this header's kernel matches laguna, not inkling).
 *   - matmul_q4 / expert_q4: packed nibble-8 weights, per-output-row f32 scale,
 *     f32 accumulate, no activation quantization (== CPU matmul_q4). gate_up is
 *     block-concatenated per expert (all gate rows, then all up rows).
 */
#ifndef BACKEND_CUDA_LAGUNA_H
#define BACKEND_CUDA_LAGUNA_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

int    lag_cuda_init(int dev);                 /* 0 = ok */
size_t lag_cuda_free_bytes(void);
size_t lag_cuda_total_bytes(void);
void  *lag_cuda_upload(const void *h, size_t n);   /* NULL = OOM/error */
void   lag_cuda_free(void *d);
/* y[S,O] = x[S,I] @ W^T, W = device bf16 [O,I]; x,y host f32. Weight is expanded
 * bf16->f32 exact, x stays f32 (== CPU matmul_bf16). 0 = ok */
int    lag_cuda_matmul_bf16(float *y, const float *x, const void *W, int S, int I, int O);
/* y[S,O] = x[S,I] @ dequant(W)^T, W = device buffer [int8 O*I][f32 scale O]
 * (laguna's combined int8-resident layout); activations stay f32. Matches CPU
 * matmul_q8 (MOE_Q8_F32). Halves the VRAM read + footprint vs bf16 residents. */
int    lag_cuda_matmul_q8(float *y, const float *x, const void *W, int S, int I, int O);
/* y[S,O] = x[S,I] @ dequant(W)^T, W = device packed int4 [O,I/2] + device f32
 * row scales [O] (nibble-8, per-row scale). Matches CPU matmul_q4. 0 = ok */
int    lag_cuda_matmul_q4(float *y, const float *x, const void *packed,
                          const void *scale, int S, int I, int O);
/* Fused routed expert (int4, S=1 decode): out[D] = W2 @ siluglu(W13 @ x), device
 * weights (packed int4 + f32 row scales), gate_up block-concatenated. One HtoD/
 * DtoH/sync, no host silu bounce. Matches CPU matmul_q4 + siluf expert path. */
int    lag_cuda_expert_q4(float *out, const float *x,
                          const void *p13, const void *s13,
                          const void *p2, const void *s2, int I, int D);
/* Batched routed experts for one VRAM-resident layer: out[D] = sum_k w[k] *
 * expert(sel[k], x) for all K selected experts in ONE GPU submission / one sync
 * (device-side accumulate). gu_q/gu_s/dn_q/dn_s are the layer's device expert
 * blobs; sel/w are host arrays. Kills the per-expert sync round-trips. */
int    lag_cuda_moe_experts(float *out, const float *x,
                            const void *gu_q, const float *gu_s,
                            const void *dn_q, const float *dn_s,
                            const int *sel, const float *w, int K, int I, int D);

#ifdef __cplusplus
}
#endif
#endif
