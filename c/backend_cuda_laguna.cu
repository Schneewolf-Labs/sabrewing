/* CUDA backend for laguna.c — see backend_cuda_laguna.h for scope.
 * One warp per output row, f32 accumulate. The bf16 kernel expands the weight
 * bf16->f32 exact and keeps the activation in full f32, so it reproduces the
 * CPU matmul_bf16 dot (laguna.c) up to reduction order. The int4 kernels match
 * the CPU matmul_q4 scalar contract bit-for-bit apart from reduction order. */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include "backend_cuda_laguna.h"

static cudaStream_t g_st;
static float *g_dx = nullptr, *g_dy = nullptr;      /* device activation buffers */
static float *g_hx = nullptr, *g_hy = nullptr;      /* PINNED host staging */
static size_t g_xcap = 0, g_ycap = 0;
static int g_ok = 0;

extern "C" int lag_cuda_init(int dev) {
    /* many tiny matmul calls per token; spin-sync in the driver avoids the
     * yield wakeup latency (one busy thread, affordable — the OMP team is idle
     * during the GPU stream sync). */
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    if (cudaSetDevice(dev) != cudaSuccess) return -1;
    if (cudaStreamCreate(&g_st) != cudaSuccess) return -1;
    void *probe = nullptr;
    if (cudaMalloc(&probe, 1 << 20) != cudaSuccess) return -1;   /* fail fast */
    cudaFree(probe);
    g_ok = 1;
    return 0;
}

extern "C" size_t lag_cuda_free_bytes(void) {
    size_t fr = 0, to = 0;
    if (cudaMemGetInfo(&fr, &to) != cudaSuccess) return 0;
    return fr;
}
extern "C" size_t lag_cuda_total_bytes(void) {
    size_t fr = 0, to = 0;
    if (cudaMemGetInfo(&fr, &to) != cudaSuccess) return 0;
    return to;
}

extern "C" void *lag_cuda_upload(const void *h, size_t n) {
    void *d = nullptr;
    if (cudaMalloc(&d, n) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, h, n, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return nullptr; }
    return d;
}
extern "C" void lag_cuda_free(void *d) { if (d) cudaFree(d); }

/* grow the shared x/y staging buffers to hold S*I / S*O floats */
static int ensure_xy(size_t xn, size_t yn) {
    if (xn > g_xcap) {
        cudaFree(g_dx); cudaFreeHost(g_hx);
        if (cudaMalloc(&g_dx, xn * 2) != cudaSuccess ||
            cudaMallocHost(&g_hx, xn * 2) != cudaSuccess) { g_dx = g_hx = nullptr; g_xcap = 0; return -1; }
        g_xcap = xn * 2;
    }
    if (yn > g_ycap) {
        cudaFree(g_dy); cudaFreeHost(g_hy);
        if (cudaMalloc(&g_dy, yn * 2) != cudaSuccess ||
            cudaMallocHost(&g_hy, yn * 2) != cudaSuccess) { g_dy = g_hy = nullptr; g_ycap = 0; return -1; }
        g_ycap = yn * 2;
    }
    return 0;
}

/* 4 warps per block, one output row per warp; lanes stride the contraction
 * dim two bf16 at a time (4-byte coalesced load of the weight row). Weight is
 * expanded bf16->f32 exact; x stays f32 (matches laguna.c matmul_bf16). */
__global__ void mm_bf16_kernel(const __nv_bfloat16 * __restrict__ W,
                               const float * __restrict__ x,
                               float * __restrict__ y, int I, int O) {
    int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int s = blockIdx.y;
    if (o >= O) return;
    const __nv_bfloat16 *w = W + (size_t)o * I;
    const float *xs = x + (size_t)s * I;
    float acc = 0.f;
    for (int i = lane * 2; i < I; i += 64) {
        __nv_bfloat162 wv = *(const __nv_bfloat162 *)(w + i);
        acc += __bfloat162float(wv.x) * xs[i] + __bfloat162float(wv.y) * xs[i + 1];
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (!lane) y[(size_t)s * O + o] = acc;
}

extern "C" int lag_cuda_matmul_bf16(float *y, const float *x, const void *W, int S, int I, int O) {
    if (!g_ok || (I & 1)) return -1;
    size_t xn = (size_t)S * I * 4, yn = (size_t)S * O * 4;
    if (ensure_xy(xn, yn)) return -1;
    memcpy(g_hx, x, xn);
    cudaMemcpyAsync(g_dx, g_hx, xn, cudaMemcpyHostToDevice, g_st);
    dim3 grid((unsigned)((O + 3) / 4), (unsigned)S);
    mm_bf16_kernel<<<grid, 128, 0, g_st>>>((const __nv_bfloat16 *)W, g_dx, g_dy, I, O);
    cudaMemcpyAsync(g_hy, g_dy, yn, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(y, g_hy, yn);
    return 0;
}

/* Resident int8 GEMM: W is the combined [int8 O*I][f32 scale O] buffer (laguna's
 * matmul_q8 layout). Activations stay f32; one warp per output row, f32 accumulate
 * — matches the CPU MOE_Q8_F32 contract. Half the VRAM read/footprint of bf16. */
__global__ void mm_q8_kernel(const signed char * __restrict__ W, const float * __restrict__ scale,
                             const float * __restrict__ x, float * __restrict__ y, int I, int O) {
    int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int s = blockIdx.y;
    if (o >= O) return;
    const signed char *w = W + (size_t)o * I;
    const float *xs = x + (size_t)s * I;
    float acc = 0.f;
    for (int i = lane; i < I; i += 32) acc += xs[i] * (float)w[i];
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (!lane) y[(size_t)s * O + o] = acc * scale[o];
}

extern "C" int lag_cuda_matmul_q8(float *y, const float *x, const void *W, int S, int I, int O) {
    if (!g_ok) return -1;
    size_t xn = (size_t)S * I * 4, yn = (size_t)S * O * 4;
    if (ensure_xy(xn, yn)) return -1;
    memcpy(g_hx, x, xn);
    cudaMemcpyAsync(g_dx, g_hx, xn, cudaMemcpyHostToDevice, g_st);
    const signed char *q8 = (const signed char *)W;
    const float *scale = (const float *)(q8 + (size_t)I * O);   /* embedded per-row scales */
    dim3 grid((unsigned)((O + 3) / 4), (unsigned)S);
    mm_q8_kernel<<<grid, 128, 0, g_st>>>(q8, scale, g_dx, g_dy, I, O);
    cudaMemcpyAsync(g_hy, g_dy, yn, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(y, g_hy, yn);
    return 0;
}

/* Routed-expert int4 GEMM. Matches CPU matmul_q4: packed nibbles (low = even
 * col, high = odd col, value = nibble-8), per-output-row f32 scale, f32
 * accumulate. packed/scale are DEVICE pointers. One warp per output row. */
__global__ void mm_q4_kernel(const unsigned char * __restrict__ P,
                             const float * __restrict__ scale,
                             const float * __restrict__ x,
                             float * __restrict__ y, int I, int O) {
    int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int s = blockIdx.y;
    if (o >= O) return;
    const unsigned char *w = P + (size_t)o * (I >> 1);
    const float *xs = x + (size_t)s * I;
    int nb = I >> 1;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        unsigned char byte = w[b];
        acc += xs[2*b]     * (float)((int)(byte & 0xF) - 8)
             + xs[2*b + 1] * (float)((int)(byte >> 4)  - 8);
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (!lane) y[(size_t)s * O + o] = acc * scale[o];
}

extern "C" int lag_cuda_matmul_q4(float *y, const float *x, const void *packed,
                                  const void *scale, int S, int I, int O) {
    if (!g_ok || (I & 1)) return -1;
    size_t xn = (size_t)S * I * 4, yn = (size_t)S * O * 4;
    if (ensure_xy(xn, yn)) return -1;
    memcpy(g_hx, x, xn);
    cudaMemcpyAsync(g_dx, g_hx, xn, cudaMemcpyHostToDevice, g_st);
    dim3 grid((unsigned)((O + 3) / 4), (unsigned)S);
    mm_q4_kernel<<<grid, 128, 0, g_st>>>((const unsigned char *)packed, (const float *)scale,
                                         g_dx, g_dy, I, O);
    cudaMemcpyAsync(g_hy, g_dy, yn, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(y, g_hy, yn);
    return 0;
}

/* SiLU-GLU gate: gg[i] = silu(g[i]) * g[I+i], reading the fused 2I gate+up row
 * (block-concat: g[0..I) gate, g[I..2I) up). expf (not __expf) to match siluf. */
__global__ void silu_glu_kernel(const float * __restrict__ g, float * __restrict__ gg, int I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    float a = g[i];
    gg[i] = (a / (1.f + expf(-a))) * g[I + i];
}

/* device scratch for the fused single-expert path (S=1): gate+up[2I], glu[I],
 * out[D]; x reuses g_dx. Grown on demand. */
static float *e_dg = nullptr, *e_dgg = nullptr, *e_dout = nullptr, *e_hout = nullptr;
static int e_I = 0, e_D = 0;

extern "C" int lag_cuda_expert_q4(float *out, const float *x,
                                  const void *p13, const void *s13,
                                  const void *p2, const void *s2, int I, int D) {
    if (!g_ok || (D & 1) || (I & 1)) return -1;
    size_t xn = (size_t)D * 4;
    if (xn > g_xcap) {
        cudaFree(g_dx); cudaFreeHost(g_hx);
        if (cudaMalloc(&g_dx, xn * 2) != cudaSuccess ||
            cudaMallocHost(&g_hx, xn * 2) != cudaSuccess) { g_dx = g_hx = nullptr; g_xcap = 0; return -1; }
        g_xcap = xn * 2;
    }
    if (I > e_I) { cudaFree(e_dg); cudaFree(e_dgg);
        if (cudaMalloc(&e_dg, (size_t)2*I*4) != cudaSuccess ||
            cudaMalloc(&e_dgg, (size_t)I*4) != cudaSuccess) { e_dg = e_dgg = nullptr; e_I = 0; return -1; }
        e_I = I; }
    if (D > e_D) { cudaFree(e_dout); cudaFreeHost(e_hout);
        if (cudaMalloc(&e_dout, (size_t)D*4) != cudaSuccess ||
            cudaMallocHost(&e_hout, (size_t)D*4) != cudaSuccess) { e_dout = e_hout = nullptr; e_D = 0; return -1; }
        e_D = D; }
    memcpy(g_hx, x, xn);
    cudaMemcpyAsync(g_dx, g_hx, xn, cudaMemcpyHostToDevice, g_st);
    mm_q4_kernel<<<dim3((unsigned)((2*I + 3) / 4), 1), 128, 0, g_st>>>(
        (const unsigned char *)p13, (const float *)s13, g_dx, e_dg, D, 2*I);
    silu_glu_kernel<<<(unsigned)((I + 255) / 256), 256, 0, g_st>>>(e_dg, e_dgg, I);
    mm_q4_kernel<<<dim3((unsigned)((D + 3) / 4), 1), 128, 0, g_st>>>(
        (const unsigned char *)p2, (const float *)s2, e_dgg, e_dout, I, D);
    cudaMemcpyAsync(e_hout, e_dout, (size_t)D*4, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(out, e_hout, (size_t)D*4);
    return 0;
}

/* acc[i] += w * v[i] (device weighted-accumulate) */
__global__ void axpy_kernel(float * __restrict__ acc, float w, const float * __restrict__ v, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) acc[i] += w * v[i];
}
__global__ void zero_kernel(float * __restrict__ p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.f;
}

/* Batched routed experts for one VRAM-resident MoE layer (S=1 decode): computes
 * out[D] = sum_k w[k] * expert(sel[k], x) for all K selected experts in a SINGLE
 * GPU submission — one HtoD (x), K*(mm_q4 + silu + mm_q4 + axpy) launches queued
 * on the stream with a device-side accumulator, one DtoH, ONE sync. Replaces K
 * calls to lag_cuda_expert_q4 (K syncs) with one — the decode bottleneck was the
 * per-expert sync round-trips, not the expert compute. gu_q/gu_s/dn_q/dn_s are the
 * layer's DEVICE expert blobs; sel/w are host arrays. Each expert is numerically
 * the fused single-expert path, so it matches lag_cuda_expert_q4. */
static float *me_acc = nullptr; static int me_D = 0;
extern "C" int lag_cuda_moe_experts(float *out, const float *x,
                                    const void *gu_q, const float *gu_s,
                                    const void *dn_q, const float *dn_s,
                                    const int *sel, const float *w, int K, int I, int D) {
    if (!g_ok || (D & 1) || (I & 1)) return -1;
    size_t xn = (size_t)D * 4;
    if (xn > g_xcap) {
        cudaFree(g_dx); cudaFreeHost(g_hx);
        if (cudaMalloc(&g_dx, xn * 2) != cudaSuccess ||
            cudaMallocHost(&g_hx, xn * 2) != cudaSuccess) { g_dx = g_hx = nullptr; g_xcap = 0; return -1; }
        g_xcap = xn * 2;
    }
    if (I > e_I) { cudaFree(e_dg); cudaFree(e_dgg);
        if (cudaMalloc(&e_dg, (size_t)2*I*4) != cudaSuccess ||
            cudaMalloc(&e_dgg, (size_t)I*4) != cudaSuccess) { e_dg = e_dgg = nullptr; e_I = 0; return -1; }
        e_I = I; }
    if (D > e_D) { cudaFree(e_dout); cudaFreeHost(e_hout);
        if (cudaMalloc(&e_dout, (size_t)D*4) != cudaSuccess ||
            cudaMallocHost(&e_hout, (size_t)D*4) != cudaSuccess) { e_dout = e_hout = nullptr; e_D = 0; return -1; }
        e_D = D; }
    if (D > me_D) { cudaFree(me_acc);
        if (cudaMalloc(&me_acc, (size_t)D*4) != cudaSuccess) { me_acc = nullptr; me_D = 0; return -1; }
        me_D = D; }
    memcpy(g_hx, x, xn);
    cudaMemcpyAsync(g_dx, g_hx, xn, cudaMemcpyHostToDevice, g_st);
    zero_kernel<<<(unsigned)((D + 255) / 256), 256, 0, g_st>>>(me_acc, D);
    for (int k = 0; k < K; k++) {
        int e = sel[k];
        const unsigned char *p13 = (const unsigned char *)gu_q + (size_t)e * (2*I) * (D/2);
        const float         *s13 = gu_s + (size_t)e * (2*I);
        const unsigned char *p2  = (const unsigned char *)dn_q + (size_t)e * D * (I/2);
        const float         *s2  = dn_s + (size_t)e * D;
        mm_q4_kernel<<<dim3((unsigned)((2*I + 3) / 4), 1), 128, 0, g_st>>>(p13, s13, g_dx, e_dg, D, 2*I);
        silu_glu_kernel<<<(unsigned)((I + 255) / 256), 256, 0, g_st>>>(e_dg, e_dgg, I);
        mm_q4_kernel<<<dim3((unsigned)((D + 3) / 4), 1), 128, 0, g_st>>>(p2, s2, e_dgg, e_dout, I, D);
        axpy_kernel<<<(unsigned)((D + 255) / 256), 256, 0, g_st>>>(me_acc, w[k], e_dout, D);
    }
    cudaMemcpyAsync(e_hout, me_acc, (size_t)D*4, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(out, e_hout, (size_t)D*4);
    return 0;
}


/* ---------------- fp4 (e2m1) + per-32-block ue8m0 scales ----------------
 * DeepSeek-V4 ships its routed experts in this format, so the container keeps it
 * verbatim rather than re-gridding onto int4 (which measured 17% relative error —
 * e2m1's levels are non-uniform). Mirrors mm_q4_kernel exactly apart from the
 * value decode: one warp per output row, f32 accumulate, same reduction order. */
__constant__ float d_fp4_lut[16] = {
    0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

/* 2^(e-127). e==0 is the denormal 2^-127, e==0xFF the format NaN — same two
 * special cases as moe_ue8m0() on the host, so the decode agrees bit-for-bit. */
__device__ __forceinline__ float ue8m0f(unsigned char e) {
    if (e == 0xFF) return 0.f;
    if (e == 0)    return 5.8774717541114375e-39f;
    return __int_as_float(((int)e) << 23);
}

__global__ void mm_fp4_kernel(const unsigned char * __restrict__ P,
                              const unsigned char * __restrict__ ES,
                              const float * __restrict__ x,
                              float * __restrict__ y, int I, int O) {
    int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int s = blockIdx.y;
    if (o >= O) return;
    const unsigned char *w = P + (size_t)o * (I >> 1);
    const unsigned char *e = ES + (size_t)o * (I >> 5);
    const float *xs = x + (size_t)s * I;
    int nb = I >> 1;
    float acc = 0.f;
    /* byte b holds columns 2b and 2b+1, which live in scale block (2b)/32 = b/16 */
    for (int b = lane; b < nb; b += 32) {
        unsigned char byte = w[b];
        float sc = ue8m0f(e[b >> 4]);
        acc += (xs[2*b] * d_fp4_lut[byte & 0xF] + xs[2*b + 1] * d_fp4_lut[byte >> 4]) * sc;
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (!lane) y[(size_t)s * O + o] = acc;
}

extern "C" int lag_cuda_matmul_fp4(float *y, const float *x, const void *packed,
                                   const void *es, int S, int I, int O) {
    if (!g_ok || (I & 31)) return -1;          /* scale blocks are 32 wide */
    size_t xn = (size_t)S * I * 4, yn = (size_t)S * O * 4;
    if (ensure_xy(xn, yn)) return -1;
    memcpy(g_hx, x, xn);
    cudaMemcpyAsync(g_dx, g_hx, xn, cudaMemcpyHostToDevice, g_st);
    dim3 grid((unsigned)((O + 3) / 4), (unsigned)S);
    mm_fp4_kernel<<<grid, 128, 0, g_st>>>((const unsigned char *)packed,
                                          (const unsigned char *)es, g_dx, g_dy, I, O);
    cudaMemcpyAsync(g_hy, g_dy, yn, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(y, g_hy, yn);
    return 0;
}

/* V4's SwiGLU is clamped, and ASYMMETRICALLY: gate.clamp(max=limit) but
 * up.clamp(-limit, limit). Getting that symmetric is silent — it only shows on
 * outliers — so this is a separate kernel rather than a tweak to silu_glu_kernel.
 * expf (not __expf) to match the host siluf. */
__global__ void silu_glu_clamp_kernel(const float * __restrict__ g, float * __restrict__ gg,
                                      int I, float lim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    float a = g[i], u = g[I + i];
    a = a > lim ? lim : a;
    u = u > lim ? lim : (u < -lim ? -lim : u);
    gg[i] = (a / (1.f + expf(-a))) * u;
}

extern "C" int lag_cuda_moe_experts_fp4(float *out, const float *x,
                                        const void *gu_q, const void *gu_es,
                                        const void *dn_q, const void *dn_es,
                                        const int *sel, const float *w, int K,
                                        int I, int D, float limit) {
    if (!g_ok || (D & 31) || (I & 31)) return -1;
    size_t xn = (size_t)D * 4;
    if (xn > g_xcap) {
        cudaFree(g_dx); cudaFreeHost(g_hx);
        if (cudaMalloc(&g_dx, xn * 2) != cudaSuccess ||
            cudaMallocHost(&g_hx, xn * 2) != cudaSuccess) { g_dx = g_hx = nullptr; g_xcap = 0; return -1; }
        g_xcap = xn * 2;
    }
    if (I > e_I) { cudaFree(e_dg); cudaFree(e_dgg);
        if (cudaMalloc(&e_dg, (size_t)2*I*4) != cudaSuccess ||
            cudaMalloc(&e_dgg, (size_t)I*4) != cudaSuccess) { e_dg = e_dgg = nullptr; e_I = 0; return -1; }
        e_I = I; }
    if (D > e_D) { cudaFree(e_dout); cudaFreeHost(e_hout);
        if (cudaMalloc(&e_dout, (size_t)D*4) != cudaSuccess ||
            cudaMallocHost(&e_hout, (size_t)D*4) != cudaSuccess) { e_dout = e_hout = nullptr; e_D = 0; return -1; }
        e_D = D; }
    if (D > me_D) { cudaFree(me_acc);
        if (cudaMalloc(&me_acc, (size_t)D*4) != cudaSuccess) { me_acc = nullptr; me_D = 0; return -1; }
        me_D = D; }
    memcpy(g_hx, x, xn);
    cudaMemcpyAsync(g_dx, g_hx, xn, cudaMemcpyHostToDevice, g_st);
    zero_kernel<<<(unsigned)((D + 255) / 256), 256, 0, g_st>>>(me_acc, D);
    for (int k = 0; k < K; k++) {
        int e = sel[k];
        const unsigned char *p13 = (const unsigned char *)gu_q  + (size_t)e * (2*I) * (D/2);
        const unsigned char *b13 = (const unsigned char *)gu_es + (size_t)e * (2*I) * (D/32);
        const unsigned char *p2  = (const unsigned char *)dn_q  + (size_t)e * D * (I/2);
        const unsigned char *b2  = (const unsigned char *)dn_es + (size_t)e * D * (I/32);
        mm_fp4_kernel<<<dim3((unsigned)((2*I + 3) / 4), 1), 128, 0, g_st>>>(p13, b13, g_dx, e_dg, D, 2*I);
        silu_glu_clamp_kernel<<<(unsigned)((I + 255) / 256), 256, 0, g_st>>>(e_dg, e_dgg, I, limit);
        mm_fp4_kernel<<<dim3((unsigned)((D + 3) / 4), 1), 128, 0, g_st>>>(p2, b2, e_dgg, e_dout, I, D);
        axpy_kernel<<<(unsigned)((D + 255) / 256), 256, 0, g_st>>>(me_acc, w[k], e_dout, D);
    }
    cudaMemcpyAsync(e_hout, me_acc, (size_t)D*4, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(out, e_hout, (size_t)D*4);
    return 0;
}

/* ---------------- grouped (Megablocks-style) routed experts ----------------
 * See backend_cuda_laguna.h for the contract. The layer's S*K (row, expert) pairs
 * arrive sorted by expert; each group runs as a GEMM over its rows so the expert
 * weight is read from VRAM once per group instead of once per row. Numerics match
 * the per-row kernels exactly (same warp reduction, same rank-order combine). */

/* xg[j,:] = x[grow[j],:] — pack each group's rows contiguously */
__global__ void gather_rows_kernel(float * __restrict__ xg, const float * __restrict__ x,
                                   const int * __restrict__ grow, int T, int D) {
    size_t n = (size_t)T * D, stride = (size_t)gridDim.x * blockDim.x;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        int j = (int)(t / D), i = (int)(t % D);
        xg[t] = x[(size_t)grow[j] * D + i];
    }
}

/* Grouped int4 GEMM: blockIdx.y = group, one warp per output row of that group's
 * expert, looping the group's rows inside. The weight row is fetched once and
 * stays in L1/L2 across the group's rows (the bandwidth win); the per-(row,o) dot
 * is lane-strided and shfl-reduced exactly like mm_q4_kernel, so every output is
 * bit-identical to the ungrouped kernel. */
__global__ void mm_q4_group_kernel(const unsigned char * __restrict__ P,
                                   const float * __restrict__ scale,
                                   const float * __restrict__ X, float * __restrict__ Y,
                                   const int * __restrict__ gexp, const int * __restrict__ gbeg,
                                   const int * __restrict__ gcnt, int Ic, int O) {
    int g = blockIdx.y;
    int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (o >= O) return;
    int e = gexp[g], beg = gbeg[g], n = gcnt[g], nb = Ic >> 1;
    const unsigned char *w = P + (size_t)e * O * nb + (size_t)o * nb;
    float sc = scale[(size_t)e * O + o];
    for (int j = 0; j < n; j++) {
        const float *xs = X + (size_t)(beg + j) * Ic;
        float acc = 0.f;
        for (int b = lane; b < nb; b += 32) {
            unsigned char byte = w[b];
            acc += xs[2*b]     * (float)((int)(byte & 0xF) - 8)
                 + xs[2*b + 1] * (float)((int)(byte >> 4)  - 8);
        }
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (!lane) Y[(size_t)(beg + j) * O + o] = acc * sc;
    }
}

/* SiLU-GLU over packed rows: gg[r,i] = silu(g[r,i]) * g[r,I+i] (fused 2I rows) */
__global__ void silu_glu_rows_kernel(const float * __restrict__ g, float * __restrict__ gg,
                                     int T, int I) {
    size_t n = (size_t)T * I, stride = (size_t)gridDim.x * blockDim.x;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        int r = (int)(t / I), i = (int)(t % I);
        float a = g[(size_t)r * 2 * I + i];
        gg[t] = (a / (1.f + expf(-a))) * g[(size_t)r * 2 * I + I + i];
    }
}

/* acc[s,i] = sum_a w[s,a] * slots[rowof[s,a], i] — accumulated in ascending rank
 * order, i.e. the same float op sequence as the per-token axpy chain. */
__global__ void combine_rank_kernel(float * __restrict__ acc, const float * __restrict__ slots,
                                    const int * __restrict__ rowof, const float * __restrict__ w,
                                    int S, int K, int D) {
    size_t n = (size_t)S * D, stride = (size_t)gridDim.x * blockDim.x;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        int s = (int)(t / D), i = (int)(t % D);
        float sum = 0.f;
        for (int a = 0; a < K; a++)
            sum += w[s * K + a] * slots[(size_t)rowof[s * K + a] * D + i];
        acc[t] = sum;
    }
}

/* grow a device buffer to `need` bytes (2x slack so batch/chunk churn doesn't
 * re-malloc every call); -1 on OOM, leaving the buffer released. */
static int grow_dev(void **p, size_t *cap, size_t need) {
    if (need <= *cap) return 0;
    cudaFree(*p); *p = nullptr; *cap = 0;
    if (cudaMalloc(p, need * 2) != cudaSuccess) { *p = nullptr; return -1; }
    *cap = need * 2;
    return 0;
}
static int grow_pinned(void **p, size_t *cap, size_t need) {
    if (need <= *cap) return 0;
    cudaFreeHost(*p); *p = nullptr; *cap = 0;
    if (cudaMallocHost(p, need * 2) != cudaSuccess) { *p = nullptr; return -1; }
    *cap = need * 2;
    return 0;
}

/* grouped-path scratch (all grown on demand, reused across layers/steps) */
static void *gp_dx = nullptr, *gp_dxg = nullptr, *gp_dgu = nullptr, *gp_dglu = nullptr;
static void *gp_dslot = nullptr, *gp_dacc = nullptr, *gp_didx = nullptr;
static void *gp_hx = nullptr, *gp_hacc = nullptr, *gp_hidx = nullptr;
static size_t gp_cx = 0, gp_cxg = 0, gp_cgu = 0, gp_cglu = 0, gp_cslot = 0, gp_cacc = 0,
              gp_cidx = 0, gp_chx = 0, gp_chacc = 0, gp_chidx = 0;

extern "C" int lag_cuda_moe_group(float *acc, const float *x,
                                  const void *gu_q, const float *gu_s,
                                  const void *dn_q, const float *dn_s,
                                  const int *gexp, const int *gbeg, const int *gcnt, int G,
                                  const int *grow, const int *rowof, const float *w,
                                  int S, int K, int I, int D) {
    if (!g_ok || (D & 1) || (I & 1) || G <= 0) return -1;
    int T = S * K;                                  /* packed rows: each token contributes K */
    size_t xn = (size_t)S * D * 4, accn = xn;
    size_t xgn = (size_t)T * D * 4, gun = (size_t)T * 2 * I * 4;
    size_t glun = (size_t)T * I * 4, slotn = (size_t)T * D * 4;
    size_t nint = (size_t)3 * G + T + T;            /* gexp,gbeg,gcnt | grow | rowof */
    size_t idxn = nint * sizeof(int) + (size_t)T * sizeof(float);   /* + w[S*K] */
    if (grow_dev(&gp_dx, &gp_cx, xn) || grow_dev(&gp_dxg, &gp_cxg, xgn) ||
        grow_dev(&gp_dgu, &gp_cgu, gun) || grow_dev(&gp_dglu, &gp_cglu, glun) ||
        grow_dev(&gp_dslot, &gp_cslot, slotn) || grow_dev(&gp_dacc, &gp_cacc, accn) ||
        grow_dev(&gp_didx, &gp_cidx, idxn) ||
        grow_pinned(&gp_hx, &gp_chx, xn) || grow_pinned(&gp_hacc, &gp_chacc, accn) ||
        grow_pinned(&gp_hidx, &gp_chidx, idxn)) return -1;

    /* one pinned staging blob for every index array + the combine weights */
    int *hi = (int*)gp_hidx;
    memcpy(hi,                 gexp,  (size_t)G * sizeof(int));
    memcpy(hi + G,             gbeg,  (size_t)G * sizeof(int));
    memcpy(hi + 2*G,           gcnt,  (size_t)G * sizeof(int));
    memcpy(hi + 3*G,           grow,  (size_t)T * sizeof(int));
    memcpy(hi + 3*G + T,       rowof, (size_t)T * sizeof(int));
    memcpy(hi + 3*G + 2*T,     w,     (size_t)T * sizeof(float));
    memcpy(gp_hx, x, xn);
    cudaMemcpyAsync(gp_dx, gp_hx, xn, cudaMemcpyHostToDevice, g_st);
    cudaMemcpyAsync(gp_didx, gp_hidx, idxn, cudaMemcpyHostToDevice, g_st);
    const int *d_gexp = (const int*)gp_didx, *d_gbeg = d_gexp + G, *d_gcnt = d_gexp + 2*G;
    const int *d_grow = d_gexp + 3*G, *d_rowof = d_grow + T;
    const float *d_w = (const float*)(d_rowof + T);

    int blk = 256, gridT = (int)(((size_t)T * D + blk - 1) / blk); if (gridT > 4096) gridT = 4096;
    gather_rows_kernel<<<gridT, blk, 0, g_st>>>((float*)gp_dxg, (const float*)gp_dx, d_grow, T, D);
    mm_q4_group_kernel<<<dim3((unsigned)((2*I + 3) / 4), (unsigned)G), 128, 0, g_st>>>(
        (const unsigned char*)gu_q, gu_s, (const float*)gp_dxg, (float*)gp_dgu,
        d_gexp, d_gbeg, d_gcnt, D, 2*I);
    int gridG = (int)(((size_t)T * I + blk - 1) / blk); if (gridG > 4096) gridG = 4096;
    silu_glu_rows_kernel<<<gridG, blk, 0, g_st>>>((const float*)gp_dgu, (float*)gp_dglu, T, I);
    mm_q4_group_kernel<<<dim3((unsigned)((D + 3) / 4), (unsigned)G), 128, 0, g_st>>>(
        (const unsigned char*)dn_q, dn_s, (const float*)gp_dglu, (float*)gp_dslot,
        d_gexp, d_gbeg, d_gcnt, I, D);
    int gridS = (int)(((size_t)S * D + blk - 1) / blk); if (gridS > 4096) gridS = 4096;
    combine_rank_kernel<<<gridS, blk, 0, g_st>>>((float*)gp_dacc, (const float*)gp_dslot,
                                                 d_rowof, d_w, S, K, D);
    cudaMemcpyAsync(gp_hacc, gp_dacc, accn, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(acc, gp_hacc, accn);
    return 0;
}
