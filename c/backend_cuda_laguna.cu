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
