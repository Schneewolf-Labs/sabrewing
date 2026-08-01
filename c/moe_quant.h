/* moe_quant.h — shared quantized GEMV kernels for the MoE engines.
 * Phase 1 of the MoE-runtime refactor (docs/moe-runtime-plan.md).
 *
 * These are NOT oracle-covered (the tiny oracle runs bits=0 = f32), so they are
 * validated by a kernel harness (tools/kernel_check.c: dequant double-precision
 * reference) plus real-model perplexity. The int4 kernel carries BOTH engines'
 * activation contracts, selected per call so each stays bit-identical to its
 * origin:
 *   MOE_Q4_F32  — activations kept in f32, AVX-512 FMA over unpacked nibbles
 *                 (laguna's default; the accurate path).
 *   MOE_Q4_IDOT — activations quantized to int8 per 32-block, int8xint4 dot via
 *                 AVX2/VNNI (inkling's default; fast, lossy).
 * Packing is the shared container: nibbles low=even col / high=odd col, value =
 * nibble-8, one f32 scale per output row. */
#ifndef MOE_QUANT_H
#define MOE_QUANT_H
#include <stdint.h>
#include <math.h>
#include <string.h>            /* memcpy: ue8m0 bit-pattern decode */
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

enum { MOE_Q4_F32 = 0, MOE_Q4_IDOT = 1 };
enum { MOE_Q8_F32 = 0, MOE_Q8_IDOT = 1 };

#if defined(__AVX2__)
/* int8·int8 -> int32 dot accumulate over a 32-lane block, sign-folded so the
 * unsigned-times-signed VNNI op is exact for signed a. */
static inline __m256i i8dot_block(__m256i acc, __m256i a, __m256i b) {
    __m256i ax = _mm256_sign_epi8(a, a);        /* |a| as u8 */
    __m256i sy = _mm256_sign_epi8(b, a);        /* b * sign(a) */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#else
    __m256i p = _mm256_maddubs_epi16(ax, sy);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
#endif
}
#endif

/* y[O] = x[I] @ dequant(packed)^T; packed [O,I/2] int4 (nibble-8) * scale[o].
 * mode selects the activation contract; exact forces the double-accumulate
 * scalar reference (only consulted on the MOE_Q4_F32 path). */
static void matmul_q4_k(float *y, const float *x, const uint8_t *packed, const float *scale,
                        int I, int O, int mode, int exact) {
#if defined(__AVX2__)
    if (mode == MOE_Q4_IDOT && I % 32 == 0 && I <= 8192) {
        /* per-32-block int8 activation quant, then int8xint4 integer dot */
        int nb = I / 32;
        int8_t xi[8192]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b * 32;
            float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am / 127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f / s;
            for (int i = 0; i < 32; i++) xi[b * 32 + i] = (int8_t)lrintf(xb[i] * inv);
        }
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m256i b8 = _mm256_set1_epi8(8);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w = packed + (int64_t)o * (I / 2);
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                __m128i by = _mm_loadu_si128((const __m128i*)(w + b * 16));  /* 16 B = 32 nibbles */
                __m128i lo = _mm_and_si128(by, m4);                         /* even columns */
                __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);      /* odd columns  */
                __m256i nib = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi),   /* cols 16..31 */
                                               _mm_unpacklo_epi8(lo, hi));  /* cols  0..15 */
                nib = _mm256_sub_epi8(nib, b8);
                __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                           _mm256_loadu_si256((const __m256i*)(xi + b * 32)), nib);
                __m128i l = _mm256_castsi256_si128(vacc), h = _mm256_extracti128_si256(vacc, 1);
                __m128i s4 = _mm_add_epi32(l, h);
                s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                acc += xs[b] * (float)_mm_cvtsi128_si32(s4);
            }
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
#if defined(__AVX512F__)
    if (mode == MOE_Q4_F32 && !exact) {
        #pragma omp parallel for schedule(static) if(O >= 512)   /* single-row kernel: O is the whole work */
        for (int o = 0; o < O; o++) {
            const uint8_t *p = packed + (int64_t)o * (I / 2);
            __m512 acc = _mm512_setzero_ps();
            const __m128i m0f = _mm_set1_epi8(0x0F);
            const __m512 v8 = _mm512_set1_ps(8.f);
            int c = 0;
            for (; c + 16 <= I / 2; c += 16) {                 /* 16 bytes -> 32 weights */
                __m128i b  = _mm_loadu_si128((const __m128i*)(p + c));
                __m128i lo = _mm_and_si128(b, m0f);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
                __m128i il0 = _mm_unpacklo_epi8(lo, hi);        /* x[2c..2c+15] order */
                __m128i il1 = _mm_unpackhi_epi8(lo, hi);        /* x[2c+16..2c+31] order */
                __m512 f0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(il0)), v8);
                __m512 f1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(il1)), v8);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + 2 * c), f0, acc);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + 2 * c + 16), f1, acc);
            }
            float s = _mm512_reduce_add_ps(acc);
            for (; c < I / 2; c++) { uint8_t b = p[c]; s += x[2 * c] * ((int)(b & 0xF) - 8) + x[2 * c + 1] * ((int)(b >> 4) - 8); }
            y[o] = s * scale[o];
        }
        return;
    }
#endif
    /* portable double-accumulate reference (laguna's exact path; also the fallback
     * for the f32 contract when AVX-512 is absent). More accurate than a float
     * accumulate — inkling's old IDOT=0 debug path converges here, harmlessly, as
     * its real path is always MOE_Q4_IDOT. */
    #pragma omp parallel for schedule(static) if(O >= 512)   /* single-row kernel: O is the whole work */
    for (int o = 0; o < O; o++) {
        const uint8_t *p = packed + (int64_t)o * (I / 2);
        double s = 0;
        for (int c = 0; c < I / 2; c++) {
            uint8_t b = p[c];
            s += (double)x[2 * c] * ((int)(b & 0xF) - 8) + (double)x[2 * c + 1] * ((int)(b >> 4) - 8);
        }
        y[o] = (float)(s * scale[o]);
    }
}

/* Batched int4: y[S,O] = x[S,I] @ dequant(packed)^T, reading each weight ROW ONCE
 * for all S activation rows. This is the group-by-expert lever: an expert's int4
 * blob streams from RAM (or VRAM) once and serves every token routed to it, so a
 * group of n tokens costs one weight read instead of n.
 *
 * Every (o,s) output reduces independently, in the same order as the S=1 kernel,
 * so each row is BIT-IDENTICAL to matmul_q4_k — grouping changes bandwidth, not
 * numerics. The fast paths unpack the weight row once (into an L1-resident buffer)
 * and reuse it across the S rows, which also removes the repeated nibble unpack;
 * they need I%32==0 (laguna: D=3072, I=1024) and defer to the per-row loop
 * otherwise, same result either way. */
static void matmul_q4_kb(float *y, const float *x, const uint8_t *packed, const float *scale,
                         int S, int I, int O, int mode, int exact) {
    if (S == 1) { matmul_q4_k(y, x, packed, scale, I, O, mode, exact); return; }
#if defined(__AVX2__)
    if (mode == MOE_Q4_IDOT && I % 32 == 0 && I <= 8192) {
        int nb = I / 32;
        int8_t *xi = (int8_t*)malloc((size_t)S * I);
        float  *xs = (float*)malloc((size_t)S * nb * sizeof(float));
        for (int s = 0; s < S; s++) {   /* per-row int8 quant, identical to the S=1 path */
            const float *xr = x + (int64_t)s * I;
            for (int b = 0; b < nb; b++) {
                const float *xb = xr + b * 32;
                float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
                float sc = am / 127.f; if (sc < 1e-12f) sc = 1e-12f;
                xs[(int64_t)s * nb + b] = sc; float inv = 1.f / sc;
                for (int i = 0; i < 32; i++) xi[(int64_t)s * I + b * 32 + i] = (int8_t)lrintf(xb[i] * inv);
            }
        }
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m256i b8 = _mm256_set1_epi8(8);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w = packed + (int64_t)o * (I / 2);
            int8_t wq[8192];                                  /* the row's nibbles, unpacked once */
            for (int b = 0; b < nb; b++) {
                __m128i by = _mm_loadu_si128((const __m128i*)(w + b * 16));
                __m128i lo = _mm_and_si128(by, m4);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                __m256i nib = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
                _mm256_storeu_si256((__m256i*)(wq + b * 32), _mm256_sub_epi8(nib, b8));
            }
            for (int s = 0; s < S; s++) {
                float acc = 0.f;
                for (int b = 0; b < nb; b++) {
                    __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                               _mm256_loadu_si256((const __m256i*)(xi + (int64_t)s * I + b * 32)),
                                               _mm256_loadu_si256((const __m256i*)(wq + b * 32)));
                    __m128i l = _mm256_castsi256_si128(vacc), h = _mm256_extracti128_si256(vacc, 1);
                    __m128i s4 = _mm_add_epi32(l, h);
                    s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                    acc += xs[(int64_t)s * nb + b] * (float)_mm_cvtsi128_si32(s4);
                }
                y[(int64_t)s * O + o] = acc * scale[o];
            }
        }
        free(xi); free(xs);
        return;
    }
#endif
#if defined(__AVX512F__)
    if (mode == MOE_Q4_F32 && !exact && I % 32 == 0 && I <= 8192) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *p = packed + (int64_t)o * (I / 2);
            const __m128i m0f = _mm_set1_epi8(0x0F);
            const __m512 v8 = _mm512_set1_ps(8.f);
            float wf[8192];                                   /* dequantized row, reused for all S */
            for (int c = 0; c < I / 2; c += 16) {              /* 16 bytes -> 32 weights */
                __m128i b  = _mm_loadu_si128((const __m128i*)(p + c));
                __m128i lo = _mm_and_si128(b, m0f);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
                _mm512_storeu_ps(wf + 2 * c,
                    _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi))), v8));
                _mm512_storeu_ps(wf + 2 * c + 16,
                    _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(lo, hi))), v8));
            }
            for (int s = 0; s < S; s++) {
                const float *xr = x + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                for (int i = 0; i < I; i += 16)                /* same FMA order as the S=1 kernel */
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(xr + i), _mm512_loadu_ps(wf + i), acc);
                y[(int64_t)s * O + o] = _mm512_reduce_add_ps(acc) * scale[o];
            }
        }
        return;
    }
#endif
    for (int s = 0; s < S; s++)
        matmul_q4_k(y + (int64_t)s * O, x + (int64_t)s * I, packed, scale, I, O, mode, exact);
}

/* ---------- fp4 (e2m1) with per-32-block ue8m0 scales ----------
 *
 * DeepSeek-V4-Flash ships its routed experts in this format natively, so the
 * sabrewing container stores them VERBATIM: converting them to the int4 container
 * above would be a lossy re-grid, not a repack. Measured on real V4-Flash experts
 * (layer 0, w1/w2/w3), taking the fp4 weights as ground truth:
 *
 *     int4 per-row scale   17.0% relative error   (the int4 container's shape)
 *     int4 group-128       11.8%
 *     int4 group-32         9.7%                  (same scale granularity!)
 *     fp4 verbatim          0%
 *
 * The loss is not dynamic range — the per-row ue8m0 exponent spread is only ~1
 * power of 2. It is that e2m1's levels are NON-UNIFORM ({0,.5,1,1.5,2,3,4,6}),
 * so re-gridding them onto int4's uniform ladder quantizes twice. Keeping the
 * source format is both lossless and a pure byte repack at convert time.
 *
 * Layout matches the int4 container's nibble convention exactly — low nibble =
 * even column, high nibble = odd column — so only the value mapping differs:
 * int4 is (nibble - 8), fp4 is LUT[nibble] * 2^(es[block] - 127).
 *
 *   packed [O, I/2]  u8   two e2m1 codes per byte
 *   es     [O, I/32] u8   ue8m0 biased exponent, one per 32 input columns
 */
static const float moe_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

/* 2^(e-127) for a ue8m0 byte. e=0xFF is the format's NaN; it never appears in a
 * weight file, and mapping it to 0 keeps a corrupt file from poisoning the sum. */
static inline float moe_ue8m0(uint8_t e) {
    if (e == 0xFF) return 0.f;
    if (e == 0)    return 5.8774717541114375e-39f;  /* 2^-127: denormal, not (0<<23) */
    uint32_t u = (uint32_t)e << 23;                 /* exponent field, mantissa 0 */
    float f; memcpy(&f, &u, 4); return f;
}

/* y[O] = x[I] @ dequant(packed, es)^T. `exact` forces the double-accumulate
 * reference, which is the oracle contract the f32 path is compared against. */
static void matmul_fp4_k(float *y, const float *x, const uint8_t *packed, const uint8_t *es,
                         int I, int O, int exact) {
#if defined(__AVX512F__)
    if (!exact && I % 32 == 0) {
        const __m512 lut = _mm512_loadu_ps(moe_fp4_lut);
        #pragma omp parallel for schedule(static) if(O >= 512)
        for (int o = 0; o < O; o++) {
            const uint8_t *p = packed + (int64_t)o * (I / 2);
            const uint8_t *e = es + (int64_t)o * (I / 32);
            __m512 acc = _mm512_setzero_ps();
            const __m128i m0f = _mm_set1_epi8(0x0F);
            /* 16 bytes = 32 nibbles = 32 input columns = EXACTLY one scale block */
            for (int c = 0, b = 0; c + 16 <= I / 2; c += 16, b++) {
                __m128i by = _mm_loadu_si128((const __m128i*)(p + c));
                __m128i lo = _mm_and_si128(by, m0f);                    /* even columns */
                __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m0f); /* odd columns  */
                __m128i il0 = _mm_unpacklo_epi8(lo, hi);                /* cols 2c..2c+15 */
                __m128i il1 = _mm_unpackhi_epi8(lo, hi);                /* cols 2c+16..2c+31 */
                __m512 f0 = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(il0), lut);
                __m512 f1 = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(il1), lut);
                __m512 bs = _mm512_set1_ps(moe_ue8m0(e[b]));
                acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(x + 2 * c), bs), f0, acc);
                acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(x + 2 * c + 16), bs), f1, acc);
            }
            y[o] = _mm512_reduce_add_ps(acc);
        }
        return;
    }
#endif
    /* portable double-accumulate reference */
    #pragma omp parallel for schedule(static) if(O >= 512)
    for (int o = 0; o < O; o++) {
        const uint8_t *p = packed + (int64_t)o * (I / 2);
        const uint8_t *e = es + (int64_t)o * (I / 32);
        double s = 0;
        for (int c = 0; c < I / 2; c++) {
            uint8_t b = p[c];
            double sc = moe_ue8m0(e[(2 * c) / 32]);
            s += ((double)x[2 * c] * moe_fp4_lut[b & 0xF]
                + (double)x[2 * c + 1] * moe_fp4_lut[b >> 4]) * sc;
        }
        y[o] = (float)s;
    }
}

/* Batched fp4: y[S,O] = x[S,I] @ dequant(...)^T, reading each weight row once for
 * all S activation rows — the group-by-expert lever, same contract as
 * matmul_q4_kb. Each (o,s) reduces in the same order as the S=1 kernel above, so
 * grouping changes bandwidth, not numerics. */
static void matmul_fp4_kb(float *y, const float *x, const uint8_t *packed, const uint8_t *es,
                          int S, int I, int O, int exact) {
#if defined(__AVX512F__)
    if (!exact && I % 32 == 0 && S > 1) {
        const __m512 lut = _mm512_loadu_ps(moe_fp4_lut);
        #pragma omp parallel for schedule(static) if(O >= 64)
        for (int o = 0; o < O; o++) {
            const uint8_t *p = packed + (int64_t)o * (I / 2);
            const uint8_t *e = es + (int64_t)o * (I / 32);
            for (int s = 0; s < S; s++) {
                const float *xs = x + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                const __m128i m0f = _mm_set1_epi8(0x0F);
                for (int c = 0, b = 0; c + 16 <= I / 2; c += 16, b++) {
                    __m128i by = _mm_loadu_si128((const __m128i*)(p + c));
                    __m128i lo = _mm_and_si128(by, m0f);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m0f);
                    __m512 f0 = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(lo, hi)), lut);
                    __m512 f1 = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(lo, hi)), lut);
                    __m512 bs = _mm512_set1_ps(moe_ue8m0(e[b]));
                    acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(xs + 2 * c), bs), f0, acc);
                    acc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(xs + 2 * c + 16), bs), f1, acc);
                }
                y[(int64_t)s * O + o] = _mm512_reduce_add_ps(acc);
            }
        }
        return;
    }
#endif
    for (int s = 0; s < S; s++)
        matmul_fp4_k(y + (int64_t)s * O, x + (int64_t)s * I, packed, es, I, O, exact);
}

/* y[O] = x[I] @ dequant(q)^T; q = signed int8 [O,I], per-row f32 scale[o] (given
 * as a SEPARATE pointer — laguna's embedded [int8 O*I][f32 O] buffer splits into
 * (q, q+I*O) at the wrapper). mode selects the activation contract, like q4:
 *   MOE_Q8_F32  — activations f32, AVX-512 cvtepi8->f32 FMA (laguna's default).
 *   MOE_Q8_IDOT — activations int8 per-32-block, VNNI dot (inkling's default). */
static void matmul_q8_k(float *y, const float *x, const int8_t *q, const float *scale,
                        int I, int O, int mode, int exact) {
#if defined(__AVX2__)
    if (mode == MOE_Q8_IDOT && I % 32 == 0 && I <= 8192) {
        int nb = I / 32;
        int8_t xi[8192]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b * 32;
            float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am / 127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f / s;
            for (int i = 0; i < 32; i++) xi[b * 32 + i] = (int8_t)lrintf(xb[i] * inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                           _mm256_loadu_si256((const __m256i*)(xi + b * 32)),
                                           _mm256_loadu_si256((const __m256i*)(w + b * 32)));
                __m128i lo = _mm256_castsi256_si128(vacc), hi = _mm256_extracti128_si256(vacc, 1);
                __m128i s4 = _mm_add_epi32(lo, hi);
                s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                acc += xs[b] * (float)_mm_cvtsi128_si32(s4);
            }
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
#if defined(__AVX512F__)
    if (mode == MOE_Q8_F32 && !exact) {
        #pragma omp parallel for schedule(static) if(O >= 512)   /* single-row kernel: O is the whole work */
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 16 <= I; i += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + i),
                        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(w + i)))), acc);
            float s = _mm512_reduce_add_ps(acc);
            for (; i < I; i++) s += x[i] * w[i];
            y[o] = s * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if(O >= 512)   /* single-row kernel: O is the whole work */
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        double s = 0; for (int i = 0; i < I; i++) s += (double)x[i] * w[i];
        y[o] = (float)(s * scale[o]);
    }
}

#endif
