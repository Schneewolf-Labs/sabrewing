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
        #pragma omp parallel for schedule(static) if(O >= 512)
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
    #pragma omp parallel for schedule(static) if(O >= 512)
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
        #pragma omp parallel for schedule(static) if(O >= 512)
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
    #pragma omp parallel for schedule(static) if(O >= 512)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        double s = 0; for (int i = 0; i < I; i++) s += (double)x[i] * w[i];
        y[o] = (float)(s * scale[o]);
    }
}

#endif
