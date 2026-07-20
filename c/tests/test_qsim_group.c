/* Tests for qsim_row's group-scale path (QSIM_GROUP, docs/heat-quant-design.md).
 *
 * Why this test exists: the heat-tiered quant probe re-quantizes packed-int4
 * expert rows to 2/3-bit in place, and the group-scale variant is the quality
 * lever the whole sub-int4 plan leans on (int2-g64 measured ~0.43 rel-err vs
 * ~0.55 per-row). The probe's numbers are only meaningful if the degradation
 * math is exactly what the analysis assumed — a wrong group boundary or a
 * missed partial tail silently shifts the measured damage and looks like a
 * model property instead of a bug.
 *
 * Covers: gb=0 (per-row), gb=nb (single group == per-row), nb a clean multiple
 * of gb, a partial last group (nb % gb != 0), gb > nb (clamped), the all-zero
 * group skip (mx==0 leaves nibbles untouched), and the qualitative property the
 * design doc claims: on a row whose halves differ in magnitude, group scales
 * strictly reduce reconstruction error vs one per-row scale. */
#define main inkling_main_unused
#include "../inkling.c"
#undef main

static uint32_t rng_state = 0xC0FFEEu;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }

/* Independent scalar reference: same intended semantics, written dumb. */
static void ref_qsim(const uint8_t *in, uint8_t *out, int64_t rows, int64_t nb, int bits, int64_t gb) {
    int lv = 1 << (bits - 1);
    if (gb <= 0 || gb > nb) gb = nb;
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *src = in + r*nb;
        uint8_t *dst = out + r*nb;
        memcpy(dst, src, (size_t)nb);
        for (int64_t g = 0; g < nb; g += gb) {
            int64_t ge = g + gb < nb ? g + gb : nb;
            int mx = 0;
            for (int64_t b = g; b < ge; b++) {
                int lo = (src[b]&0xF)-8, hi = (src[b]>>4)-8;
                if (abs(lo) > mx) mx = abs(lo);
                if (abs(hi) > mx) mx = abs(hi);
            }
            if (mx == 0) continue;
            float step = (float)mx / lv;
            for (int64_t b = g; b < ge; b++) {
                int lo = (src[b]&0xF)-8, hi = (src[b]>>4)-8;
                int qlo = lrintf(lo/step); if (qlo < -lv) qlo = -lv; if (qlo > lv-1) qlo = lv-1;
                int qhi = lrintf(hi/step); if (qhi < -lv) qhi = -lv; if (qhi > lv-1) qhi = lv-1;
                int vlo = lrintf(qlo*step)+8; if (vlo < 0) vlo = 0; if (vlo > 15) vlo = 15;
                int vhi = lrintf(qhi*step)+8; if (vhi < 0) vhi = 0; if (vhi > 15) vhi = 15;
                dst[b] = (uint8_t)(vlo | (vhi<<4));
            }
        }
    }
}

/* Mean |reconstructed - original| over the int4 integer grid. */
static double mae_vs(const uint8_t *a, const uint8_t *b, int64_t nb) {
    double s = 0;
    for (int64_t i = 0; i < nb; i++) {
        s += abs(((a[i]&0xF)-8) - ((b[i]&0xF)-8));
        s += abs(((a[i]>>4)-8) - ((b[i]>>4)-8));
    }
    return s / (double)(2*nb);
}

int main(void) {
    int fails = 0;

    /* 1) byte-exact vs the reference across group shapes, both bit widths */
    for (int bits = 2; bits <= 3; bits++) {
        int64_t nbs[] = {16, 10};                   /* 10 % 4 = partial last group */
        int64_t gbs[] = {0, 4, 5, 7, 10, 16, 100};  /* 0 = per-row, 100 > nb = clamp */
        for (size_t ni = 0; ni < sizeof nbs/sizeof *nbs; ni++)
            for (size_t gi = 0; gi < sizeof gbs/sizeof *gbs; gi++) {
                int64_t nb = nbs[ni], gb = gbs[gi], rows = 3;
                uint8_t in[3*16], got[3*16], want[3*16];
                for (int64_t i = 0; i < rows*nb; i++) in[i] = (uint8_t)(xr() & 0xFF);
                memcpy(got, in, (size_t)(rows*nb));
                qsim_row(got, rows, nb, bits, gb);
                ref_qsim(in, want, rows, nb, bits, gb);
                if (memcmp(got, want, (size_t)(rows*nb)) != 0) {
                    fprintf(stderr, "FAIL: mismatch bits=%d nb=%lld gb=%lld\n",
                            bits, (long long)nb, (long long)gb);
                    fails++;
                }
            }
    }

    /* 2) gb = nb must equal gb = 0 (one group per row IS the per-row path) */
    {
        uint8_t a[24], b[24];
        for (int i = 0; i < 24; i++) a[i] = b[i] = (uint8_t)(xr() & 0xFF);
        qsim_row(a, 2, 12, 2, 0);
        qsim_row(b, 2, 12, 2, 12);
        if (memcmp(a, b, 24) != 0) { fprintf(stderr, "FAIL: gb=nb != per-row\n"); fails++; }
    }

    /* 3) all-zero group stays untouched (mx==0 skip); nibble 8 encodes value 0 */
    {
        uint8_t row[8] = {0x88,0x88,0x88,0x88, 0xF0,0x1F,0xA5,0x3C};
        qsim_row(row, 1, 8, 2, 4);
        for (int i = 0; i < 4; i++)
            if (row[i] != 0x88) { fprintf(stderr, "FAIL: zero group modified\n"); fails++; break; }
    }

    /* 4) the design-doc claim: mixed-magnitude row -> group scales strictly beat
     *    per-row. First half tiny (|q| <= 1), second half full-range (|q| <= 7):
     *    one per-row int2 step (7/2) wipes the tiny half; a per-group step keeps it. */
    {
        uint8_t orig[16], pr[16], gr[16];
        for (int i = 0; i < 8; i++)  orig[i] = (uint8_t)(((7 + (int)(xr()%3)-1) << 4) | (7 + (int)(xr()%3)-1));
        for (int i = 8; i < 16; i++) orig[i] = (uint8_t)((xr()%16) << 4 | (xr()%16));
        orig[15] = 0xF1;                            /* force full range in half 2 */
        memcpy(pr, orig, 16); memcpy(gr, orig, 16);
        qsim_row(pr, 1, 16, 2, 0);
        qsim_row(gr, 1, 16, 2, 8);
        double e_row = mae_vs(pr, orig, 16), e_grp = mae_vs(gr, orig, 16);
        if (!(e_grp < e_row)) {
            fprintf(stderr, "FAIL: group scales did not help (row %.3f grp %.3f)\n", e_row, e_grp);
            fails++;
        }
    }

    if (fails) { fprintf(stderr, "test_qsim_group: %d FAILURE(S)\n", fails); return 1; }
    printf("test_qsim_group: all passed\n");
    return 0;
}
