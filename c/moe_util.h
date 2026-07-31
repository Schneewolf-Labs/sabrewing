/* moe_util.h — small shared utilities for the MoE engines.
 * Phase 1 of the MoE-runtime refactor (docs/moe-runtime-plan.md): the last of the
 * byte-identical copy-paste helpers — a float allocator that aborts on OOM, and
 * two config/reference JSON accessors (need json.h's jval). */
#ifndef MOE_UTIL_H
#define MOE_UTIL_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "json.h"

static float *falloc(int64_t n) { float *p = malloc(n * sizeof(float)); if (!p) { fprintf(stderr, "OOM %lld\n", (long long)n); exit(1); } return p; }

/* config number with default: jnum(cfg, "key", fallback) */
static double jnum(jval *o, const char *k, double d) { jval *v = json_get(o, k); return (v && v->t == J_NUM) ? v->num : d; }

/* parse a JSON int array into a malloc'd int[] (n written to *n_out; NULL if absent) */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { *n_out = 0; return NULL; }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

/* Count PHYSICAL cores, not logical CPUs.
 *
 * Why this needs its own helper: OpenMP defaults to nproc, so on an SMT machine
 * every physical core runs two threads that contend for the same core's vector
 * units. On the int4 kernels that is not a small tax — it is a cliff.
 *
 * Counts distinct thread_siblings_list entries. Returns 0 when the topology is
 * unreadable (non-Linux, restricted /sys) so callers fall back to the OpenMP
 * default. */
static int moe_physical_cores(void) {
#ifdef __linux__
    static char seen[512][64];
    int n = 0;
    for (int c = 0; c < 512; c++) {
        char p[128];
        snprintf(p, sizeof p,
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE *f = fopen(p, "r");
        if (!f) break;
        char line[64] = {0};
        if (!fgets(line, sizeof line, f)) { fclose(f); break; }
        fclose(f);
        char *nl = strchr(line, '\n'); if (nl) *nl = 0;
        int dup = 0;
        for (int i = 0; i < n; i++) if (!strcmp(seen[i], line)) { dup = 1; break; }
        if (!dup) { snprintf(seen[n], sizeof seen[0], "%s", line); n++; }
    }
    return n;
#else
    return 0;
#endif
}

/* Clamp the OpenMP team to one thread per physical core, unless the user asked
 * for a specific count. Call early in main(); OpenMP reads the ICV at each
 * parallel-region entry, so this takes effect for everything after it.
 *
 * Measured cost of NOT doing this, Ryzen 9 7900 (12c/24t), one binary per arm,
 * A/B alternated so each slow run sits between two fast ones:
 *   laguna 118B int4    24 thr 0.14 tok/s    12 thr 3.99-4.04 tok/s   (~28x)
 *   inkling-small int4  24 thr 0.26-0.29     12 thr 2.74-3.02         (~10x)
 * Prefill moves with it (laguna 43.3s -> 1.1s; inkling 8-15s -> 1.1s).
 *
 * This is not a page-cache artifact: every run above reported fill 0.0s and
 * cache hit 100.0%, i.e. zero disk reads in BOTH arms, and all three compute
 * phases scale together (expert-mm 13-18x, shared 7-9x, attn 10-13x). The
 * 24-thread arm also degrades across repeats while the 12-thread arm stays
 * flat — the opposite of a warming curve. */
static void moe_omp_autotune(void) {
    if (getenv("OMP_NUM_THREADS")) return;          /* user knows better */
    if (getenv("MOE_NO_OMP_AUTOTUNE")) return;      /* escape hatch */
    int phys = moe_physical_cores();
#ifdef _OPENMP
    if (phys > 0) omp_set_num_threads(phys);
#else
    (void)phys;
#endif
}

#endif
