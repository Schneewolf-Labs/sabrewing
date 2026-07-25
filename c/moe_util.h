/* moe_util.h — small shared utilities for the MoE engines.
 * Phase 1 of the MoE-runtime refactor (docs/moe-runtime-plan.md): the last of the
 * byte-identical copy-paste helpers — a float allocator that aborts on OOM, and
 * two config/reference JSON accessors (need json.h's jval). */
#ifndef MOE_UTIL_H
#define MOE_UTIL_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

#endif
