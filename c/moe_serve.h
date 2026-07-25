/* moe_serve.h — shared serve-protocol front end for the MoE engines.
 * Phase 1 of the MoE-runtime refactor (docs/moe-runtime-plan.md).
 *
 * The openai_server.py gateway drives every engine over the same stdin/stdout
 * protocol, so the request struct, the bounded submit queue, the non-blocking
 * stdin poll, and the SUBMIT/CANCEL line parser are identical across engines and
 * live here. The per-request generation loop (serve_one) and the READY/STAT
 * banner loop (serve_loop) stay per-engine — they touch the engine's Model,
 * forward() and KV cache. Oracle-neutral (pure I/O).
 *
 * Protocol:
 *   stdin:  SUBMIT <id> <slot> <len> <max_tokens> <temp> <top_p>\n<payload>\n
 *           CANCEL <id>\n
 *   stdout: READY sentinel once loaded, then per request DATA <id> <n>\n<bytes>\n
 *           frames and a final DONE <id> ... line. */
#ifndef MOE_SERVE_H
#define MOE_SERVE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } SReq;
#define SRV_QMAX 16
static SReq g_q[SRV_QMAX]; static int g_qn = 0;

/* Every CANCEL id is also recorded here. serve_read_cmd's return value can only
 * answer "was this a cancel for cur_id", which is enough for a one-request-at-a-time
 * engine; an engine that keeps SEVERAL requests in flight (continuous batching) needs
 * to know WHICH id was cancelled, so it drains this ring instead. Bounded, and full
 * means the OLDEST is dropped: entries only linger when they name a request that
 * never runs (a cancel that raced its own DONE), and evicting the stale end keeps a
 * long-lived server from wedging the ring shut against real cancels. */
#define SRV_CANCELS 32
static char g_cancels[SRV_CANCELS][64]; static int g_ncancel = 0;
/* consume a pending cancel for `id` (1 = it was cancelled) */
static int serve_take_cancel(const char *id) {
    for (int i = 0; i < g_ncancel; i++) {
        if (strcmp(g_cancels[i], id)) continue;
        memmove(g_cancels + i, g_cancels + i + 1, (size_t)(--g_ncancel - i) * sizeof(g_cancels[0]));
        return 1;
    }
    return 0;
}

/* non-blocking: is a full line waiting on stdin? (drives mid-generation CANCEL) */
static int stdin_readable(void) {
    fd_set r; struct timeval tv = {0, 0};
    FD_ZERO(&r); FD_SET(0, &r);
    return select(1, &r, NULL, NULL, &tv) > 0;
}

/* Read one protocol line. Returns -1 on EOF/short read, 1 if it is a CANCEL for
 * cur_id (the request being generated), 0 otherwise (SUBMIT enqueued, or a line
 * to ignore). SUBMIT reads its <len>-byte payload and queues an SReq. */
static int serve_read_cmd(const char *cur_id) {
    char ln[512];
    if (!fgets(ln, sizeof(ln), stdin)) return -1;
    char cmd[16], id[64];
    if (sscanf(ln, "%15s %63s", cmd, id) < 2) return 0;
    if (!strcmp(cmd, "CANCEL")) {
        int mine = cur_id && !strcmp(id, cur_id);      /* the caller acts on it itself */
        if (!mine) {
            if (g_ncancel == SRV_CANCELS)              /* full: drop the oldest */
                memmove(g_cancels, g_cancels + 1, (size_t)(--g_ncancel) * sizeof(g_cancels[0]));
            snprintf(g_cancels[g_ncancel++], sizeof(g_cancels[0]), "%s", id);
        }
        return mine;
    }
    if (!strcmp(cmd, "SUBMIT")) {
        int slot, plen, max_tok; float temp, top_p;
        if (sscanf(ln, "%*s %*s %d %d %d %f %f", &slot, &plen, &max_tok, &temp, &top_p) != 5 ||
            plen < 0 || plen > (1 << 22)) { printf("ERROR %s bad submit header\n", id); fflush(stdout); return 0; }
        (void)slot;
        char *pl = malloc((size_t)plen + 1);
        if (fread(pl, 1, (size_t)plen, stdin) != (size_t)plen) { free(pl); return -1; }
        int nl = fgetc(stdin); (void)nl; pl[plen] = 0;
        if (g_qn < SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", id);
            q->max_tok = max_tok; q->temp = temp; q->top_p = top_p;
            q->payload = pl; q->plen = plen;
        } else { printf("ERROR %s queue full\n", id); fflush(stdout); free(pl); }
    }
    return 0;
}

#endif
