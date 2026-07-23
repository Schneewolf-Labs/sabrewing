/* Indicizzazione e lettura on-demand di tensori da piu' file safetensors.
 * Equivale a Shards in engine.py, ma:
 *   - legge con pread (niente mmap) + posix_fadvise(DONTNEED) -> le pagine NON
 *     restano residenti nel processo. E' la correzione del bug di RSS: cosi' la
 *     RAM di picco resta densa+cache, non l'intero modello. (vedi memoria mmap-rss-bug)
 *   - converte sempre in float32 in uscita (BF16/F16/F32 supportati). */
#ifndef ST_H
#define ST_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include "json.h"
#include "compat.h"

/* tetto sulla dimensione dell'header safetensors: gli header reali sono piccoli
 * (KB..pochi MB). Un file crafted che dichiara un hlen enorme causerebbe una
 * malloc gigante prima ancora di leggere: lo respingiamo. */
#define ST_MAX_HEADER (512ll << 20)

typedef struct {
    char   *name;
    int     fd;
    int64_t off;       /* offset assoluto del dato dentro al file */
    int64_t nbytes;
    int     dtype;     /* 0=BF16 1=F16 2=F32 */
    int64_t numel;
} st_tensor;

typedef struct {
    st_tensor *t;
    int        n, cap;
    int        fds[512];
    int        dfds[512];  /* gemelli O_DIRECT (aperti pigramente): -2 = non ancora provato */
    char      *paths[512];
    int        nfd;
    int        mfds[512];  /* MIRROR: fds of the second model copy (dual-SSD), -1 = absent */
    int        mdfds[512]; /* O_DIRECT twins of the second copy, -1 = absent */
    int        nmirror;    /* files accepted into the mirror (0 = mirror inactive) */
    int       *hidx;      /* hash map nome->indice (open addressing): con ~120k tensori
                           * (GLM: 256 expert x 78 layer x 3 x 2) la scansione lineare
                           * costava decine di secondi/token (misurato sul primo run reale) */
    int        hcap;
} shards;
#define ST_MAX_SHARDS 512

static uint64_t st_hash(const char *s){
    uint64_t h=1469598103934665603ULL;
    while(*s){ h^=(unsigned char)*s++; h*=1099511628211ULL; }
    return h;
}

static int st_dtype_code(const char *s) {
    if (!strcmp(s, "BF16")) return 0;
    if (!strcmp(s, "F16"))  return 1;
    if (!strcmp(s, "F32"))  return 2;
    if (!strcmp(s, "U8"))   return 3;   /* dati quantizzati (int4 packed / int8) */
    if (!strcmp(s, "I8"))   return 3;
    fprintf(stderr, "unsupported dtype: %s\n", s); exit(1);
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {            /* subnormale o zero */
        if (man == 0) u = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; u = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {  /* inf/nan */
        u = sign | 0x7F800000 | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

static int st_open_fd(shards *S, const char *path) {
    for (int i = 0; i < S->nfd; i++) if (!strcmp(S->paths[i], path)) return S->fds[i];
    int fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    S->paths[S->nfd] = strdup(path); S->fds[S->nfd] = fd;
#ifdef O_DIRECT
    S->dfds[S->nfd] = open(path, COMPAT_O_RDONLY | O_DIRECT);   /* eager: lookup poi thread-safe */
#elif defined(__APPLE__) || defined(_WIN32)
    S->dfds[S->nfd] = compat_open_direct(path);          /* macOS: F_NOCACHE; Windows: NO_BUFFERING */
#else
    S->dfds[S->nfd] = -1;                                /* niente equivalente: solo buffered */
#endif
    S->nfd++;
    return fd;
}

/* fd gemello O_DIRECT dello stesso file (bypassa la page cache: il buffered read su
 * ext4-in-VHDX si strozza a ~0.8 GB/s, O_DIRECT arriva a 2.3+; misurato). -1 se non disponibile. */
static int st_fidx(shards *S, int fd) {
    for (int i = 0; i < S->nfd; i++) if (S->fds[i] == fd) return i;
    return -1;
}
static int st_direct_fd(shards *S, int fd) {
    int i = st_fidx(S, fd); return i < 0 ? -1 : S->dfds[i];
}

/* ---- MIRROR (dual-SSD): second read-only copy of the model on another drive ----
 * st_fd_rep/st_direct_fd_rep: fd of replica `rep` (0 = primary, 1 = mirror) for
 * the SAME file identified by its primary fd. -1 if that replica is absent. */
static int st_fd_rep(shards *S, int fd, int rep) {
    if (!rep) return fd;
    if (!S->nmirror) return -1;
    int i = st_fidx(S, fd); return i < 0 ? -1 : S->mfds[i];
}
static int st_direct_fd_rep(shards *S, int fd, int rep) {
    if (!rep) return st_direct_fd(S, fd);
    if (!S->nmirror) return -1;
    int i = st_fidx(S, fd); return i < 0 ? -1 : S->mdfds[i];
}

/* Registers <dir>/<basename> as a read replica of every already-indexed shard.
 * A file is accepted ONLY if its size and safetensors header are byte-identical
 * to the primary: the data_offsets then match by construction, so every pread
 * is valid on either copy. Missing or divergent files simply stay on the
 * primary (the mirror may be partial, e.g. a smaller SSD holding only the
 * expert shards). Returns the number of accepted files. The mirror is NEVER
 * written to: .coli_usage/.coli_kv keep deriving from the primary alone. */
static int st_mirror_init(shards *S, const char *dir) {
    if (S->nmirror) for (int i = 0; i < S->nfd; i++) {   /* re-init: drop the old replica */
        if (S->mfds[i] >= 0) close(S->mfds[i]);
        if (S->mdfds[i] >= 0) close(S->mdfds[i]);
    }
    for (int i = 0; i < ST_MAX_SHARDS; i++) { S->mfds[i] = -1; S->mdfds[i] = -1; }
    S->nmirror = 0;
    for (int i = 0; i < S->nfd; i++) {
        const char *base = strrchr(S->paths[i], '/');
#ifdef _WIN32
        const char *b2 = strrchr(S->paths[i], '\\');
        if (b2 && (!base || b2 > base)) base = b2;
#endif
        base = base ? base + 1 : S->paths[i];
        char mp[2048]; snprintf(mp, sizeof(mp), "%s/%s", dir, base);
        int mfd = open(mp, COMPAT_O_RDONLY);
        if (mfd < 0) continue;               /* partial mirror: this shard stays on the primary */
        int64_t sza = lseek(S->fds[i], 0, SEEK_END), szb = lseek(mfd, 0, SEEK_END);
        if (sza != szb) {
            fprintf(stderr, "[MIRROR] %s: size differs from the primary copy — file skipped\n", mp);
            close(mfd); continue;
        }
        uint64_t ha = 0, hb = 0; int ok = 1;   /* identical header => identical data_offsets */
        if (pread(S->fds[i], &ha, 8, 0) != 8 || pread(mfd, &hb, 8, 0) != 8 ||
            ha != hb || ha == 0 || ha > (uint64_t)256 << 20 || (int64_t)(8 + ha) > sza) ok = 0;
        if (ok) {
            char *ba = malloc(ha), *bb = malloc(ha);
            if (!ba || !bb || pread(S->fds[i], ba, ha, 8) != (ssize_t)ha ||
                pread(mfd, bb, ha, 8) != (ssize_t)ha || memcmp(ba, bb, ha)) ok = 0;
            free(ba); free(bb);
        }
        if (!ok) {
            fprintf(stderr, "[MIRROR] %s: header differs from the primary copy — file skipped\n", mp);
            close(mfd); continue;
        }
        S->mfds[i] = mfd;
#ifdef O_DIRECT
        S->mdfds[i] = open(mp, COMPAT_O_RDONLY | O_DIRECT);
#elif defined(__APPLE__)
        S->mdfds[i] = compat_open_direct(mp);
#endif
        S->nmirror++;
    }
    return S->nmirror;
}

/* indicizza tutti i model-*.safetensors in snap_dir */
/* pread completo: chunk-loop (una singola pread si ferma a ~2^31 byte su Linux
 * — i tensori bf16 grandi la superano), riprova su EINTR e riporta un errore
 * ONESTO: perror stampava "Success" su una short-read (errno resta 0), lo
 * stesso sintomo corretto in glm.c per #236. ST_PREAD_CHUNK e' sovrascrivibile
 * per i test. EN: full pread — chunk loop (one pread caps at ~2^31 bytes and
 * big bf16 tensors exceed it), EINTR retry, honest short-read errors.
 * Exits on failure, like every st.h reader. */
#ifndef ST_PREAD_CHUNK
#define ST_PREAD_CHUNK (1u << 30)
#endif
static void st_pread_full(int fd, void *buf, int64_t n, int64_t off, const char *tag) {
    char *p = (char *)buf;
    int64_t got = 0;
    while (got < n) {
        int64_t want = n - got;
        if (want > (int64_t)ST_PREAD_CHUNK) want = ST_PREAD_CHUNK;
        ssize_t r = pread(fd, p + got, (size_t)want, off + got);
        if (r < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "%s: %s (off %lld, %lld/%lld bytes)\n", tag, strerror(errno),
                    (long long)off, (long long)got, (long long)n);
            exit(1);
        }
        if (r == 0) {
            fprintf(stderr, "%s: short read at EOF (off %lld, %lld/%lld bytes) — truncated file?\n",
                    tag, (long long)off, (long long)got, (long long)n);
            exit(1);
        }
        got += r;
    }
}

/* Scan one directory for *.safetensors shards, appending to files[] (dedup by
 * basename, so a list of directories acts as a SEARCH PATH: the same shard
 * present on two drives is taken from the first-listed one only). *added
 * returns how many shards this dir contributed. */
static void st_scan_dir(const char *dir, char files[][1024], int *nf, int *added) {
    DIR *d = opendir(dir); struct dirent *e;
    if (!d) { perror(dir); exit(1); }
    int base_n = *nf;
    while ((e = readdir(d))) {
        const char *dot = strrchr(e->d_name, '.');
        if (dot && !strcmp(dot, ".safetensors")) {  /* model.safetensors o model-0000N-of-... */
            int dup = 0;
            for (int i = 0; i < *nf; i++) {
                const char *b = strrchr(files[i], '/');
#ifdef _WIN32
                const char *b2 = strrchr(files[i], '\\'); if (b2 && (!b || b2 > b)) b = b2;
#endif
                b = b ? b + 1 : files[i];
                if (!strcmp(b, e->d_name)) { dup = 1; break; }  /* already taken from a higher-priority drive */
            }
            if (dup) continue;
            if (*nf >= ST_MAX_SHARDS) { fprintf(stderr, "too many shards (>%d): raise ST_MAX_SHARDS\n", ST_MAX_SHARDS); exit(1); }
            snprintf(files[(*nf)++], 1024, "%s/%s", dir, e->d_name);
        }
    }
    closedir(d);
    if (added) *added = *nf - base_n;
}

/* Index shards from snap_dir, optionally SPLIT across extra drives listed in
 * extra_dirs (';' or ',' separated). Each shard lives on exactly ONE drive
 * (no duplication — unlike the dual-SSD mirror); a demand pread hits whichever
 * drive holds that shard, so concurrent expert loads parallelise across drives
 * and combined capacity is used. Scales to N drives. Metadata (config /
 * tokenizer / .coli_usage / .coli_kv) is read from snap_dir only. */
static void st_init_multi(shards *S, const char *snap_dir, const char *extra_dirs) {
    memset(S, 0, sizeof(*S));
    S->cap = 4096; S->t = calloc(S->cap, sizeof(st_tensor));
    /* raccoglie ordinatamente i nomi dei file shard */
    static char files[ST_MAX_SHARDS][1024]; int nf = 0;
    int c0 = 0; st_scan_dir(snap_dir, files, &nf, &c0);
    int ndir = 1;
    if (extra_dirs && *extra_dirs) {
        char buf[4096]; snprintf(buf, sizeof(buf), "%s", extra_dirs);
        char *p = buf;
        while (p && *p) {
            char *sep = p; while (*sep && *sep != ';' && *sep != ',') sep++;
            int last = (*sep == 0); *sep = 0;
            while (*p == ' ') p++;
            size_t plen = strlen(p); while (plen > 0 && p[plen-1] == ' ') p[--plen] = 0;
            if (*p) {
                int cN = 0; st_scan_dir(p, files, &nf, &cN);
                fprintf(stderr, "[SPLIT] +%s -> %d shard(s)\n", p, cN);
                ndir++;
            }
            p = last ? NULL : sep + 1;
        }
        fprintf(stderr, "[SPLIT] model across %d dir(s): %d shard(s) total (primary %s -> %d shard(s)), no duplication\n",
                ndir, nf, snap_dir, c0);
    }
    for (int a = 0; a < nf; a++) for (int b = a+1; b < nf; b++)
        if (strcmp(files[a], files[b]) > 0) { char tmp[1024]; strcpy(tmp, files[a]); strcpy(files[a], files[b]); strcpy(files[b], tmp); }

    for (int fi = 0; fi < nf; fi++) {
        int fd = st_open_fd(S, files[fi]);
        struct stat sst;
        if (fstat(fd, &sst) != 0) { perror("fstat shard"); exit(1); }
        int64_t fsz = (int64_t)sst.st_size;
        uint64_t hlen;
        st_pread_full(fd, &hlen, 8, 0, "pread hlen");
        /* file malevolo/troncato: hlen deve stare nel file dopo gli 8 byte di
         * prefisso e sotto il tetto. Senza questo bound hlen+1 puo' andare in
         * overflow (malloc(0) e poi hdr[hlen]=0 fuori limiti) o forzare una
         * malloc gigante. */
        if (fsz < 8 || hlen > (uint64_t)(fsz - 8) || hlen > (uint64_t)ST_MAX_HEADER) {
            fprintf(stderr, "%s: bad safetensors header length %llu (file %lld bytes)\n",
                    files[fi], (unsigned long long)hlen, (long long)fsz); exit(1); }
        char *hdr = malloc(hlen + 1);
        if (!hdr) { perror("malloc safetensors header"); exit(1); }
        st_pread_full(fd, hdr, (int64_t)hlen, 8, "pread hdr");
        hdr[hlen] = 0;
        int64_t data_start = 8 + (int64_t)hlen;
        char *arena = NULL;
        jval *root = json_parse(hdr, &arena);
        if (!root || root->t != J_OBJ) {
            fprintf(stderr, "%s: safetensors header is not a JSON object\n", files[fi]); exit(1); }
        for (int i = 0; i < root->len; i++) {
            const char *name = root->keys[i];
            if (!strcmp(name, "__metadata__")) continue;
            jval *m = root->kids[i];
            jval *dt = json_get(m, "dtype");
            jval *off = json_get(m, "data_offsets");
            jval *shp = json_get(m, "shape");
            /* un header crafted puo' omettere i campi o dare tipi sbagliati:
             * senza questi guard si dereferenzia NULL (json_get) o si legge
             * off->kids[0/1] oltre i limiti dell'array. */
            if (!dt || dt->t != J_STR || !off || off->t != J_ARR || off->len < 2 ||
                !shp || shp->t != J_ARR) {
                fprintf(stderr, "%s: tensor '%s' has malformed dtype/data_offsets/shape\n",
                        files[fi], name); exit(1); }
            int64_t a0 = (int64_t)off->kids[0]->num, b0 = (int64_t)off->kids[1]->num;
            /* offset dichiarati dal file: non-negativi, ordinati e dentro al
             * file. Altrimenti nbytes=b0-a0 diventa negativo -> malloc((size_t))
             * gigante e la memcpy in st_read_f32 sfora il buffer del chiamante;
             * oppure off punta fuori dal file. */
            if (a0 < 0 || b0 < a0 || data_start + b0 > fsz) {
                fprintf(stderr, "%s: tensor '%s' data_offsets [%lld,%lld] out of file bounds (%lld)\n",
                        files[fi], name, (long long)a0, (long long)b0, (long long)fsz); exit(1); }
            /* SEC: lo shape viene da un file non fidato (mirror). Senza il guard
             * di overflow, uno shape tipo [65535,65535,65535,...] fa avvolgere
             * numel a un valore piccolo/negativo che poi passerebbe il cross-check
             * numel*esz==nbytes in st_read_f32, riaprendo l'OOB. */
            int64_t numel = 1; int bad_shape = 0;
            for (int k = 0; k < shp->len; k++) {
                int64_t d = (int64_t)shp->kids[k]->num;
                if (d < 0 || (d != 0 && numel > INT64_MAX / d)) { bad_shape = 1; break; }
                numel *= d;
            }
            if (bad_shape) {
                fprintf(stderr, "%s: tensor '%s' shape overflows int64 — refusing (hostile or corrupt file)\n",
                        files[fi], name); exit(1); }
            if (S->n == S->cap) { S->cap *= 2; S->t = realloc(S->t, S->cap*sizeof(st_tensor)); }
            st_tensor *t = &S->t[S->n++];
            t->name = strdup(name); t->fd = fd; t->off = data_start + a0;
            t->nbytes = b0 - a0; t->dtype = st_dtype_code(dt->str); t->numel = numel;
            /* cross-check the declared element count against the byte span for FLOAT
             * dtypes: st_read_f32 writes `numel` floats (BF16/F16 loop or F32 memcpy)
             * into a caller-sized buffer, so a header with numel != nbytes/esz is an
             * OOB write primitive. U8/I8 (raw quant bytes) are read by byte count, so
             * their numel is unused by the read path and legitimately may differ. */
            { int esz = t->dtype==2 ? 4 : (t->dtype==3 ? 1 : 2);
              if (t->dtype != 3 && t->nbytes != numel * (int64_t)esz) {
                  fprintf(stderr, "%s: tensor '%s' numel %lld disagrees with byte span %lld (esz %d)\n",
                          files[fi], name, (long long)numel, (long long)t->nbytes, esz); exit(1); } }
        }
        free(arena); /* i jval restano leakati: ok, una tantum all'avvio */
        free(hdr);
    }
    /* indice hash costruito a fine indicizzazione (gli indici restano validi dopo i realloc) */
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = malloc(S->hcap * sizeof(int));
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t h = st_hash(S->t[i].name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (S->hcap - 1);
        S->hidx[h] = i;
    }
}

/* backward-compatible single-directory entry point */
static void st_init(shards *S, const char *snap_dir) { st_init_multi(S, snap_dir, NULL); }

static st_tensor *st_find(shards *S, const char *name) {
    if (S->hidx) {
        uint64_t h = st_hash(name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) {
            st_tensor *t = &S->t[S->hidx[h]];
            if (!strcmp(t->name, name)) return t;
            h = (h + 1) & (S->hcap - 1);
        }
        return NULL;
    }
    for (int i = 0; i < S->n; i++) if (!strcmp(S->t[i].name, name)) return &S->t[i];
    return NULL;
}
static int st_has(shards *S, const char *name) { return st_find(S, name) != NULL; }

/* prefetch ASINCRONO: dice al kernel di iniziare a leggere le pagine del tensore in
 * background (readahead). Serve a sovrapporre l'I/O degli expert col calcolo: si
 * prefetcha tutto il set di expert di un layer, poi le pread sincrone trovano la cache
 * gia' calda. No-op se il tensore non esiste (es. il primo .qs prima della lettura). */
static void st_prefetch(shards *S, const char *name) {
    st_tensor *t = st_find(S, name);
    if (t) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* like st_prefetch, but on replica `rep`'s drive: the WILLNEED must warm the
 * page cache of the SAME fd the later demand pread will hit. */
static void st_prefetch_rep(shards *S, const char *name, int rep) {
    st_tensor *t = st_find(S, name);
    if (!t) return;
    int fd = st_fd_rep(S, t->fd, rep);
    if (fd < 0) fd = t->fd;
    posix_fadvise(fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* legge un tensore in un buffer float32 fornito dal chiamante (numel float).
 * drop=1 -> consiglia al kernel di scartare le pagine (per gli expert in streaming). */
static int64_t st_read_f32(shards *S, const char *name, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    /* SEC: numel viene dallo shape, nbytes dagli offset — due campi indipendenti
     * del file. Se non concordano, la memcpy F32 (nbytes) o i loop BF16/F16
     * (numel elementi da un raw di soli nbytes) sforano il buffer del chiamante,
     * che e' dimensionato sul config, non sul file. Il chiamante che alloca su
     * st_numel resta coerente; questo blocca l'ingresso ostile a monte. */
    int esz = (t->dtype == 2) ? 4 : 2;
    if (t->numel < 0 || t->numel > t->nbytes / esz || t->numel * (int64_t)esz != t->nbytes) {
        fprintf(stderr, "%s: tensor '%s' shape/bytes mismatch (numel %lld, %lld bytes, dtype %d) — refusing (hostile or corrupt file)\n",
                name, name, (long long)t->numel, (long long)t->nbytes, t->dtype); exit(1); }
    void *raw = malloc(t->nbytes);
    if (!raw) { fprintf(stderr, "malloc %lld bytes for tensor %s failed\n", (long long)t->nbytes, name); exit(1); }
    st_pread_full(t->fd, raw, t->nbytes, t->off, "pread data");
    if (t->dtype == 2) {
        memcpy(out, raw, t->nbytes);
    } else if (t->dtype == 0) {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = bf16_to_f32(p[i]);
    } else {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = f16_to_f32(p[i]);
    }
    free(raw);
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    return t->numel;
}

/* like st_read_f32 but refuses to write more than `cap` floats into `out`.
 * Callers that size `out` from CONFIG dims (qt_alloc's O*I, per-row scales) must
 * use this: a crafted header whose tensor holds more elements than the config
 * shape would otherwise overrun `out`. Callers that size `out` from st_numel are
 * self-consistent and may keep using st_read_f32. */
static int64_t st_read_f32_cap(shards *S, const char *name, float *out, int64_t cap, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->numel > cap) {
        fprintf(stderr, "tensor %s: numel %lld exceeds destination capacity %lld\n",
                name, (long long)t->numel, (long long)cap); exit(1); }
    return st_read_f32(S, name, out, drop);
}

static int64_t st_numel(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->numel : -1;
}
static int64_t st_nbytes(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->nbytes : -1;
}

/* legge i byte GREZZI di un tensore (nessuna conversione di dtype): per i pesi gia'
 * quantizzati int4/int8 del nostro container (dtype U8). drop=1 -> fadvise DONTNEED. */
static void st_read_raw(shards *S, const char *name, void *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    st_pread_full(t->fd, out, t->nbytes, t->off, "pread raw");
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
}

/* legge una FETTA di un tensore: n_elems a partire dall'elemento elem_off.
 * Serve per gli expert fusi di GLM (un tensore = blocco [E, ...]): si legge il
 * solo expert richiesto via pread del sotto-range, niente lettura dell'intero blocco. */
static void st_read_slice_f32(shards *S, const char *name, int64_t elem_off, int64_t n_elems, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    if (elem_off < 0 || n_elems < 0 || elem_off + n_elems > t->numel) {   /* keep the slice inside the tensor */
        fprintf(stderr, "slice %s [%lld,+%lld) out of tensor bounds (numel %lld)\n",
                name, (long long)elem_off, (long long)n_elems, (long long)t->numel); exit(1); }
    int64_t boff = t->off + elem_off * esz, nb = n_elems * esz;
    void *raw = malloc(nb);
    if (!raw) { fprintf(stderr, "malloc %lld bytes for slice %s failed\n", (long long)nb, name); exit(1); }
    st_pread_full(t->fd, raw, nb, boff, "pread slice");   /* dev #331: chunked + EINTR + honest short-read */
    if (t->dtype == 2) memcpy(out, raw, nb);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = bf16_to_f32(p[i]); }
    else { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    if (drop) posix_fadvise(t->fd, boff, nb, POSIX_FADV_DONTNEED);
}

#endif
