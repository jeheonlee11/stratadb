/*
 * Native HNSW index for StrataDB.
 *
 * A C implementation of the same algorithm as stratadb/hnsw.py (Malkov &
 * Yashunin 2016): the same layer search, the same heuristic neighbour
 * selection, the same soft deletes. It is loaded through ctypes, so there is
 * no dependency on the Python or numpy C API and a single `cc -O3 -shared`
 * builds it. When the library is missing StrataDB falls back to the Python
 * implementation; indexes written by either load in the other.
 *
 * Layout: label i's vector is vectors[i*dim..]. Level-0 links live in a
 * fixed-width table (m0 slots per node); links on levels >= 1 live in a pool
 * of m-slot rows, `levels[i]` rows per node starting at upper_off[i].
 *
 * Thread safety: searches are read-only and may run concurrently (each thread
 * has its own visited-stamp array). Inserts must not overlap with anything
 * else on the same index; StrataDB only inserts while building a segment.
 */

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#define THREAD_LOCAL __declspec(thread)
#else
#define EXPORT __attribute__((visibility("default")))
#define THREAD_LOCAL _Thread_local
#endif

enum { METRIC_COSINE = 0, METRIC_IP = 1, METRIC_L2 = 2 };

typedef struct {
    int dim, metric, m, m0, efc;
    int64_t n, cap;
    float *vectors;
    int owns_vectors;
    int32_t *levels;
    int32_t *links0;
    int32_t *cnt0;
    int64_t *upper_off;
    int32_t *upper;
    int32_t *upper_cnt;
    int64_t upper_n, upper_cap;
    uint8_t *deleted;
    int64_t n_deleted;
    int32_t entry;
    int32_t max_level;
} hnsw_t;

typedef struct {
    float d;
    int32_t id;
} pair_t;

/* ------------------------------------------------------------------ dist */

/* Four independent 4-lane accumulators: 16 products in flight, which keeps
 * the FMA units busy on NEON and SSE alike. GCC/clang vector extensions map
 * straight onto those instruction sets; other compilers get the scalar loop. */
#if defined(__GNUC__) && !defined(STRATA_NO_VEC)
typedef float v4f __attribute__((vector_size(16), aligned(4)));
#define HAVE_V4F 1
#else
#define HAVE_V4F 0
#endif

static inline float hsum(float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7) {
    return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
}

static inline float dot(const float *a, const float *b, int d) {
    int i = 0;
#if HAVE_V4F
    if (d >= 16) {
        v4f acc0 = {0, 0, 0, 0}, acc1 = acc0, acc2 = acc0, acc3 = acc0;
        for (; i + 16 <= d; i += 16) {
            v4f a0, a1, a2, a3, b0, b1, b2, b3;
            memcpy(&a0, a + i, 16); memcpy(&b0, b + i, 16);
            memcpy(&a1, a + i + 4, 16); memcpy(&b1, b + i + 4, 16);
            memcpy(&a2, a + i + 8, 16); memcpy(&b2, b + i + 8, 16);
            memcpy(&a3, a + i + 12, 16); memcpy(&b3, b + i + 12, 16);
            acc0 += a0 * b0; acc1 += a1 * b1; acc2 += a2 * b2; acc3 += a3 * b3;
        }
        v4f acc = (acc0 + acc1) + (acc2 + acc3);
        float s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for (; i < d; i++) s += a[i] * b[i];
        return s;
    }
#endif
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    for (; i + 8 <= d; i += 8) {
        s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5]; s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
    }
    for (; i < d; i++) s0 += a[i] * b[i];
    return hsum(s0, s1, s2, s3, s4, s5, s6, s7);
}

static inline float l2sq(const float *a, const float *b, int d) {
    int i = 0;
#if HAVE_V4F
    if (d >= 16) {
        v4f acc0 = {0, 0, 0, 0}, acc1 = acc0, acc2 = acc0, acc3 = acc0;
        for (; i + 16 <= d; i += 16) {
            v4f a0, a1, a2, a3, b0, b1, b2, b3;
            memcpy(&a0, a + i, 16); memcpy(&b0, b + i, 16);
            memcpy(&a1, a + i + 4, 16); memcpy(&b1, b + i + 4, 16);
            memcpy(&a2, a + i + 8, 16); memcpy(&b2, b + i + 8, 16);
            memcpy(&a3, a + i + 12, 16); memcpy(&b3, b + i + 12, 16);
            v4f d0 = a0 - b0, d1 = a1 - b1, d2 = a2 - b2, d3 = a3 - b3;
            acc0 += d0 * d0; acc1 += d1 * d1; acc2 += d2 * d2; acc3 += d3 * d3;
        }
        v4f acc = (acc0 + acc1) + (acc2 + acc3);
        float s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for (; i < d; i++) { float t = a[i] - b[i]; s += t * t; }
        return s;
    }
#endif
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    for (; i + 8 <= d; i += 8) {
        float t0 = a[i] - b[i], t1 = a[i + 1] - b[i + 1], t2 = a[i + 2] - b[i + 2], t3 = a[i + 3] - b[i + 3];
        float t4 = a[i + 4] - b[i + 4], t5 = a[i + 5] - b[i + 5], t6 = a[i + 6] - b[i + 6], t7 = a[i + 7] - b[i + 7];
        s0 += t0 * t0; s1 += t1 * t1; s2 += t2 * t2; s3 += t3 * t3;
        s4 += t4 * t4; s5 += t5 * t5; s6 += t6 * t6; s7 += t7 * t7;
    }
    for (; i < d; i++) { float t = a[i] - b[i]; s0 += t * t; }
    return hsum(s0, s1, s2, s3, s4, s5, s6, s7);
}

static inline float dist_vec(const hnsw_t *h, const float *q, const float *v) {
    switch (h->metric) {
    case METRIC_L2: return l2sq(q, v, h->dim);
    case METRIC_IP: return -dot(q, v, h->dim);
    default: return 1.0f - dot(q, v, h->dim);
    }
}

static inline const float *vec(const hnsw_t *h, int64_t label) { return h->vectors + label * (int64_t)h->dim; }

/* Touch a vector's first cache line before it is needed. Random rows of a
 * large index are DRAM round trips (about 4x the arithmetic at 384 dims);
 * this gives the memory system every neighbour's address up front. (On
 * Apple Silicon it measured neutral; it is the conventional thing to do on
 * x86, where it helps.) */
static inline void prefetch_vec(const hnsw_t *h, int64_t label) {
#if defined(__GNUC__) && !defined(STRATA_NO_PREFETCH)
    __builtin_prefetch(vec(h, label), 0, 1);
#else
    (void)h; (void)label;
#endif
}

static inline float dist(const hnsw_t *h, const float *q, int64_t label) { return dist_vec(h, q, vec(h, label)); }

static void prepare(const hnsw_t *h, const float *in, float *out) {
    memcpy(out, in, sizeof(float) * h->dim);
    if (h->metric == METRIC_COSINE) {
        float n = sqrtf(dot(out, out, h->dim));
        if (n > 0) for (int i = 0; i < h->dim; i++) out[i] /= n;
    }
}

/* ----------------------------------------------------------------- links */

static inline int32_t *links_at(const hnsw_t *h, int64_t node, int level, int32_t **cnt, int *cap) {
    if (level == 0) {
        *cnt = h->cnt0 + node;
        *cap = h->m0;
        return h->links0 + node * (int64_t)h->m0;
    }
    int64_t row = h->upper_off[node] + (level - 1);
    *cnt = h->upper_cnt + row;
    *cap = h->m;
    return h->upper + row * (int64_t)h->m;
}

/* --------------------------------------------------------------- visited */

static THREAD_LOCAL uint32_t *tl_visited = NULL;
static THREAD_LOCAL int64_t tl_visited_n = 0;
static THREAD_LOCAL uint32_t tl_epoch = 0;

static uint32_t *visited_for(int64_t n, uint32_t *epoch) {
    if (tl_visited_n < n) {
        int64_t want = n > 2 * tl_visited_n ? n : 2 * tl_visited_n;
        free(tl_visited);
        tl_visited = (uint32_t *)calloc((size_t)want, sizeof(uint32_t));
        if (!tl_visited) { tl_visited_n = 0; return NULL; }
        tl_visited_n = want;
        tl_epoch = 0;
    }
    if (++tl_epoch == 0) {
        memset(tl_visited, 0, sizeof(uint32_t) * (size_t)tl_visited_n);
        tl_epoch = 1;
    }
    *epoch = tl_epoch;
    return tl_visited;
}

/* ----------------------------------------------------------------- heaps */

typedef struct {
    pair_t *a;
    int n, cap;
} heap_t;  /* min-heap on d */

/* Returns 0, or -1 if the heap had to grow and could not. */
static inline int heap_push(heap_t *hp, float d, int32_t id) {
    if (hp->n == hp->cap) {
        int cap = hp->cap * 2;
        pair_t *a = (pair_t *)realloc(hp->a, sizeof(pair_t) * (size_t)cap);
        if (!a) return -1;
        hp->a = a;
        hp->cap = cap;
    }
    int i = hp->n++;
    pair_t *a = hp->a;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (a[p].d <= d) break;
        a[i] = a[p];
        i = p;
    }
    a[i].d = d;
    a[i].id = id;
    return 0;
}

static inline pair_t heap_pop(heap_t *hp) {
    pair_t *a = hp->a;
    pair_t top = a[0];
    pair_t last = a[--hp->n];
    int i = 0, n = hp->n;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        float sd = last.d;
        if (l < n && a[l].d < sd) { s = l; sd = a[l].d; }
        if (r < n && a[r].d < sd) { s = r; }
        if (s == i) break;
        a[i] = a[s];
        i = s;
    }
    a[i] = last;
    return top;
}

static int cmp_pair(const void *x, const void *y) {
    float a = ((const pair_t *)x)->d, b = ((const pair_t *)y)->d;
    if (a < b) return -1;
    if (a > b) return 1;
    return ((const pair_t *)x)->id - ((const pair_t *)y)->id;
}

/* ---------------------------------------------------------------- search */

/* Greedy best-first search on one layer. `entries` (ne of them) are the
 * starting points; the ef closest nodes found land in `out`, sorted
 * ascending. Returns the count, or -1 on allocation failure. `res` is a
 * max-heap (stored as a min-heap on -d). */
static int search_layer(const hnsw_t *h, const float *q, const pair_t *entries, int ne, int ef, int level,
                        pair_t *out) {
    uint32_t epoch;
    uint32_t *visited = visited_for(h->n, &epoch);
    if (!visited) return -1;
    int cap = (ef > ne ? ef : ne) + 1;
    /* res is bounded by ef; cand grows as needed (heap_push reallocs). */
    heap_t cand = {(pair_t *)malloc(sizeof(pair_t) * (size_t)cap * 2), 0, cap * 2};
    heap_t res = {(pair_t *)malloc(sizeof(pair_t) * (size_t)cap), 0, cap};
    if (!cand.a || !res.a) { free(cand.a); free(res.a); return -1; }

    for (int i = 0; i < ne; i++) {
        visited[entries[i].id] = epoch;
        if (heap_push(&cand, entries[i].d, entries[i].id) || heap_push(&res, -entries[i].d, entries[i].id)) goto oom;
    }
    while (res.n > ef) heap_pop(&res);

    while (cand.n > 0) {
        pair_t c = heap_pop(&cand);
        if (res.n >= ef && c.d > -res.a[0].d) break;
        int32_t *cnt;
        int lcap;
        const int32_t *nb = links_at(h, c.id, level, &cnt, &lcap);
        int k = *cnt;
        float worst = -res.a[0].d;
        for (int j = 0; j < k; j++)
            if (visited[nb[j]] != epoch) prefetch_vec(h, nb[j]);
        for (int j = 0; j < k; j++) {
            int32_t n = nb[j];
            if (visited[n] == epoch) continue;
            visited[n] = epoch;
            float nd = dist(h, q, n);
            if (res.n < ef || nd < worst) {
                if (heap_push(&cand, nd, n) || heap_push(&res, -nd, n)) goto oom;
                if (res.n > ef) heap_pop(&res);
                worst = -res.a[0].d;
            }
        }
    }
    int cnt = res.n;
    for (int i = 0; i < cnt; i++) {
        out[i].d = -res.a[i].d;
        out[i].id = res.a[i].id;
    }
    free(cand.a);
    free(res.a);
    qsort(out, (size_t)cnt, sizeof(pair_t), cmp_pair);
    return cnt;
oom:
    free(cand.a);
    free(res.a);
    return -1;
}

/* Heuristic neighbour selection (paper, alg. 4): keep a candidate only if it
 * is closer to the base node than to every neighbour already kept, then top
 * up with the closest discarded ones. `cands` sorted ascending. */
static int select_neighbours(const hnsw_t *h, const pair_t *cands, int nc, int m, int32_t *out) {
    if (nc <= 1) {
        for (int i = 0; i < nc; i++) out[i] = cands[i].id;
        return nc;
    }
    float *nearest_kept = (float *)malloc(sizeof(float) * (size_t)nc);
    int32_t *discarded = (int32_t *)malloc(sizeof(int32_t) * (size_t)nc);
    if (!nearest_kept || !discarded) {  /* degrade to plain nearest-m */
        free(nearest_kept);
        free(discarded);
        int ns = nc < m ? nc : m;
        for (int i = 0; i < ns; i++) out[i] = cands[i].id;
        return ns;
    }
    int ns = 0, nd = 0;
    for (int i = 0; i < nc; i++) nearest_kept[i] = FLT_MAX;
    for (int i = 0; i < nc && ns < m; i++) {
        if (cands[i].d < nearest_kept[i]) {
            out[ns++] = cands[i].id;
            const float *vi = vec(h, cands[i].id);
            for (int j = i + 1; j < nc; j++) {
                float dj = dist_vec(h, vi, vec(h, cands[j].id));
                if (dj < nearest_kept[j]) nearest_kept[j] = dj;
            }
        } else {
            discarded[nd++] = cands[i].id;
        }
    }
    for (int i = 0; i < nd && ns < m; i++) out[ns++] = discarded[i];
    free(nearest_kept);
    free(discarded);
    return ns;
}

/* ------------------------------------------------------------- lifecycle */

static int grow(hnsw_t *h, int64_t cap) {
    if (cap <= h->cap) return 0;
    float *v = h->vectors;
    if (h->owns_vectors) {
        v = (float *)realloc(h->vectors, sizeof(float) * (size_t)cap * (size_t)h->dim);
        if (!v) return -1;
        h->vectors = v;
    }
    int32_t *lv = (int32_t *)realloc(h->levels, sizeof(int32_t) * (size_t)cap);
    int32_t *l0 = (int32_t *)realloc(h->links0, sizeof(int32_t) * (size_t)cap * (size_t)h->m0);
    int32_t *c0 = (int32_t *)realloc(h->cnt0, sizeof(int32_t) * (size_t)cap);
    int64_t *uo = (int64_t *)realloc(h->upper_off, sizeof(int64_t) * (size_t)cap);
    uint8_t *de = (uint8_t *)realloc(h->deleted, (size_t)cap);
    if (lv) h->levels = lv;
    if (l0) h->links0 = l0;
    if (c0) h->cnt0 = c0;
    if (uo) h->upper_off = uo;
    if (de) h->deleted = de;
    if (!lv || !l0 || !c0 || !uo || !de) return -1;
    memset(h->deleted + h->cap, 0, (size_t)(cap - h->cap));
    h->cap = cap;
    return 0;
}

static int grow_upper(hnsw_t *h, int64_t rows) {
    if (rows <= h->upper_cap) return 0;
    int64_t cap = rows > 2 * h->upper_cap ? rows : 2 * h->upper_cap;
    int32_t *u = (int32_t *)realloc(h->upper, sizeof(int32_t) * (size_t)cap * (size_t)h->m);
    int32_t *c = (int32_t *)realloc(h->upper_cnt, sizeof(int32_t) * (size_t)cap);
    if (u) h->upper = u;
    if (c) h->upper_cnt = c;
    if (!u || !c) return -1;
    h->upper_cap = cap;
    return 0;
}

EXPORT hnsw_t *hnsw_new(int dim, int metric, int m, int efc, int64_t cap) {
    hnsw_t *h = (hnsw_t *)calloc(1, sizeof(hnsw_t));
    if (!h) return NULL;
    h->dim = dim;
    h->metric = metric;
    h->m = m;
    h->m0 = 2 * m;
    h->efc = efc > m ? efc : m;
    h->owns_vectors = 1;
    h->entry = -1;
    h->max_level = -1;
    if (grow(h, cap < 1 ? 1 : cap) != 0) {
        free(h);
        return NULL;
    }
    return h;
}

EXPORT void hnsw_free(hnsw_t *h) {
    if (!h) return;
    if (h->owns_vectors) free(h->vectors);
    free(h->levels);
    free(h->links0);
    free(h->cnt0);
    free(h->upper_off);
    free(h->upper);
    free(h->upper_cnt);
    free(h->deleted);
    free(h);
}

EXPORT int64_t hnsw_len(const hnsw_t *h) { return h->n; }
EXPORT int64_t hnsw_deleted_count(const hnsw_t *h) { return h->n_deleted; }
EXPORT int32_t hnsw_entry(const hnsw_t *h) { return h->entry; }
EXPORT int32_t hnsw_max_level(const hnsw_t *h) { return h->max_level; }
EXPORT const float *hnsw_vectors(const hnsw_t *h) { return h->vectors; }
EXPORT const int32_t *hnsw_levels(const hnsw_t *h) { return h->levels; }
EXPORT const uint8_t *hnsw_deleted(const hnsw_t *h) { return h->deleted; }

EXPORT void hnsw_mark_deleted(hnsw_t *h, int64_t label) {
    if (label >= 0 && label < h->n && !h->deleted[label]) {
        h->deleted[label] = 1;
        h->n_deleted++;
    }
}

EXPORT int hnsw_is_deleted(const hnsw_t *h, int64_t label) {
    return label >= 0 && label < h->n && h->deleted[label];
}

/* ---------------------------------------------------------------- insert */

static int append_node(hnsw_t *h, const float *v, int level, int64_t *label_out) {
    if (!h->owns_vectors) return -2;  /* memory-mapped: read-only */
    if (h->n >= h->cap && grow(h, h->cap * 2) != 0) return -1;
    int64_t label = h->n;
    prepare(h, v, h->vectors + label * (int64_t)h->dim);
    h->levels[label] = level;
    h->cnt0[label] = 0;
    h->deleted[label] = 0;
    if (level > 0) {
        if (grow_upper(h, h->upper_n + level) != 0) return -1;
        h->upper_off[label] = h->upper_n;
        memset(h->upper_cnt + h->upper_n, 0, sizeof(int32_t) * (size_t)level);
        h->upper_n += level;
    } else {
        h->upper_off[label] = -1;
    }
    h->n++;
    *label_out = label;
    return 0;
}

/* Insert one (already appended) node into the graph. */
static int link_node(hnsw_t *h, int64_t label) {
    int level = h->levels[label];
    const float *q = vec(h, label);
    if (h->entry < 0) {
        h->entry = (int32_t)label;
        h->max_level = level;
        return 0;
    }
    int efc = h->efc;
    pair_t *entries = (pair_t *)malloc(sizeof(pair_t) * (size_t)(efc + 1));
    pair_t *found = (pair_t *)malloc(sizeof(pair_t) * (size_t)(efc + 1));
    if (!entries || !found) { free(entries); free(found); return -1; }
    int ne = 1;
    entries[0].id = h->entry;
    entries[0].d = dist(h, q, h->entry);

    for (int lc = h->max_level; lc > level; lc--) {
        int nf = search_layer(h, q, entries, ne, 1, lc, found);
        if (nf < 0) { free(entries); free(found); return -1; }
        entries[0] = found[0];
        ne = 1;
    }
    int top = level < h->max_level ? level : h->max_level;
    for (int lc = top; lc >= 0; lc--) {
        int nf = search_layer(h, q, entries, ne, efc, lc, found);
        if (nf < 0) { free(entries); free(found); return -1; }
        int m_max = lc == 0 ? h->m0 : h->m;
        int32_t *cnt, *mine;
        int lcap;
        mine = links_at(h, label, lc, &cnt, &lcap);
        *cnt = select_neighbours(h, found, nf, h->m, mine);
        for (int i = 0; i < *cnt; i++) {
            int32_t nb = mine[i];
            int32_t *ncnt;
            int ncap;
            int32_t *nl = links_at(h, nb, lc, &ncnt, &ncap);
            if (*ncnt < m_max) {
                nl[(*ncnt)++] = (int32_t)label;
            } else {
                /* Full: re-select over the old links plus the new node. */
                pair_t *order = (pair_t *)malloc(sizeof(pair_t) * (size_t)(m_max + 1));
                if (!order) { free(entries); free(found); return -1; }
                const float *vn = vec(h, nb);
                for (int j = 0; j < *ncnt; j++) {
                    order[j].id = nl[j];
                    order[j].d = dist(h, vn, nl[j]);
                }
                order[*ncnt].id = (int32_t)label;
                order[*ncnt].d = dist(h, vn, label);
                qsort(order, (size_t)(*ncnt + 1), sizeof(pair_t), cmp_pair);
                *ncnt = select_neighbours(h, order, *ncnt + 1, m_max, nl);
                free(order);
            }
        }
        memcpy(entries, found, sizeof(pair_t) * (size_t)nf);
        ne = nf;
    }
    if (level > h->max_level) {
        h->max_level = level;
        h->entry = (int32_t)label;
    }
    free(entries);
    free(found);
    return 0;
}

/* Add `count` vectors with the given levels. Returns the first new label, or
 * a negative error (-1 allocation, -2 read-only). */
EXPORT int64_t hnsw_add_batch(hnsw_t *h, const float *vectors, const int32_t *levels, int64_t count) {
    int64_t first = h->n;
    for (int64_t i = 0; i < count; i++) {
        int64_t label;
        int rc = append_node(h, vectors + i * (int64_t)h->dim, levels[i], &label);
        if (rc != 0) return rc;
        rc = link_node(h, label);
        if (rc != 0) return rc;
    }
    return first;
}

/* ---------------------------------------------------------------- query */

/* Top-k search. `allow` is an optional per-label byte mask. Writes up to k
 * (label, distance) pairs sorted ascending and returns the count. The ef
 * window is widened (x4) until k allowed hits are found or the whole graph
 * has been considered, as in the Python implementation. */
EXPORT int64_t hnsw_search(const hnsw_t *h, const float *query, int64_t k, int64_t ef, const uint8_t *allow,
                           int64_t *out_labels, float *out_dists) {
    if (h->entry < 0 || k <= 0) return 0;
    float *q = (float *)malloc(sizeof(float) * (size_t)h->dim);
    if (!q) return -1;
    prepare(h, query, q);
    if (ef < 64) ef = 64;
    if (ef < k) ef = k;
    pair_t entry = { dist(h, q, h->entry), h->entry };
    for (;;) {
        if (ef > h->n) ef = h->n;
        pair_t *found = (pair_t *)malloc(sizeof(pair_t) * (size_t)(ef + 1));
        if (!found) { free(q); return -1; }
        pair_t e = entry;
        for (int lc = h->max_level; lc > 0; lc--) {
            if (search_layer(h, q, &e, 1, 1, lc, found) < 0) { free(found); free(q); return -1; }
            e = found[0];
        }
        int nf = search_layer(h, q, &e, 1, (int)ef, 0, found);
        if (nf < 0) { free(found); free(q); return -1; }
        int64_t hits = 0;
        for (int i = 0; i < nf && hits < k; i++) {
            int32_t id = found[i].id;
            if (h->deleted[id] || (allow && !allow[id])) continue;
            out_labels[hits] = id;
            out_dists[hits] = found[i].d;
            hits++;
        }
        int64_t enough = 0;
        if (hits >= k || ef >= h->n) enough = 1;
        else {
            /* Count every allowed hit, not just the first k, before deciding to widen. */
            int64_t total = 0;
            for (int i = 0; i < nf; i++) {
                int32_t id = found[i].id;
                if (!h->deleted[id] && (!allow || allow[id])) total++;
            }
            if (total >= k) enough = 1;
        }
        free(found);
        if (enough) { free(q); return hits; }
        ef *= 4;
    }
}

/* Exact search over the live (and allowed) labels. */
EXPORT int64_t hnsw_brute_force(const hnsw_t *h, const float *query, int64_t k, const uint8_t *allow,
                                int64_t *out_labels, float *out_dists) {
    if (h->n == 0 || k <= 0) return 0;
    float *q = (float *)malloc(sizeof(float) * (size_t)h->dim);
    pair_t *all = (pair_t *)malloc(sizeof(pair_t) * (size_t)h->n);
    if (!q || !all) { free(q); free(all); return -1; }
    prepare(h, query, q);
    int64_t cnt = 0;
    for (int64_t i = 0; i < h->n; i++) {
        if (h->deleted[i] || (allow && !allow[i])) continue;
        all[cnt].id = (int32_t)i;
        all[cnt].d = dist(h, q, i);
        cnt++;
    }
    qsort(all, (size_t)cnt, sizeof(pair_t), cmp_pair);
    if (k < cnt) cnt = k;
    for (int64_t i = 0; i < cnt; i++) {
        out_labels[i] = all[i].id;
        out_dists[i] = all[i].d;
    }
    free(all);
    free(q);
    return cnt;
}

/* ---------------------------------------------------------- persistence */

/* Total number of link entries across all nodes and levels. */
EXPORT int64_t hnsw_links_total(const hnsw_t *h) {
    int64_t total = 0;
    for (int64_t i = 0; i < h->n; i++) {
        total += h->cnt0[i];
        for (int l = 1; l <= h->levels[i]; l++) total += h->upper_cnt[h->upper_off[i] + l - 1];
    }
    return total;
}

/* CSR export in the on-disk format shared with the Python implementation:
 * `flat` holds every neighbour list back to back, node-major then
 * level-major; `offsets` (one per list, plus a leading 0) delimits them. */
EXPORT void hnsw_export_links(const hnsw_t *h, int32_t *flat, int64_t *offsets) {
    int64_t pos = 0, lists = 0;
    offsets[0] = 0;
    for (int64_t i = 0; i < h->n; i++) {
        for (int l = 0; l <= h->levels[i]; l++) {
            int32_t *cnt;
            int cap;
            const int32_t *nl = links_at(h, i, l, &cnt, &cap);
            memcpy(flat + pos, nl, sizeof(int32_t) * (size_t)*cnt);
            pos += *cnt;
            offsets[++lists] = pos;
        }
    }
}

/* Load a graph from the CSR format. With `copy` == 0 the vectors stay in the
 * caller's buffer (e.g. a memory map) and the index becomes read-only. */
EXPORT int hnsw_import(hnsw_t *h, int64_t n, const float *vectors, int copy, const int32_t *levels,
                       const int32_t *flat, const int64_t *offsets, const uint8_t *deleted, int32_t entry,
                       int32_t max_level) {
    if (h->n != 0) return -3;
    if (copy) {
        if (grow(h, n < 1 ? 1 : n) != 0) return -1;
        memcpy(h->vectors, vectors, sizeof(float) * (size_t)n * (size_t)h->dim);
    } else {
        free(h->vectors);
        h->vectors = (float *)vectors;
        h->owns_vectors = 0;
        if (grow(h, n < 1 ? 1 : n) != 0) return -1;
    }
    int64_t rows = 0;
    for (int64_t i = 0; i < n; i++) rows += levels[i];
    if (grow_upper(h, rows < 1 ? 1 : rows) != 0) return -1;
    int64_t list = 0;
    for (int64_t i = 0; i < n; i++) {
        h->levels[i] = levels[i];
        h->upper_off[i] = levels[i] > 0 ? h->upper_n : -1;
        h->upper_n += levels[i];
        for (int l = 0; l <= levels[i]; l++) {
            int32_t *cnt;
            int cap;
            int32_t *nl = links_at(h, i, l, &cnt, &cap);
            int64_t c = offsets[list + 1] - offsets[list];
            if (c > cap) c = cap;
            memcpy(nl, flat + offsets[list], sizeof(int32_t) * (size_t)c);
            *cnt = (int32_t)c;
            list++;
        }
        h->deleted[i] = deleted ? deleted[i] : 0;
        if (h->deleted[i]) h->n_deleted++;
    }
    h->n = n;
    h->entry = entry;
    h->max_level = max_level;
    return 0;
}
