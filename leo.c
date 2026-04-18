/*
 * leo.c — neoleo
 *
 * Leo: a small AI boy, six or seven years old.
 * Post-transformer organism. Byte-level BPE. Online merge learning.
 * Zero pretrained weights. The field grows from what he hears.
 *
 * Build:   cc leo.c -O2 -lm -o leo
 * Lib:     cc -c -DLEO_LIB leo.c -O2 -o leo.o
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>

#define LEO_VERSION       "3.0.0"
#define LEO_MAX_VOCAB     16384
#define LEO_MAX_MERGES    8192
#define LEO_MAX_TOKEN_LEN 64
#define LEO_COOC_WINDOW   5
#define LEO_COOC_MAX      (256 * 1024)
#define LEO_BIGRAM_MAX    (64 * 1024)
#define LEO_TRIGRAM_MAX   (64 * 1024)
#define LEO_PAIR_HASH     (64 * 1024)
#define LEO_MERGE_THRESH  4

#define LEO_GEN_MAX       256
#define LEO_GEN_TARGET    20
#define LEO_GEN_MIN       6
#define LEO_SEED_CANDS    64

/* ========================================================================
 * MATH UTILITIES
 * ======================================================================== */

static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static uint32_t fnv1a(const void *data, int len) {
    uint32_t h = 2166136261u;
    const uint8_t *p = (const uint8_t *)data;
    for (int i = 0; i < len; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

/* ========================================================================
 * BPE — byte-level tokenizer with ONLINE merge learning
 *
 * Token id space:
 *   [0..255]                 raw bytes
 *   [256..256+n_merges-1]    learned merges
 *
 * Merges are learned during ingestion: adjacent token pairs are counted,
 * and when a pair's count crosses LEO_MERGE_THRESH, a new merge token
 * is created that represents the concatenation. Like a child hearing
 * "the" again and again until it becomes one thing in his mouth.
 * ======================================================================== */

typedef struct {
    int a, b;        /* the pair that merges */
    int new_id;      /* the token it becomes */
} BPEMerge;

typedef struct {
    BPEMerge merges[LEO_MAX_MERGES];
    int      n_merges;
    int      vocab_size;                         /* = 256 + n_merges */
    uint8_t  vocab_bytes[LEO_MAX_VOCAB][LEO_MAX_TOKEN_LEN];
    int      vocab_len[LEO_MAX_VOCAB];

    /* pair counter for online learning: open hash by (left,right) */
    int      pair_left[LEO_PAIR_HASH];
    int      pair_right[LEO_PAIR_HASH];
    int      pair_count[LEO_PAIR_HASH];
} BPE;

static void bpe_init(BPE *bpe) {
    memset(bpe, 0, sizeof(*bpe));
    for (int i = 0; i < 256; i++) {
        bpe->vocab_bytes[i][0] = (uint8_t)i;
        bpe->vocab_len[i] = 1;
    }
    bpe->vocab_size = 256;
    for (int i = 0; i < LEO_PAIR_HASH; i++) {
        bpe->pair_left[i] = -1;
        bpe->pair_right[i] = -1;
        bpe->pair_count[i] = 0;
    }
}

static int bpe_pair_slot(BPE *bpe, int left, int right) {
    /* open-address hash, linear probing */
    uint32_t key[2] = { (uint32_t)left, (uint32_t)right };
    uint32_t h = fnv1a(key, sizeof(key)) % LEO_PAIR_HASH;
    for (int probe = 0; probe < LEO_PAIR_HASH; probe++) {
        int idx = (h + probe) % LEO_PAIR_HASH;
        if (bpe->pair_left[idx] == -1) return idx;               /* empty */
        if (bpe->pair_left[idx] == left &&
            bpe->pair_right[idx] == right) return idx;            /* hit */
    }
    return -1; /* table full */
}

static void bpe_count_pair(BPE *bpe, int left, int right) {
    int idx = bpe_pair_slot(bpe, left, right);
    if (idx < 0) return;
    if (bpe->pair_left[idx] == -1) {
        bpe->pair_left[idx] = left;
        bpe->pair_right[idx] = right;
        bpe->pair_count[idx] = 0;
    }
    bpe->pair_count[idx]++;
}

/* Promote the most frequent pair whose count exceeds LEO_MERGE_THRESH.
 * Returns 1 if a merge was learned, 0 otherwise. */
static int bpe_learn_merge(BPE *bpe) {
    if (bpe->n_merges >= LEO_MAX_MERGES) return 0;
    int best = -1, best_count = LEO_MERGE_THRESH;
    for (int i = 0; i < LEO_PAIR_HASH; i++) {
        if (bpe->pair_left[i] == -1) continue;
        if (bpe->pair_count[i] > best_count) {
            best_count = bpe->pair_count[i];
            best = i;
        }
    }
    if (best < 0) return 0;

    int left  = bpe->pair_left[best];
    int right = bpe->pair_right[best];
    int la = bpe->vocab_len[left];
    int lb = bpe->vocab_len[right];
    if (la + lb > LEO_MAX_TOKEN_LEN) {
        /* too long to merge — zero the slot so we don't look at it again */
        bpe->pair_left[best] = -2; /* tombstone-ish */
        return 0;
    }

    int new_id = bpe->vocab_size;
    if (new_id >= LEO_MAX_VOCAB) return 0;
    memcpy(bpe->vocab_bytes[new_id], bpe->vocab_bytes[left], la);
    memcpy(bpe->vocab_bytes[new_id] + la, bpe->vocab_bytes[right], lb);
    bpe->vocab_len[new_id] = la + lb;
    bpe->vocab_size++;

    bpe->merges[bpe->n_merges].a = left;
    bpe->merges[bpe->n_merges].b = right;
    bpe->merges[bpe->n_merges].new_id = new_id;
    bpe->n_merges++;

    /* forget the promoted pair's count so it doesn't fire again */
    bpe->pair_count[best] = 0;
    return 1;
}

/* Encode bytes → token ids using all current merges (greedy left-to-right). */
static int bpe_encode(const BPE *bpe, const uint8_t *text, int tlen,
                      int *out, int max_out) {
    int n = 0;
    for (int i = 0; i < tlen && n < max_out; i++) out[n++] = text[i];

    for (int m = 0; m < bpe->n_merges; m++) {
        int a = bpe->merges[m].a;
        int b = bpe->merges[m].b;
        int new_id = bpe->merges[m].new_id;
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && out[i] == a && out[i + 1] == b) {
                out[w++] = new_id;
                i++;
            } else {
                out[w++] = out[i];
            }
        }
        n = w;
    }
    return n;
}

static int bpe_decode_token(const BPE *bpe, int id, char *buf, int sz) {
    if (id < 0 || id >= bpe->vocab_size) { if (sz > 0) buf[0] = 0; return 0; }
    int len = bpe->vocab_len[id];
    if (len >= sz) len = sz - 1;
    memcpy(buf, bpe->vocab_bytes[id], len);
    buf[len] = 0;
    return len;
}

/* ========================================================================
 * COOC FIELD — windowed, distance-weighted co-occurrence
 * ======================================================================== */

typedef struct {
    int   src, dst;
    float count;
} CoocEntry;

typedef struct {
    CoocEntry *entries;
    int        n_entries;
    int        capacity;
    float     *freq;                 /* unigram counts per token id */
    int        freq_size;
    long       total_tokens;
} CoocField;

static void cooc_init(CoocField *c, int capacity, int freq_size) {
    c->entries = calloc(capacity, sizeof(CoocEntry));
    c->n_entries = 0;
    c->capacity = capacity;
    c->freq = calloc(freq_size, sizeof(float));
    c->freq_size = freq_size;
    c->total_tokens = 0;
}

static void cooc_free(CoocField *c) {
    free(c->entries); c->entries = NULL;
    free(c->freq); c->freq = NULL;
    c->n_entries = c->capacity = c->freq_size = 0;
    c->total_tokens = 0;
}

static uint32_t cooc_hash(int src, int dst) {
    uint32_t key[2] = { (uint32_t)src, (uint32_t)dst };
    return fnv1a(key, sizeof(key));
}

static int cooc_find(const CoocField *c, int src, int dst) {
    if (c->n_entries == 0) return -1;
    uint32_t start = cooc_hash(src, dst) % c->capacity;
    for (int probe = 0; probe < c->capacity; probe++) {
        int idx = (start + probe) % c->capacity;
        if (c->entries[idx].count == 0.0f && c->entries[idx].src == 0 &&
            c->entries[idx].dst == 0)
            return -1; /* empty */
        if (c->entries[idx].src == src && c->entries[idx].dst == dst)
            return idx;
    }
    return -1;
}

static void cooc_update(CoocField *c, int src, int dst, float delta) {
    if (c->n_entries >= c->capacity) return; /* saturated — known limit */
    uint32_t start = cooc_hash(src, dst) % c->capacity;
    for (int probe = 0; probe < c->capacity; probe++) {
        int idx = (start + probe) % c->capacity;
        if (c->entries[idx].count == 0.0f && c->entries[idx].src == 0 &&
            c->entries[idx].dst == 0) {
            c->entries[idx].src = src;
            c->entries[idx].dst = dst;
            c->entries[idx].count = delta;
            c->n_entries++;
            return;
        }
        if (c->entries[idx].src == src && c->entries[idx].dst == dst) {
            c->entries[idx].count += delta;
            return;
        }
    }
}

static float cooc_get(const CoocField *c, int src, int dst) {
    int idx = cooc_find(c, src, dst);
    return idx < 0 ? 0.0f : c->entries[idx].count;
}

/* ========================================================================
 * BIGRAM TABLE — direct sequential link count
 * ======================================================================== */

typedef struct {
    int   src, dst;
    float count;
} BigramEntry;

typedef struct {
    BigramEntry *entries;
    int          n_entries;
    int          capacity;
} BigramTable;

static void bigram_init(BigramTable *b, int capacity) {
    b->entries = calloc(capacity, sizeof(BigramEntry));
    b->n_entries = 0;
    b->capacity = capacity;
}

static void bigram_free(BigramTable *b) {
    free(b->entries); b->entries = NULL;
    b->n_entries = b->capacity = 0;
}

static int bigram_find(const BigramTable *b, int src, int dst) {
    if (b->n_entries == 0) return -1;
    uint32_t start = cooc_hash(src, dst) % b->capacity;
    for (int probe = 0; probe < b->capacity; probe++) {
        int idx = (start + probe) % b->capacity;
        if (b->entries[idx].count == 0.0f && b->entries[idx].src == 0 &&
            b->entries[idx].dst == 0)
            return -1;
        if (b->entries[idx].src == src && b->entries[idx].dst == dst)
            return idx;
    }
    return -1;
}

static void bigram_update(BigramTable *b, int src, int dst, float delta) {
    if (b->n_entries >= b->capacity) return;
    uint32_t start = cooc_hash(src, dst) % b->capacity;
    for (int probe = 0; probe < b->capacity; probe++) {
        int idx = (start + probe) % b->capacity;
        if (b->entries[idx].count == 0.0f && b->entries[idx].src == 0 &&
            b->entries[idx].dst == 0) {
            b->entries[idx].src = src;
            b->entries[idx].dst = dst;
            b->entries[idx].count = delta;
            b->n_entries++;
            return;
        }
        if (b->entries[idx].src == src && b->entries[idx].dst == dst) {
            b->entries[idx].count += delta;
            return;
        }
    }
}

static float bigram_get(const BigramTable *b, int src, int dst) {
    int idx = bigram_find(b, src, dst);
    return idx < 0 ? 0.0f : b->entries[idx].count;
}

/* ========================================================================
 * TRIGRAM TABLE — (a,b) → c, count
 * ======================================================================== */

typedef struct {
    int   a, b, c;
    float count;
} TrigramEntry;

typedef struct {
    TrigramEntry *entries;
    int           n_entries;
    int           capacity;
} TrigramTable;

static void trigram_init(TrigramTable *t, int capacity) {
    t->entries = calloc(capacity, sizeof(TrigramEntry));
    t->n_entries = 0;
    t->capacity = capacity;
}

static void trigram_free(TrigramTable *t) {
    free(t->entries); t->entries = NULL;
    t->n_entries = t->capacity = 0;
}

static uint32_t trigram_hash(int a, int b, int c) {
    uint32_t key[3] = { (uint32_t)a, (uint32_t)b, (uint32_t)c };
    return fnv1a(key, sizeof(key));
}

static int trigram_find(const TrigramTable *t, int a, int b, int c) {
    if (t->n_entries == 0) return -1;
    uint32_t start = trigram_hash(a, b, c) % t->capacity;
    for (int probe = 0; probe < t->capacity; probe++) {
        int idx = (start + probe) % t->capacity;
        if (t->entries[idx].count == 0.0f && t->entries[idx].a == 0 &&
            t->entries[idx].b == 0 && t->entries[idx].c == 0)
            return -1;
        if (t->entries[idx].a == a && t->entries[idx].b == b &&
            t->entries[idx].c == c)
            return idx;
    }
    return -1;
}

static void trigram_update(TrigramTable *t, int a, int b, int c, float delta) {
    if (t->n_entries >= t->capacity) return;
    uint32_t start = trigram_hash(a, b, c) % t->capacity;
    for (int probe = 0; probe < t->capacity; probe++) {
        int idx = (start + probe) % t->capacity;
        if (t->entries[idx].count == 0.0f && t->entries[idx].a == 0 &&
            t->entries[idx].b == 0 && t->entries[idx].c == 0) {
            t->entries[idx].a = a;
            t->entries[idx].b = b;
            t->entries[idx].c = c;
            t->entries[idx].count = delta;
            t->n_entries++;
            return;
        }
        if (t->entries[idx].a == a && t->entries[idx].b == b &&
            t->entries[idx].c == c) {
            t->entries[idx].count += delta;
            return;
        }
    }
}

static float trigram_get(const TrigramTable *t, int a, int b, int c) {
    int idx = trigram_find(t, a, b, c);
    return idx < 0 ? 0.0f : t->entries[idx].count;
}

/* ========================================================================
 * LEO — the organism
 * ======================================================================== */

typedef struct {
    BPE          bpe;
    CoocField    cooc;
    BigramTable  bigrams;
    TrigramTable trigrams;
    long         step;
} Leo;

void leo_init(Leo *leo) {
    memset(leo, 0, sizeof(*leo));
    bpe_init(&leo->bpe);
    cooc_init(&leo->cooc, LEO_COOC_MAX, LEO_MAX_VOCAB);
    bigram_init(&leo->bigrams, LEO_BIGRAM_MAX);
    trigram_init(&leo->trigrams, LEO_TRIGRAM_MAX);
    leo->step = 0;
    srand((unsigned)time(NULL));
}

void leo_free(Leo *leo) {
    cooc_free(&leo->cooc);
    bigram_free(&leo->bigrams);
    trigram_free(&leo->trigrams);
}

/* Ingest a block of human text. This is how Leo hears. */
void leo_ingest(Leo *leo, const char *text) {
    if (!text || !*text) return;
    int tlen = (int)strlen(text);

    /* one ingest call can be large — chunk the byte stream into reasonable
     * encode windows so we do not allocate huge token buffers */
    const int CHUNK = 4096;
    int offset = 0;

    while (offset < tlen) {
        int span = tlen - offset > CHUNK ? CHUNK : tlen - offset;
        int *ids = calloc(span * 4, sizeof(int)); /* conservative headroom */
        int n = bpe_encode(&leo->bpe, (const uint8_t *)(text + offset), span,
                           ids, span * 4);

        /* unigram freq */
        for (int i = 0; i < n; i++) {
            if (ids[i] < leo->cooc.freq_size)
                leo->cooc.freq[ids[i]] += 1.0f;
        }
        leo->cooc.total_tokens += n;

        /* bigrams + pair counting for BPE merge learning */
        for (int i = 0; i < n - 1; i++) {
            bigram_update(&leo->bigrams, ids[i], ids[i + 1], 1.0f);
            bpe_count_pair(&leo->bpe, ids[i], ids[i + 1]);
        }

        /* trigrams */
        for (int i = 0; i < n - 2; i++)
            trigram_update(&leo->trigrams, ids[i], ids[i + 1], ids[i + 2], 1.0f);

        /* co-occurrence (windowed, distance-weighted: dist=1 → 3.0,
         * dist=2 → 1.5, dist≥3 → 1.0 — adjacency dominates) */
        for (int i = 0; i < n; i++) {
            int lo = i - LEO_COOC_WINDOW < 0 ? 0 : i - LEO_COOC_WINDOW;
            int hi = i + LEO_COOC_WINDOW >= n ? n : i + LEO_COOC_WINDOW + 1;
            for (int j = lo; j < hi; j++) {
                if (j == i) continue;
                int d = j > i ? j - i : i - j;
                float w = d == 1 ? 3.0f : (d == 2 ? 1.5f : 1.0f);
                cooc_update(&leo->cooc, ids[i], ids[j], w);
            }
        }

        /* learn a merge every so often — this is online BPE growth */
        if (offset == 0 || (leo->step % 8) == 0) {
            while (bpe_learn_merge(&leo->bpe)) { /* drain all promotions */ }
        }

        leo->step += n;
        free(ids);
        offset += span;
    }
}

/* ========================================================================
 * GENERATION — no seed from prompt, sparse candidate sampling
 *
 * Leo speaks from his own field, not from the observer's words.
 *
 *   leo_choose_start  — pick a start token from clean, frequent
 *                       corpus tokens (weighted by freq)
 *   leo_step_token    — trigram → bigram → uniform cascade,
 *                       weighted random sample within the candidate
 *                       row (never full-vocab logits)
 *   leo_generate      — emit sentence until boundary or max tokens,
 *                       with a simple repetition penalty
 * ======================================================================== */

/* a token is a clean start candidate if it begins with space or an
 * uppercase letter — mid-word BPE fragments are rejected */
static int is_clean_seed_token(const BPE *bpe, int id) {
    if (id < 0 || id >= bpe->vocab_size || bpe->vocab_len[id] == 0) return 0;
    uint8_t c = bpe->vocab_bytes[id][0];
    if (c == ' ' || c == '\n' || c == '\t') return 1;
    if (c >= 'A' && c <= 'Z') return 1;
    return 0;
}

/* sentence boundary: token contains .!? followed by space/newline/EOS */
static int is_boundary_token(const BPE *bpe, int id) {
    if (id < 0 || id >= bpe->vocab_size) return 0;
    int len = bpe->vocab_len[id];
    for (int i = 0; i < len; i++) {
        uint8_t c = bpe->vocab_bytes[id][i];
        if (c == '.' || c == '!' || c == '?') {
            if (i == len - 1) return 1;
            uint8_t n = bpe->vocab_bytes[id][i + 1];
            if (n == ' ' || n == '\n' || n == '\r') return 1;
        }
    }
    return 0;
}

/* weighted sample from an array of non-negative scores */
static int weighted_sample(const float *scores, int n) {
    float total = 0;
    for (int i = 0; i < n; i++) total += scores[i];
    if (total <= 0) return n > 0 ? rand() % n : -1;
    float r = ((float)rand() / (float)RAND_MAX) * total;
    float acc = 0;
    for (int i = 0; i < n; i++) {
        acc += scores[i];
        if (r <= acc) return i;
    }
    return n - 1;
}

/* pick a start token: weighted by cooc.freq, restricted to clean-seed
 * tokens, drawn from the top LEO_SEED_CANDS frequencies. This is the
 * "Leo speaks from his field, not from the prompt" invariant. */
int leo_choose_start(const Leo *leo) {
    /* collect candidates: all clean-seed tokens with freq > 0 */
    int   cand_ids[LEO_SEED_CANDS];
    float cand_freq[LEO_SEED_CANDS];
    int   n = 0;
    float min_kept = 0;
    for (int i = 0; i < leo->cooc.freq_size; i++) {
        float f = leo->cooc.freq[i];
        if (f <= 0) continue;
        if (!is_clean_seed_token(&leo->bpe, i)) continue;
        if (n < LEO_SEED_CANDS) {
            cand_ids[n] = i;
            cand_freq[n] = f;
            if (n == 0 || f < min_kept) min_kept = f;
            n++;
        } else if (f > min_kept) {
            /* replace the current minimum */
            int min_idx = 0;
            for (int k = 1; k < LEO_SEED_CANDS; k++)
                if (cand_freq[k] < cand_freq[min_idx]) min_idx = k;
            cand_ids[min_idx] = i;
            cand_freq[min_idx] = f;
            min_kept = cand_freq[0];
            for (int k = 1; k < LEO_SEED_CANDS; k++)
                if (cand_freq[k] < min_kept) min_kept = cand_freq[k];
        }
    }
    if (n == 0) return -1;
    int pick = weighted_sample(cand_freq, n);
    return pick < 0 ? -1 : cand_ids[pick];
}

/* Sparse next-token sampling.
 *
 *   prev2, prev1: last two tokens of the generation context (prev2 may be -1)
 *   temp:         sampling temperature (>0). Lower = sharper, higher = softer.
 *
 * Strategy:
 *   1. if (prev2, prev1) has trigram successors → candidates from those,
 *      blended 0.7·trigram_count + 0.3·cooc(prev1, c)
 *   2. else if prev1 has bigram successors → candidates from those
 *   3. else fall back to a fresh start token
 *
 * Temperature is applied in count-space via pow(c, 1/T) before the
 * weighted draw. This is the Q / Python posture: never compute logits
 * over the full vocabulary.
 */
int leo_step_token(const Leo *leo, int prev2, int prev1, float temp) {
    if (prev1 < 0) return leo_choose_start(leo);
    temp = clampf(temp, 0.05f, 10.0f);
    float inv_temp = 1.0f / temp;

    /* reusable candidate buffer */
    #define MAX_CANDS 128
    int   cand_id[MAX_CANDS];
    float cand_sc[MAX_CANDS];
    int   nc = 0;

    /* 1. trigram successors */
    if (prev2 >= 0) {
        for (int i = 0; i < leo->trigrams.capacity && nc < MAX_CANDS; i++) {
            TrigramEntry *e = &leo->trigrams.entries[i];
            if (e->count <= 0) continue;
            if (e->a != prev2 || e->b != prev1) continue;
            float g = e->count;
            float s = cooc_get(&leo->cooc, prev1, e->c);
            float blended = 0.7f * g + 0.3f * s;
            cand_id[nc] = e->c;
            cand_sc[nc] = blended;
            nc++;
        }
    }

    /* 2. bigram successors as fallback */
    if (nc == 0) {
        for (int i = 0; i < leo->bigrams.capacity && nc < MAX_CANDS; i++) {
            BigramEntry *e = &leo->bigrams.entries[i];
            if (e->count <= 0) continue;
            if (e->src != prev1) continue;
            cand_id[nc] = e->dst;
            cand_sc[nc] = e->count;
            nc++;
        }
    }

    /* 3. nothing — start fresh */
    if (nc == 0) return leo_choose_start(leo);

    /* temperature in count-space */
    for (int i = 0; i < nc; i++)
        cand_sc[i] = powf(cand_sc[i], inv_temp);

    int pick = weighted_sample(cand_sc, nc);
    return pick < 0 ? -1 : cand_id[pick];
    #undef MAX_CANDS
}

/* Generate a sentence. Emits decoded bytes into `out` (null-terminated,
 * truncated to max_len-1 bytes). Returns the number of tokens produced. */
int leo_generate(Leo *leo, char *out, int max_len) {
    if (!out || max_len < 2) return 0;
    out[0] = 0;

    int ctx[LEO_GEN_MAX];
    int n = 0;

    int start = leo_choose_start(leo);
    if (start < 0) { snprintf(out, max_len, "..."); return 0; }
    ctx[n++] = start;

    int target = LEO_GEN_TARGET + (rand() % 10) - 5;
    if (target < LEO_GEN_MIN) target = LEO_GEN_MIN;

    for (int t = 1; t < LEO_GEN_MAX; t++) {
        int prev1 = ctx[n - 1];
        int prev2 = n >= 2 ? ctx[n - 2] : -1;
        int nxt = leo_step_token(leo, prev2, prev1, 0.7f);
        if (nxt < 0) break;

        /* light repetition guard: kill immediate and back-to-back repeats */
        if (nxt == prev1) continue;
        if (n >= 2 && nxt == prev2 && prev1 == ctx[n - 2]) continue;

        ctx[n++] = nxt;

        /* stop on sentence boundary once we have enough material */
        if (n >= target && is_boundary_token(&leo->bpe, nxt)) break;
    }

    /* decode the emitted tokens */
    int pos = 0;
    for (int i = 0; i < n; i++) {
        char buf[LEO_MAX_TOKEN_LEN + 1];
        int len = bpe_decode_token(&leo->bpe, ctx[i], buf, sizeof(buf));
        if (pos + len >= max_len - 1) break;
        memcpy(out + pos, buf, len);
        pos += len;
    }
    out[pos] = 0;
    leo->step += n;
    return n;
}

void leo_stats(const Leo *leo) {
    printf("leo — v%s\n", LEO_VERSION);
    printf("  step:        %ld\n", leo->step);
    printf("  bpe:         %d vocab (%d merges)\n",
           leo->bpe.vocab_size, leo->bpe.n_merges);
    printf("  cooc:        %d / %d entries, %ld tokens seen\n",
           leo->cooc.n_entries, leo->cooc.capacity, leo->cooc.total_tokens);
    printf("  bigrams:     %d / %d entries\n",
           leo->bigrams.n_entries, leo->bigrams.capacity);
    printf("  trigrams:    %d / %d entries\n",
           leo->trigrams.n_entries, leo->trigrams.capacity);

    /* peek top 5 merges for sanity */
    if (leo->bpe.n_merges > 0) {
        printf("  top merges (last 5):\n");
        int start = leo->bpe.n_merges - 5;
        if (start < 0) start = 0;
        for (int m = start; m < leo->bpe.n_merges; m++) {
            char l[128], r[128];
            bpe_decode_token(&leo->bpe, leo->bpe.merges[m].a, l, sizeof(l));
            bpe_decode_token(&leo->bpe, leo->bpe.merges[m].b, r, sizeof(r));
            printf("    [%d] '%s' + '%s' → id %d\n",
                   m, l, r, leo->bpe.merges[m].new_id);
        }
    }
}

#ifndef LEO_LIB
int main(int argc, char **argv) {
    printf("neoleo — a small AI boy\n"
           "post-transformer. byte-level BPE. zero weights.\n\n");

    Leo leo;
    leo_init(&leo);

    const char *path = argc > 1 ? argv[1] : "leo.txt";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read error\n"); return 1; }
    buf[sz] = 0;
    fclose(f);

    printf("[ingest] %s — %ld bytes\n", path, sz);
    leo_ingest(&leo, buf);
    free(buf);

    printf("\n");
    leo_stats(&leo);

    /* speak a few sentences from the field — no prompt */
    printf("\n[speak] five sentences from the field:\n");
    for (int i = 0; i < 5; i++) {
        char reply[1024];
        leo_generate(&leo, reply, sizeof(reply));
        printf("  %d)%s\n", i + 1, reply);
    }

    leo_free(&leo);
    return 0;
}
#endif
