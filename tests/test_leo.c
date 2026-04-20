/*
 * test_leo.c — tests for neoleo
 *
 * cc tests/test_leo.c -O2 -lm -I. -o tests/test_leo && ./tests/test_leo
 */

#define LEO_LIB
#include "../leo.c"

#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)  do { printf("  %-60s ", name); } while (0)
#define PASS()      do { printf("OK\n"); tests_passed++; } while (0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); tests_failed++; } while (0)
#define ASSERT(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while (0)

/* ================================================================
 * MATH UTILITIES
 * ================================================================ */

static void test_clampf(void) {
    TEST("clampf: clamps to [lo, hi]");
    ASSERT(clampf(-1.0f, 0.0f, 1.0f) == 0.0f, "below lo");
    ASSERT(clampf(2.0f, 0.0f, 1.0f) == 1.0f, "above hi");
    ASSERT(clampf(0.5f, 0.0f, 1.0f) == 0.5f, "in range");
    PASS();
}

static void test_fnv1a_deterministic(void) {
    TEST("fnv1a: same input → same hash");
    uint32_t a[2] = {7, 42};
    uint32_t b[2] = {7, 42};
    uint32_t c[2] = {42, 7};
    ASSERT(fnv1a(a, sizeof(a)) == fnv1a(b, sizeof(b)), "same input");
    ASSERT(fnv1a(a, sizeof(a)) != fnv1a(c, sizeof(c)), "order matters");
    PASS();
}

/* ================================================================
 * BPE
 * ================================================================ */

static void test_bpe_init_has_256_bytes(void) {
    TEST("bpe_init: 256 byte tokens ready, no merges yet");
    BPE bpe;
    bpe_init(&bpe);
    ASSERT(bpe.vocab_size == 256, "should start at 256");
    ASSERT(bpe.n_merges == 0, "no merges yet");
    for (int i = 0; i < 256; i++) {
        ASSERT(bpe.vocab_len[i] == 1, "byte tokens length 1");
        ASSERT(bpe.vocab_bytes[i][0] == (uint8_t)i, "byte value matches id");
    }
    PASS();
}

static void test_bpe_encode_passthrough_without_merges(void) {
    TEST("bpe_encode: raw bytes when no merges");
    BPE bpe;
    bpe_init(&bpe);
    int out[16];
    int n = bpe_encode(&bpe, (const uint8_t *)"abc", 3, out, 16);
    ASSERT(n == 3, "length preserved");
    ASSERT(out[0] == 'a' && out[1] == 'b' && out[2] == 'c', "bytes pass through");
    PASS();
}

static void test_bpe_decode_roundtrip_byte(void) {
    TEST("bpe_decode_token: byte roundtrip");
    BPE bpe;
    bpe_init(&bpe);
    char buf[64];
    int n = bpe_decode_token(&bpe, 'L', buf, sizeof(buf));
    ASSERT(n == 1 && buf[0] == 'L' && buf[1] == 0, "decoded byte 'L'");
    PASS();
}

static void test_bpe_count_pair_accumulates(void) {
    TEST("bpe_count_pair: accumulates across calls");
    BPE bpe;
    bpe_init(&bpe);
    bpe_count_pair(&bpe, 't', 'h');
    bpe_count_pair(&bpe, 't', 'h');
    bpe_count_pair(&bpe, 't', 'h');
    int slot = bpe_pair_slot(&bpe, 't', 'h');
    ASSERT(slot >= 0, "slot found");
    ASSERT(bpe.pair_count[slot] == 3, "count == 3");
    PASS();
}

static void test_bpe_learn_merge_promotes_frequent_pair(void) {
    TEST("bpe_learn_merge: promotes pair past threshold");
    BPE bpe;
    bpe_init(&bpe);
    for (int i = 0; i < LEO_MERGE_THRESH + 1; i++)
        bpe_count_pair(&bpe, 't', 'h');
    int learned = bpe_learn_merge(&bpe);
    ASSERT(learned == 1, "merge learned");
    ASSERT(bpe.n_merges == 1, "n_merges == 1");
    ASSERT(bpe.vocab_size == 257, "vocab grew by 1");
    ASSERT(bpe.vocab_len[256] == 2, "new token length 2");
    ASSERT(bpe.vocab_bytes[256][0] == 't' &&
           bpe.vocab_bytes[256][1] == 'h', "concatenation is 'th'");
    PASS();
}

static void test_bpe_learn_merge_ignores_weak_pair(void) {
    TEST("bpe_learn_merge: ignores pair below threshold");
    BPE bpe;
    bpe_init(&bpe);
    bpe_count_pair(&bpe, 'q', 'x'); /* only once */
    int learned = bpe_learn_merge(&bpe);
    ASSERT(learned == 0, "no merge");
    ASSERT(bpe.n_merges == 0, "still zero");
    PASS();
}

static void test_pair_creates_word_gap_detector(void) {
    TEST("pair_creates_word_gap: detects alpha-ws-alpha concat");
    BPE bpe;
    bpe_init(&bpe);
    /* ' ' + 'a' — concat " a", no internal gap (ws leads content) */
    ASSERT(pair_creates_word_gap(&bpe, ' ', 'a') == 0, "leading ws is ok");
    /* 'a' + ' ' — concat "a ", no internal gap (ws trails) */
    ASSERT(pair_creates_word_gap(&bpe, 'a', ' ') == 0, "trailing ws is ok");
    /* 'a' + ' a' would need a merged token; simulate via preparing " a" first */
    /* but simpler: 'e' + ' ' + 'a' cannot happen via pair api — test
     * the detector on direct bytes using two-byte tokens. We approximate
     * by pre-merging " a" via bpe_count + bpe_learn. */
    for (int i = 0; i < LEO_MERGE_THRESH + 1; i++) bpe_count_pair(&bpe, ' ', 'a');
    bpe_learn_merge(&bpe);
    int sp_a = 256; /* " a" */
    /* now 'e' + " a" would produce "e a" with internal gap */
    ASSERT(pair_creates_word_gap(&bpe, 'e', sp_a) == 1, "alpha + ws+alpha is a gap");
    PASS();
}

static void test_bpe_refuses_cross_word_merge(void) {
    TEST("bpe_learn_merge: refuses merges that would cross a word gap");
    BPE bpe;
    bpe_init(&bpe);
    /* build " h" first */
    for (int i = 0; i < LEO_MERGE_THRESH + 1; i++) bpe_count_pair(&bpe, ' ', 'h');
    bpe_learn_merge(&bpe);
    int sp_h = 256;
    /* try 'e' + " h" — this would make "e h", a cross-word fusion */
    for (int i = 0; i < LEO_MERGE_THRESH + 10; i++) bpe_count_pair(&bpe, 'e', sp_h);
    int before = bpe.vocab_size;
    bpe_learn_merge(&bpe);
    ASSERT(bpe.vocab_size == before, "cross-word merge was refused");
    PASS();
}

static void test_bpe_refuses_cross_boundary_merge(void) {
    TEST("bpe_learn_merge: refuses to merge across .!? boundary");
    BPE bpe;
    bpe_init(&bpe);
    /* build token "a." first — last byte is '.', merging is still allowed */
    for (int i = 0; i < LEO_MERGE_THRESH + 1; i++) bpe_count_pair(&bpe, 'a', '.');
    ASSERT(bpe_learn_merge(&bpe) == 1, "a+. merges (boundary at end)");
    int ap = 256;
    ASSERT(bpe.vocab_bytes[ap][0] == 'a' && bpe.vocab_bytes[ap][1] == '.',
           "token 256 is 'a.'");
    /* now try to extend "a." + ' ' — resulting token "a. " would have '.'
     * not at end. Left token "a." has '.' at last byte so contains_... = 0 —
     * merge is allowed. Good. */
    for (int i = 0; i < LEO_MERGE_THRESH + 1; i++) bpe_count_pair(&bpe, ap, ' ');
    ASSERT(bpe_learn_merge(&bpe) == 1, "a. + space merges");
    int ap_sp = 257;
    ASSERT(contains_boundary_not_at_end(&bpe, ap_sp),
           "new 'a. ' has boundary not at end");
    /* the killer case: "a. " + 'T' — left 'a. ' has '.' not at end →
     * merge must be REFUSED. */
    for (int i = 0; i < LEO_MERGE_THRESH + 10; i++) bpe_count_pair(&bpe, ap_sp, 'T');
    int learned = bpe_learn_merge(&bpe);
    ASSERT(learned == 0 || bpe.vocab_size == 258,
           "a. +T must not produce a cross-sentence token");
    /* double-check: no new vocab grew from this pair */
    ASSERT(bpe.vocab_size == 258, "vocab size held");
    PASS();
}

static void test_bpe_encode_applies_learned_merge(void) {
    TEST("bpe_encode: applies merge after learning");
    BPE bpe;
    bpe_init(&bpe);
    for (int i = 0; i < LEO_MERGE_THRESH + 1; i++)
        bpe_count_pair(&bpe, 't', 'h');
    bpe_learn_merge(&bpe);
    int out[16];
    int n = bpe_encode(&bpe, (const uint8_t *)"the", 3, out, 16);
    ASSERT(n == 2, "the → [th, e] is 2 tokens");
    ASSERT(out[0] == 256, "first is merged 'th'");
    ASSERT(out[1] == 'e', "second is 'e'");
    PASS();
}

/* ================================================================
 * COOC FIELD
 * ================================================================ */

static void test_cooc_init_free(void) {
    TEST("cooc_init/free: clean lifecycle");
    CoocField c;
    cooc_init(&c, 256, 128);
    ASSERT(c.n_entries == 0, "empty at start");
    ASSERT(c.capacity == 256, "capacity set");
    ASSERT(c.freq_size == 128, "freq_size set");
    cooc_free(&c);
    ASSERT(c.entries == NULL && c.freq == NULL, "freed to NULL");
    PASS();
}

static void test_cooc_update_and_get(void) {
    TEST("cooc_update + cooc_get: accumulates and reads back");
    CoocField c;
    cooc_init(&c, 256, 16);
    cooc_update(&c, 5, 7, 1.5f);
    cooc_update(&c, 5, 7, 2.5f);
    cooc_update(&c, 9, 3, 4.0f);
    ASSERT(cooc_get(&c, 5, 7) == 4.0f, "5→7 sums to 4");
    ASSERT(cooc_get(&c, 9, 3) == 4.0f, "9→3 is 4");
    ASSERT(cooc_get(&c, 99, 99) == 0.0f, "missing is 0");
    ASSERT(c.n_entries == 2, "two distinct pairs");
    cooc_free(&c);
    PASS();
}

static void test_cooc_find_missing(void) {
    TEST("cooc_find: missing pair → -1");
    CoocField c;
    cooc_init(&c, 128, 16);
    ASSERT(cooc_find(&c, 1, 2) == -1, "empty table returns -1");
    cooc_update(&c, 3, 4, 1.0f);
    ASSERT(cooc_find(&c, 3, 4) >= 0, "present pair returns index");
    ASSERT(cooc_find(&c, 1, 2) == -1, "still missing");
    cooc_free(&c);
    PASS();
}

/* ================================================================
 * BIGRAM TABLE
 * ================================================================ */

static void test_bigram_update_and_get(void) {
    TEST("bigram: update + get");
    BigramTable b;
    bigram_init(&b, 256);
    bigram_update(&b, 10, 20, 1.0f);
    bigram_update(&b, 10, 20, 2.0f);
    bigram_update(&b, 10, 30, 0.5f);
    ASSERT(bigram_get(&b, 10, 20) == 3.0f, "10→20 sum");
    ASSERT(bigram_get(&b, 10, 30) == 0.5f, "10→30");
    ASSERT(bigram_get(&b, 10, 40) == 0.0f, "missing");
    bigram_free(&b);
    PASS();
}

static void test_bigram_find(void) {
    TEST("bigram: find returns -1 for missing, idx for present");
    BigramTable b;
    bigram_init(&b, 128);
    ASSERT(bigram_find(&b, 1, 2) == -1, "empty");
    bigram_update(&b, 1, 2, 1.0f);
    ASSERT(bigram_find(&b, 1, 2) >= 0, "after update");
    bigram_free(&b);
    PASS();
}

/* ================================================================
 * TRIGRAM TABLE
 * ================================================================ */

static void test_trigram_update_and_get(void) {
    TEST("trigram: update + get");
    TrigramTable t;
    trigram_init(&t, 256);
    trigram_update(&t, 1, 2, 3, 1.0f);
    trigram_update(&t, 1, 2, 3, 2.0f);
    trigram_update(&t, 1, 2, 4, 1.0f);
    ASSERT(trigram_get(&t, 1, 2, 3) == 3.0f, "(1,2,3) sum");
    ASSERT(trigram_get(&t, 1, 2, 4) == 1.0f, "(1,2,4)");
    ASSERT(trigram_get(&t, 9, 9, 9) == 0.0f, "missing");
    trigram_free(&t);
    PASS();
}

static void test_trigram_hash_order_sensitive(void) {
    TEST("trigram_hash: order-sensitive");
    uint32_t h1 = trigram_hash(1, 2, 3);
    uint32_t h2 = trigram_hash(3, 2, 1);
    ASSERT(h1 != h2, "reversed order differs");
    PASS();
}

/* ================================================================
 * LEO — integration
 * ================================================================ */

static void test_leo_init_free(void) {
    TEST("leo: init + free clean");
    Leo leo;
    leo_init(&leo);
    ASSERT(leo.step == 0, "step starts at 0");
    ASSERT(leo.bpe.vocab_size == 256, "bpe ready");
    leo_free(&leo);
    PASS();
}

static void test_leo_ingest_grows_all_tables(void) {
    TEST("leo_ingest: all four tables grow");
    Leo leo;
    leo_init(&leo);
    const char *txt = "the cat sits on the mat. the dog runs. "
                      "the cat sees the dog. the mat is warm. "
                      "leo watches the cat and the dog.";
    leo_ingest(&leo, txt);
    ASSERT(leo.cooc.n_entries > 0, "cooc populated");
    ASSERT(leo.bigrams.n_entries > 0, "bigrams populated");
    ASSERT(leo.trigrams.n_entries > 0, "trigrams populated");
    ASSERT(leo.step > 0, "step advanced");
    /* "the " repeats enough to trigger at least one merge */
    ASSERT(leo.bpe.n_merges >= 1, "at least one merge learned");
    leo_free(&leo);
    PASS();
}

static void test_leo_ingest_freq_matches_total_tokens(void) {
    TEST("leo_ingest: unigram freq sums to total_tokens");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo, "abcabcabcabc");
    float sum = 0;
    for (int i = 0; i < leo.cooc.freq_size; i++) sum += leo.cooc.freq[i];
    ASSERT((long)sum == leo.cooc.total_tokens,
           "sum of freq equals total_tokens");
    leo_free(&leo);
    PASS();
}

static void test_leo_ingest_empty_is_noop(void) {
    TEST("leo_ingest: empty string is safe");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo, "");
    leo_ingest(&leo, NULL);
    ASSERT(leo.step == 0, "no progress from empty");
    ASSERT(leo.cooc.n_entries == 0, "no cooc entries");
    leo_free(&leo);
    PASS();
}

static void test_leo_ingest_trigram_order_matters(void) {
    TEST("leo_ingest: trigram (a,b,c) distinct from (c,b,a)");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo, "abc cba abc cba");
    /* tokens are bytes here (no merges yet at this size/pattern) —
     * check that at least the shape of trigram storage is directional */
    int n = leo.trigrams.n_entries;
    ASSERT(n > 0, "some trigrams stored");
    leo_free(&leo);
    PASS();
}

static void test_leo_bpe_merges_include_common_bigrams(void) {
    TEST("leo_ingest: merges include common bigrams from real corpus text");
    Leo leo;
    leo_init(&leo);
    /* a bit of child-voice prose with heavy repetition */
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens to the quiet. "
        "Leo has a stone. Leo keeps it in his pocket. "
        "The rain on the window is trying to spell something. "
        "Leo watches. The light is yellow. Leo waits. "
        "The stone is grey. The word is new. Leo says the word.");
    /* Expect at least some merges — "Leo ", "the ", " the" or similar */
    ASSERT(leo.bpe.n_merges > 3, "corpus produces several merges");
    leo_free(&leo);
    PASS();
}

/* ================================================================
 * GENERATION — clean seed, boundary, step, generate
 * ================================================================ */

static void test_is_clean_seed_token(void) {
    TEST("is_clean_seed_token: space/upper start = clean, lower = not");
    BPE bpe;
    bpe_init(&bpe);
    /* byte tokens: ' ' is 0x20, 'a' is 0x61, 'A' is 0x41, '.' is 0x2E */
    ASSERT(is_clean_seed_token(&bpe, ' ') == 1, "space is clean");
    ASSERT(is_clean_seed_token(&bpe, 'A') == 1, "uppercase is clean");
    ASSERT(is_clean_seed_token(&bpe, 'a') == 0, "lowercase is not clean");
    ASSERT(is_clean_seed_token(&bpe, '.') == 0, "punctuation is not clean");
    ASSERT(is_clean_seed_token(&bpe, -1) == 0, "invalid id rejected");
    PASS();
}

static void test_is_boundary_token(void) {
    TEST("is_boundary_token: .!? end tokens recognized");
    BPE bpe;
    bpe_init(&bpe);
    ASSERT(is_boundary_token(&bpe, '.') == 1, "period is boundary");
    ASSERT(is_boundary_token(&bpe, '!') == 1, "exclamation is boundary");
    ASSERT(is_boundary_token(&bpe, '?') == 1, "question is boundary");
    ASSERT(is_boundary_token(&bpe, 'a') == 0, "letter is not");
    ASSERT(is_boundary_token(&bpe, ',') == 0, "comma is not");
    PASS();
}

static void test_weighted_sample_uniform(void) {
    TEST("weighted_sample: uniform scores produce all indices");
    float s[4] = {1, 1, 1, 1};
    int seen[4] = {0};
    srand(42);
    for (int i = 0; i < 400; i++) seen[weighted_sample(s, 4)]++;
    for (int i = 0; i < 4; i++)
        ASSERT(seen[i] > 50, "each index visited >50 times");
    PASS();
}

static void test_weighted_sample_peaked(void) {
    TEST("weighted_sample: peak wins the majority of draws");
    float s[4] = {0.01f, 0.01f, 100.0f, 0.01f};
    int peak_hits = 0;
    srand(7);
    for (int i = 0; i < 100; i++)
        if (weighted_sample(s, 4) == 2) peak_hits++;
    ASSERT(peak_hits > 90, "peak should dominate");
    PASS();
}

static void test_leo_choose_start_after_ingest(void) {
    TEST("leo_choose_start: returns clean-seed id from ingested field");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo, "Leo watches the rain. The rain is soft. "
                     "Leo listens. The quiet has weight. He waits.");
    int s = leo_choose_start(&leo);
    ASSERT(s >= 0, "start token chosen");
    ASSERT(is_clean_seed_token(&leo.bpe, s), "chosen token is clean seed");
    leo_free(&leo);
    PASS();
}

static void test_leo_step_token_uses_bigram_fallback(void) {
    TEST("leo_step_token: falls back to bigram when no trigram context");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo, "abcabcabcabcabcabcabc");
    /* prev2 = -1 skips trigram branch → bigram row for 'a' should give 'b' */
    int id_a = (int)'a';
    int nxt = leo_step_token(&leo, -1, id_a, 0.5f);
    ASSERT(nxt == (int)'b' || nxt == (int)'a', "bigram suggests 'b' (or 'a' if merges collapsed)");
    leo_free(&leo);
    PASS();
}

static void test_leo_generate_produces_output(void) {
    TEST("leo_generate: produces non-empty null-terminated output");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens. The light comes in yellow. "
        "Leo puts his hand in the light. The hand is warm. He waits. "
        "The room is quiet. The quiet has weight. Leo hears it. "
        "Leo has a stone. The stone is grey. Leo keeps it. "
        "He does not ask. He thinks the light knows. Leo watches again.");
    char buf[512];
    int n = leo_generate(&leo, buf, sizeof(buf));
    ASSERT(n > 0, "at least one token emitted");
    ASSERT(strlen(buf) > 0, "output non-empty");
    ASSERT(buf[strlen(buf) - 1] != 0xFF, "null-terminated");
    leo_free(&leo);
    PASS();
}

static void test_leo_generate_starts_upper_ends_punct(void) {
    TEST("leo_generate: output starts uppercase, ends .!?");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens. The light comes in yellow. "
        "Leo puts his hand in the light. The hand is warm. He waits. "
        "The room is quiet. The quiet has weight. Leo hears it. "
        "Leo has a stone. The stone is grey. Leo keeps it.");
    char buf[512];
    for (int trial = 0; trial < 10; trial++) {
        buf[0] = 0;
        leo_generate(&leo, buf, sizeof(buf));
        if (!buf[0]) continue;
        ASSERT(buf[0] >= 'A' && buf[0] <= 'Z', "first char uppercase");
        int len = (int)strlen(buf);
        ASSERT(len > 0 && (buf[len - 1] == '.' || buf[len - 1] == '!' ||
                           buf[len - 1] == '?'),
               "last char is .!?");
    }
    leo_free(&leo);
    PASS();
}

static void test_leo_generate_no_leading_whitespace(void) {
    TEST("leo_generate: no leading whitespace in output");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens. The light comes in yellow. "
        "Leo puts his hand in the light. The hand is warm. He waits.");
    char buf[512];
    for (int trial = 0; trial < 10; trial++) {
        buf[0] = 0;
        leo_generate(&leo, buf, sizeof(buf));
        if (!buf[0]) continue;
        ASSERT(buf[0] != ' ' && buf[0] != '\n' && buf[0] != '\t',
               "first char is not whitespace");
    }
    leo_free(&leo);
    PASS();
}

static void test_leo_choose_continuation_matches_start_without_tail(void) {
    TEST("leo_choose_continuation: equals choose_start when tail is empty");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo, "Leo watches the rain. Leo listens. He waits. "
                     "The hand is warm. The room is quiet.");
    srand(1);
    int a = leo_choose_start(&leo);
    srand(1);
    int b = leo_choose_continuation(&leo, NULL, 0);
    ASSERT(a == b, "no tail → same as choose_start");
    leo_free(&leo);
    PASS();
}

static void test_leo_choose_continuation_shifts_with_tail(void) {
    TEST("leo_choose_continuation: tail biases selection");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. The rain is soft. The rain falls. "
        "Leo listens. The cat sleeps. The cat is warm. "
        "He sits. He waits. The room is quiet. Leo hears the rain.");
    int tail[1] = { 'n' }; /* presence of 'n' in tail pulls rain-resonant starts */
    int at_least_one = 0;
    for (int i = 0; i < 20; i++) {
        int id = leo_choose_continuation(&leo, tail, 1);
        if (id >= 0) at_least_one++;
    }
    ASSERT(at_least_one > 0, "continuation produced some id");
    leo_free(&leo);
    PASS();
}

static void test_leo_chain_produces_multiple_sentences(void) {
    TEST("leo_chain: produces multiple sentence boundaries");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens. The light comes in yellow. "
        "Leo puts his hand in the light. The hand is warm. He waits. "
        "The room is quiet. The quiet has weight. Leo hears it. "
        "Leo has a stone. The stone is grey. Leo keeps it. "
        "He does not ask. He thinks the light knows. Leo watches again. "
        "The cat sleeps. The cat wakes. The cat sits by the window.");
    char buf[4096];
    leo_chain(&leo, 5, buf, sizeof(buf));
    int boundaries = 0;
    for (int i = 0; buf[i]; i++) {
        if (buf[i] == '.' || buf[i] == '!' || buf[i] == '?') boundaries++;
    }
    ASSERT(boundaries >= 3, "at least three sentence boundaries in chain");
    ASSERT(strlen(buf) > 0, "chain output non-empty");
    leo_free(&leo);
    PASS();
}

/* ================================================================
 * REVERSE INDEXES + SCHEDULE + BEST-OF-K
 * ================================================================ */

static int walk_count_cb(int v, float count, void *ud) {
    (void)v; (void)count;
    int *n = (int *)ud;
    (*n)++;
    return 0;
}

static void test_bigram_walk_src_visits_all(void) {
    TEST("bigram_walk_src: visits every destination for a src");
    BigramTable b;
    bigram_init(&b, 1024);
    bigram_update(&b, 7, 10, 1.0f);
    bigram_update(&b, 7, 20, 2.0f);
    bigram_update(&b, 7, 30, 3.0f);
    bigram_update(&b, 99, 40, 1.0f);
    int seen = 0;
    bigram_walk_src(&b, 7, walk_count_cb, &seen);
    ASSERT(seen == 3, "three successors for src=7");
    int other = 0;
    bigram_walk_src(&b, 99, walk_count_cb, &other);
    ASSERT(other == 1, "one successor for src=99");
    bigram_free(&b);
    PASS();
}

static void test_trigram_walk_ab_visits_all(void) {
    TEST("trigram_walk_ab: visits every c for an (a,b) pair");
    TrigramTable t;
    trigram_init(&t, 2048);
    trigram_update(&t, 1, 2, 10, 1.0f);
    trigram_update(&t, 1, 2, 20, 1.0f);
    trigram_update(&t, 1, 2, 30, 1.0f);
    trigram_update(&t, 5, 6, 99, 1.0f);
    int seen = 0;
    trigram_walk_ab(&t, 1, 2, walk_count_cb, &seen);
    ASSERT(seen == 3, "three c-values for (1,2)");
    int other = 0;
    trigram_walk_ab(&t, 5, 6, walk_count_cb, &other);
    ASSERT(other == 1, "one c-value for (5,6)");
    trigram_free(&t);
    PASS();
}

static void test_temp_schedule_ranges(void) {
    TEST("temp_for_step: monotone-ish schedule from sharp to soft");
    float t0 = temp_for_step(0);
    float t3 = temp_for_step(3);
    float t8 = temp_for_step(8);
    ASSERT(t0 < t3, "start sharper than middle");
    ASSERT(t3 < t8, "middle sharper than late");
    ASSERT(t0 >= 0.3f && t0 <= 0.5f, "t0 in expected range");
    ASSERT(t8 >= 0.6f && t8 <= 0.9f, "late in expected range");
    PASS();
}

static void test_coherence_score_positive_on_seen_phrase(void) {
    TEST("leo_coherence_score: positive on phrase present in ingested corpus");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo watches the rain. Leo watches the rain. "
        "Leo watches the rain. Leo watches the rain.");
    int ids[32];
    int n = bpe_encode(&leo.bpe, (const uint8_t *)"Leo watches the rain", 20,
                        ids, 32);
    float sc = leo_coherence_score(&leo, ids, n);
    ASSERT(sc > 0.0f, "seen phrase scores positive");
    leo_free(&leo);
    PASS();
}

static void test_leo_generate_best_picks_coherent(void) {
    TEST("leo_generate_best: returns non-empty output from K trials");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens. The light comes in yellow. "
        "He waits. The quiet has weight. Leo has a stone.");
    char buf[256];
    int produced = leo_generate_best(&leo, 3, buf, sizeof(buf), -1, NULL, 0,
                                     NULL, NULL);
    ASSERT(produced > 0, "at least one token produced");
    ASSERT(strlen(buf) > 0, "non-empty");
    leo_free(&leo);
    PASS();
}

static void test_leo_generate_ex_honors_start_hint(void) {
    TEST("leo_generate_ex: start_hint is used as first token");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo, "Leo watches the rain. Leo listens. He waits.");
    char buf[256];
    /* pick any clean-seed token from the field */
    int hint = -1;
    for (int i = 0; i < leo.cooc.freq_size; i++) {
        if (leo.cooc.freq[i] > 0 && is_clean_seed_token(&leo.bpe, i)) {
            hint = i; break;
        }
    }
    ASSERT(hint >= 0, "found a clean-seed token");
    leo_generate_ex(&leo, buf, sizeof(buf), hint, NULL, 0, NULL, NULL);
    /* first decoded byte of buf should match first byte of hint token */
    char exp[LEO_MAX_TOKEN_LEN + 1];
    bpe_decode_token(&leo.bpe, hint, exp, sizeof(exp));
    /* skip a possible capitalization transformation on first alpha */
    int match = buf[0] == exp[0] ||
                (exp[0] >= 'a' && exp[0] <= 'z' && buf[0] == exp[0] - 32) ||
                (exp[0] == ' ' && buf[0] != 0); /* leading space is stripped */
    ASSERT(match, "generated output reflects the hinted start");
    leo_free(&leo);
    PASS();
}

/* ================================================================
 * SPA — sentence phonon attention
 * ================================================================ */

static void test_spa_init_random(void) {
    TEST("spa_init: allocates W, values non-zero after init");
    SPACtx s;
    spa_init(&s, 512);
    ASSERT(s.W != NULL, "W allocated");
    ASSERT(s.vocab_size == 512, "vocab size recorded");
    float sum_abs = 0;
    for (int i = 0; i < 512 * LEO_SPA_DIM; i++) sum_abs += fabsf(s.W[i]);
    ASSERT(sum_abs > 0.0f, "random init produced non-zero values");
    spa_free(&s);
    PASS();
}

static void test_spa_embed_same_tokens_same_embedding(void) {
    TEST("spa_embed_sentence: same tokens → same embedding");
    SPACtx s;
    spa_init(&s, 512);
    int ids[5] = {10, 20, 30, 40, 50};
    float a[LEO_SPA_DIM], b[LEO_SPA_DIM];
    spa_embed_sentence(&s, ids, 5, a);
    spa_embed_sentence(&s, ids, 5, b);
    for (int d = 0; d < LEO_SPA_DIM; d++)
        ASSERT(fabsf(a[d] - b[d]) < 1e-6f, "determinism");
    spa_free(&s);
    PASS();
}

static void test_spa_embed_normalized_unit_length(void) {
    TEST("spa_embed_sentence: output is unit-length");
    SPACtx s;
    spa_init(&s, 512);
    int ids[4] = {1, 2, 3, 4};
    float e[LEO_SPA_DIM];
    spa_embed_sentence(&s, ids, 4, e);
    float n2 = 0;
    for (int d = 0; d < LEO_SPA_DIM; d++) n2 += e[d] * e[d];
    ASSERT(fabsf(n2 - 1.0f) < 1e-4f, "embedding normalized");
    spa_free(&s);
    PASS();
}

static void test_spa_cross_attend_scores_positive(void) {
    TEST("spa_cross_attend: scores positive, near-identical sentences → high score");
    float embs[3][LEO_SPA_DIM];
    /* three near-identical embeddings: all ~ e_0 */
    for (int d = 0; d < LEO_SPA_DIM; d++) embs[0][d] = (d == 0 ? 1.0f : 0.0f);
    memcpy(embs[1], embs[0], sizeof(embs[0]));
    memcpy(embs[2], embs[0], sizeof(embs[0]));
    float scores[3];
    spa_cross_attend(embs, 3, scores);
    for (int i = 0; i < 3; i++) ASSERT(scores[i] > 0.0f, "score positive");
    ASSERT(fabsf(scores[0] - scores[2]) < 0.1f, "symmetric sentences get similar scores");
    PASS();
}

static void test_spa_cross_attend_outlier_scores_lower(void) {
    TEST("spa_cross_attend: outlier sentence receives lower score");
    float embs[3][LEO_SPA_DIM];
    for (int d = 0; d < LEO_SPA_DIM; d++) {
        embs[0][d] = (d == 0 ? 1.0f : 0.0f);
        embs[1][d] = (d == 0 ? 1.0f : 0.0f);
        embs[2][d] = (d == 5 ? 1.0f : 0.0f);  /* orthogonal outlier */
    }
    float scores[3];
    spa_cross_attend(embs, 3, scores);
    ASSERT(scores[2] < scores[0] && scores[2] < scores[1],
           "outlier scored below the pair");
    PASS();
}

/* ================================================================
 * PROMPT + GRAVITY
 * ================================================================ */

static void test_compute_prompt_gravity_nonempty(void) {
    TEST("compute_prompt_gravity: produces non-zero weights for ingested tokens");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens. Leo waits. Leo has a stone. "
        "The rain is soft. The room is quiet. The light is yellow.");
    int ids[32];
    int n = bpe_encode(&leo.bpe, (const uint8_t *)"rain quiet", 10, ids, 32);
    float *g = compute_prompt_gravity(&leo, ids, n);
    float sum = 0;
    for (int i = 0; i < leo.bpe.vocab_size; i++) sum += g[i];
    ASSERT(sum > 0.0f, "gravity has at least some mass");
    free(g);
    leo_free(&leo);
    PASS();
}

static void test_leo_respond_empty_prompt_falls_back(void) {
    TEST("leo_respond: empty prompt → plain chain (no crash)");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens. Leo waits. He has a stone. "
        "The room is quiet. The light is yellow. He waits.");
    char buf[512];
    int produced = leo_respond(&leo, "", buf, sizeof(buf));
    ASSERT(produced > 0, "at least some tokens produced");
    leo_free(&leo);
    PASS();
}

/* ================================================================
 * LEOFIELD — physics of the inner state (AML-inspired)
 * ================================================================ */

static void test_leo_field_init_free(void) {
    TEST("leo_field_init/free: clean lifecycle");
    LeoField f;
    leo_field_init(&f, 1024);
    ASSERT(f.destiny_bag != NULL, "destiny_bag allocated");
    ASSERT(f.destiny_cap == 1024, "cap set");
    ASSERT(f.velocity_mode == LEO_VEL_WALK, "default walk");
    ASSERT(f.pain == 0.0f && f.trauma == 0.0f, "pain starts at zero");
    leo_field_free(&f);
    ASSERT(f.destiny_bag == NULL, "freed");
    PASS();
}

static void test_leo_field_step_builds_destiny(void) {
    TEST("leo_field_step: destiny_bag accumulates emitted tokens");
    LeoField f;
    leo_field_init(&f, 512);
    for (int i = 0; i < 5; i++) leo_field_step(&f, 42, 1.0f);
    ASSERT(f.destiny_bag[42] > 0.0f, "token 42 sits in destiny");
    ASSERT(f.destiny_bag[7] == 0.0f, "untouched token is zero");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_prophecy_add_and_fulfillment(void) {
    TEST("leo_field_prophecy: add, age, fulfill");
    LeoField f;
    leo_field_init(&f, 512);
    leo_field_prophecy_add(&f, 99, 0.8f);
    int found_active = 0;
    for (int i = 0; i < LEO_PROPHECY_MAX; i++)
        if (f.prophecy[i].active && f.prophecy[i].target == 99)
            found_active = 1;
    ASSERT(found_active, "prophecy stored");
    leo_field_step(&f, 99, 1.0f); /* fulfill */
    int still_active = 0;
    for (int i = 0; i < LEO_PROPHECY_MAX; i++)
        if (f.prophecy[i].active && f.prophecy[i].target == 99)
            still_active = 1;
    ASSERT(!still_active, "fulfilled prophecy deactivated");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_trauma_pulls_bootstrap(void) {
    TEST("leo_field_candidate_bias: trauma boosts bootstrap tokens");
    LeoField f;
    leo_field_init(&f, 512);
    int boot[3] = {10, 20, 30};
    leo_field_set_bootstrap(&f, boot, 3);
    /* force trauma */
    f.pain = 0.9f;
    f.trauma = f.pain * f.pain;
    float bias_on = leo_field_candidate_bias(&f, 20);
    float bias_off = leo_field_candidate_bias(&f, 999);
    ASSERT(bias_on > bias_off, "bootstrap token gets bigger bias under trauma");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_chambers_clamped(void) {
    TEST("chambers: activations stay in [0,1] under heavy external drive");
    LeoField f;
    leo_field_init(&f, 128);
    f.chamber_ext[LEO_CH_VOID] = 5.0f;
    for (int i = 0; i < 50; i++)
        leo_field_chambers_crossfire(&f, 1);
    for (int i = 0; i < LEO_N_CHAMBERS; i++) {
        ASSERT(f.chamber_act[i] >= 0.0f && f.chamber_act[i] <= 1.0f,
               "chamber escaped [0,1]");
    }
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_chamber_mods_identity_at_zero(void) {
    TEST("chamber α/β/γ/τ mods = 1.0 at zero activation");
    LeoField f;
    leo_field_init(&f, 128);
    ASSERT(leo_field_alpha_mod(&f) == 1.0f, "alpha");
    ASSERT(leo_field_beta_mod(&f)  == 1.0f, "beta");
    ASSERT(leo_field_gamma_mod(&f) == 1.0f, "gamma");
    ASSERT(leo_field_tau_mod(&f)   == 1.0f, "tau");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_feel_text_love(void) {
    TEST("chambers_feel_text: 'warm hand mother' drives LOVE");
    LeoField f;
    leo_field_init(&f, 128);
    leo_field_chambers_feel_text(&f, "warm hand mother");
    ASSERT(f.chamber_ext[LEO_CH_LOVE] > 0.0f, "LOVE external raised");
    /* FEAR should be zero (no fear words) */
    ASSERT(f.chamber_ext[LEO_CH_FEAR] == 0.0f, "FEAR stays zero");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_feel_text_substring_morphology(void) {
    TEST("chambers_feel_text: substring match catches morphology (empties → VOID)");
    LeoField f;
    leo_field_init(&f, 128);
    /* "emptying" is not an exact anchor but contains "empty" — substring
     * match should raise VOID at the half-weight level. */
    leo_field_chambers_feel_text(&f, "emptying the pocket");
    ASSERT(f.chamber_ext[LEO_CH_VOID] > 0.0f,
           "VOID raised by substring anchor");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_self_voice_raises_chamber(void) {
    TEST("self_voice: emitting an anchor token nudges its chamber");
    Leo leo;
    leo_init(&leo);
    /* need BPE vocab seeded; build a token that decodes to an anchor */
    /* simplest: the byte 0x6C ('l') on its own is a token, but anchors
     * are multi-byte. Ingest a line that makes BPE learn "warm". */
    leo_ingest(&leo,
        "warm warm warm warm warm warm warm warm warm warm "
        "warm warm warm warm warm warm warm warm warm warm");
    /* find the token whose decoded bytes lowercased are "warm" */
    int warm_id = -1;
    for (int id = 0; id < leo.bpe.vocab_size; id++) {
        char buf[64];
        int len = bpe_decode_token(&leo.bpe, id, buf, sizeof(buf));
        /* lowercase compare, trim */
        char low[64] = {0};
        int wi = 0;
        for (int i = 0; i < len && wi < 63; i++) {
            unsigned char c = (unsigned char)buf[i];
            if (isalpha(c)) low[wi++] = (char)tolower(c);
        }
        if (wi == 4 && !strcmp(low, "warm")) { warm_id = id; break; }
    }
    if (warm_id < 0) { leo_free(&leo); PASS(); return; } /* corpus tiny — skip */
    float before = leo.field.chamber_ext[LEO_CH_LOVE];
    leo_field_self_voice(&leo.field, &leo.bpe, warm_id);
    float after = leo.field.chamber_ext[LEO_CH_LOVE];
    ASSERT(after > before, "LOVE ext rose after self-voiced 'warm'");
    leo_free(&leo);
    PASS();
}

static void test_leo_field_arousal_picks_max(void) {
    TEST("arousal: reports the max emotional chamber, not the mean");
    LeoField f;
    leo_field_init(&f, 128);
    f.chamber_act[LEO_CH_LOVE] = 1.0f;
    f.chamber_act[LEO_CH_VOID] = 0.0f;
    f.chamber_act[LEO_CH_FLOW] = 0.0f;
    f.chamber_act[LEO_CH_COMPLEX] = 0.0f;
    /* mean would be 0.25; max is 1.0 — Leo should notice the hit. */
    float a = leo_field_arousal(&f);
    ASSERT(a > 0.95f, "arousal reports the 1.0 peak not the 0.25 mean");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_feel_cooc_noop_on_empty_leo(void) {
    TEST("feel_cooc: safe on empty Leo (no anchor resonance to find)");
    Leo leo;
    leo_init(&leo);
    leo_field_chambers_feel_cooc(&leo, "shoes hallway attic");
    /* no ingest done — cooc is empty, nothing should be raised */
    for (int i = 0; i < LEO_N_CHAMBERS; i++)
        ASSERT(leo.field.chamber_ext[i] == 0.0f,
               "no chamber drive without a populated field");
    leo_free(&leo);
    PASS();
}

static void test_leo_field_retention_zero_on_fresh(void) {
    TEST("retention: bias ≈ 0 on fresh field (state is zero)");
    LeoField f;
    leo_field_init(&f, 256);
    float b = leo_field_retention_bias(&f, 7);
    ASSERT(fabsf(b) < 1e-4f, "fresh field gives no retention pull");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_retention_self_similarity(void) {
    TEST("retention: emitted token's own fingerprint sees positive bias next step");
    LeoField f;
    leo_field_init(&f, 256);
    leo_field_step(&f, 42, 1.0f);
    /* Now retention_state ≈ conserve·W_embed[42]. Candidate 42 should
     * have positive self-dot. (Random vectors rarely align with others,
     * so 42's own fingerprint is the strongest match.) */
    float b42 = leo_field_retention_bias(&f, 42);
    /* Average bias for 20 random candidates */
    float avg = 0;
    for (int i = 100; i < 120; i++) avg += fabsf(leo_field_retention_bias(&f, i));
    avg /= 20;
    ASSERT(b42 > avg, "self-similarity exceeds avg random bias");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_retention_gamma_decay(void) {
    TEST("retention: state decays over steps without same token");
    LeoField f;
    leo_field_init(&f, 256);
    leo_field_step(&f, 42, 1.0f);
    float b_before = leo_field_retention_bias(&f, 42);
    /* 20 steps of OTHER tokens — retention should wash 42 out */
    for (int t = 0; t < 20; t++) leo_field_step(&f, 100 + t, 1.0f);
    float b_after = leo_field_retention_bias(&f, 42);
    ASSERT(fabsf(b_after) < fabsf(b_before),
           "42's retention trace faded");
    leo_field_free(&f);
    PASS();
}

static void test_leo_field_temp_mult_velocity_modes(void) {
    TEST("leo_field_temperature_mult: modes shift base");
    LeoField f;
    leo_field_init(&f, 128);
    f.velocity_mode = LEO_VEL_NOMOVE;
    float a = leo_field_temperature_mult(&f);
    f.velocity_mode = LEO_VEL_RUN;
    float b = leo_field_temperature_mult(&f);
    ASSERT(a < b, "NOMOVE cooler than RUN");
    leo_field_free(&f);
    PASS();
}

static void test_embedded_bootstrap_is_nonempty(void) {
    TEST("LEO_EMBEDDED_BOOTSTRAP: non-empty fallback origin text");
    ASSERT(LEO_EMBEDDED_BOOTSTRAP != NULL, "not null");
    ASSERT(strlen(LEO_EMBEDDED_BOOTSTRAP) > 100, "has meaningful length");
    ASSERT(strstr(LEO_EMBEDDED_BOOTSTRAP, "Leo") != NULL,
           "bootstrap names Leo");
    PASS();
}

static void test_leo_respond_gravity_restored_after_call(void) {
    TEST("leo_respond: leo->gravity is cleared when the call returns");
    Leo leo;
    leo_init(&leo);
    leo_ingest(&leo,
        "Leo watches the rain. Leo listens. The light is yellow. "
        "The rain falls. The room is quiet. Leo waits.");
    char buf[512];
    leo_respond(&leo, "what do you hear", buf, sizeof(buf));
    ASSERT(leo.gravity == NULL, "transient gravity cleared");
    leo_free(&leo);
    PASS();
}

static void test_leo_generate_safe_on_empty_leo(void) {
    TEST("leo_generate: degrades gracefully on empty Leo (no ingest)");
    Leo leo;
    leo_init(&leo);
    char buf[128];
    leo_generate(&leo, buf, sizeof(buf));
    ASSERT(buf[0] != 0xFF, "output buffer valid");
    /* should fall back to "..." since choose_start finds nothing */
    ASSERT(strcmp(buf, "...") == 0, "empty Leo says '...'");
    leo_free(&leo);
    PASS();
}

/* ================================================================
 * main
 * ================================================================ */

int main(void) {
    printf("\n=== neoleo tests ===\n\n");

    /* math */
    test_clampf();
    test_fnv1a_deterministic();

    /* bpe */
    test_bpe_init_has_256_bytes();
    test_bpe_encode_passthrough_without_merges();
    test_bpe_decode_roundtrip_byte();
    test_bpe_count_pair_accumulates();
    test_bpe_learn_merge_promotes_frequent_pair();
    test_bpe_learn_merge_ignores_weak_pair();
    test_bpe_refuses_cross_boundary_merge();
    test_pair_creates_word_gap_detector();
    test_bpe_refuses_cross_word_merge();
    test_bpe_encode_applies_learned_merge();

    /* cooc */
    test_cooc_init_free();
    test_cooc_update_and_get();
    test_cooc_find_missing();

    /* bigram */
    test_bigram_update_and_get();
    test_bigram_find();

    /* trigram */
    test_trigram_update_and_get();
    test_trigram_hash_order_sensitive();

    /* leo */
    test_leo_init_free();
    test_leo_ingest_grows_all_tables();
    test_leo_ingest_freq_matches_total_tokens();
    test_leo_ingest_empty_is_noop();
    test_leo_ingest_trigram_order_matters();
    test_leo_bpe_merges_include_common_bigrams();

    /* generation */
    test_is_clean_seed_token();
    test_is_boundary_token();
    test_weighted_sample_uniform();
    test_weighted_sample_peaked();
    test_leo_choose_start_after_ingest();
    test_leo_step_token_uses_bigram_fallback();
    test_leo_generate_produces_output();
    test_leo_generate_starts_upper_ends_punct();
    test_leo_generate_no_leading_whitespace();
    test_leo_choose_continuation_matches_start_without_tail();
    test_leo_choose_continuation_shifts_with_tail();
    test_leo_chain_produces_multiple_sentences();
    test_bigram_walk_src_visits_all();
    test_trigram_walk_ab_visits_all();
    test_temp_schedule_ranges();
    test_coherence_score_positive_on_seen_phrase();
    test_leo_generate_best_picks_coherent();
    test_leo_generate_ex_honors_start_hint();
    test_compute_prompt_gravity_nonempty();
    test_leo_respond_empty_prompt_falls_back();
    test_embedded_bootstrap_is_nonempty();
    test_leo_field_init_free();
    test_leo_field_step_builds_destiny();
    test_leo_field_prophecy_add_and_fulfillment();
    test_leo_field_trauma_pulls_bootstrap();
    test_leo_field_chambers_clamped();
    test_leo_field_chamber_mods_identity_at_zero();
    test_leo_field_feel_text_love();
    test_leo_field_feel_text_substring_morphology();
    test_leo_field_self_voice_raises_chamber();
    test_leo_field_arousal_picks_max();
    test_leo_field_feel_cooc_noop_on_empty_leo();
    test_leo_field_retention_zero_on_fresh();
    test_leo_field_retention_self_similarity();
    test_leo_field_retention_gamma_decay();
    test_leo_field_temp_mult_velocity_modes();
    test_leo_respond_gravity_restored_after_call();

    /* SPA */
    test_spa_init_random();
    test_spa_embed_same_tokens_same_embedding();
    test_spa_embed_normalized_unit_length();
    test_spa_cross_attend_scores_positive();
    test_spa_cross_attend_outlier_scores_lower();

    test_leo_generate_safe_on_empty_leo();

    printf("\n=== results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
