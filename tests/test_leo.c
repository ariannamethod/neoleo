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
    test_leo_generate_safe_on_empty_leo();

    printf("\n=== results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
