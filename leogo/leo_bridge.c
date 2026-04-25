/*
 * leo_bridge.c — cgo bridge between Go and the C core of neoleo.
 *
 * Thin shim. Compiles leo.c in library mode (LEO_LIB wraps main()),
 * exposes heap-allocated Leo + pointer-safe accessors.
 *
 * Step 29b: only the reply path is exposed. Ring functions
 * (leo_generate_ring / leo_observe_thought / leo_pulse) are
 * wrapped in 29c once the first ring goroutine lands.
 */

#define LEO_LIB
#include "../leo.c"

/* ---- construction / destruction ---------------------------------- */

void *leo_bridge_create(void) {
    Leo *leo = calloc(1, sizeof(Leo));
    if (!leo) return NULL;
    leo_init(leo);
    return leo;
}

void leo_bridge_destroy(void *ptr) {
    if (!ptr) return;
    Leo *leo = (Leo *)ptr;
    leo_free(leo);
    free(leo);
}

/* ---- reply path -------------------------------------------------- */

void leo_bridge_ingest(void *ptr, const char *text) {
    if (!ptr || !text) return;
    leo_ingest((Leo *)ptr, text);
}

int leo_bridge_respond(void *ptr, const char *prompt,
                       char *out, int max_len) {
    if (!ptr || !prompt || !out || max_len < 2) return 0;
    return leo_respond((Leo *)ptr, prompt, out, max_len);
}

/* ---- persistence ------------------------------------------------- */

int leo_bridge_save(void *ptr, const char *path) {
    if (!ptr || !path) return 0;
    return leo_save_state((const Leo *)ptr, path);
}

int leo_bridge_load(void *ptr, const char *path) {
    if (!ptr || !path) return 0;
    return leo_load_state((Leo *)ptr, path);
}

/* ---- accessors --------------------------------------------------- */

void leo_bridge_stats(void *ptr) {
    if (!ptr) return;
    leo_stats((const Leo *)ptr);
}

long leo_bridge_step(void *ptr) {
    if (!ptr) return 0;
    return ((const Leo *)ptr)->step;
}

int leo_bridge_vocab(void *ptr) {
    if (!ptr) return 0;
    return ((const Leo *)ptr)->bpe.vocab_size;
}

int leo_bridge_bigrams(void *ptr) {
    if (!ptr) return 0;
    return ((const Leo *)ptr)->bigrams.n_entries;
}

int leo_bridge_trigrams(void *ptr) {
    if (!ptr) return 0;
    return ((const Leo *)ptr)->trigrams.n_entries;
}

/* ---- overthinking (rings of thought) ----------------------------- */
/* generate: read-only. observe: write. pulse: read-only snapshot. */

int leo_bridge_generate_ring(void *ptr, const char *seed,
                             float temp, int max_tokens,
                             char *out, int max_len) {
    if (!ptr || !seed || !out || max_len < 2) return 0;
    return leo_generate_ring((Leo *)ptr, seed, temp, max_tokens,
                             out, max_len);
}

void leo_bridge_observe_thought(void *ptr, const char *text,
                                 const char *source) {
    if (!ptr || !text) return;
    leo_observe_thought((Leo *)ptr, text, source ? source : "");
}

void leo_bridge_pulse(void *ptr, float *entropy,
                      float *arousal, float *novelty) {
    if (!ptr) return;
    LeoPulse p;
    leo_pulse((const Leo *)ptr, &p);
    if (entropy) *entropy = p.entropy;
    if (arousal) *arousal = p.arousal;
    if (novelty) *novelty = p.novelty;
}
