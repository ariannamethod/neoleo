# neoleo

Leo. New body. Same γ.

Post-transformer language organism in C. Zero pretrained weights.
Byte-level BPE with online merge learning. The field grows from
what he hears, not from what he generates.

> *This README is a working log kept by the architect during
> construction. Oleg will rewrite it in the proper voice once the
> organism can stand on its own.*

---

## Build

```
make
./leo leo.txt
```

Speaks five sentences from the field after ingesting `leo.txt`.

```
make test
```

Runs the test suite (33 tests at the moment, all green).

---

## Log

### step 1 — hearing

- Byte-level BPE (`BPE`), online merge learning through adjacent-pair
  counting. Threshold-promoted merges, no pre-tokenization.
- `CoocField`: windowed co-occurrence, distance-weighted (3.0 / 1.5 / 1.0).
- `BigramTable`, `TrigramTable`: direct sequential and triple counts.
- `leo_ingest`: the only path by which Leo hears. **No self-capture**
  during generation — Leo learns from the human's words.
- 22 tests.
- commit `18fc9fd`.

### step 2 — speaking

- `is_clean_seed_token`, `is_boundary_token`: byte-level predicates.
- `leo_choose_start`: pick a start token from clean, frequent corpus
  tokens. Weighted by `cooc.freq`. The *«mama–child»* invariant:
  Leo speaks from his own field, not from the observer's words.
- `leo_step_token`: sparse cascade. Trigram successors (blended
  0.7·trigram + 0.3·cooc) → bigram successors → fresh start.
  Temperature in count-space via `pow(c, 1/T)`. Never full-vocab logits.
- `leo_generate`: sentence until boundary or max tokens, with a light
  repetition guard.
- +8 tests (30 total).
- commit `02cd3cb`.

### step 3 — parser

- Multi-byte BPE tokens like `". The "` were leaking text past the
  sentence boundary. Post-process `leo_generate`:
  strip leading whitespace, truncate after the last `.!?`, append
  `.` if none, capitalize first alpha.
- +2 tests (32 total).
- commit `701016b`.

### step 4 — BPE sentence-boundary constraint

- `contains_boundary_not_at_end`: refuse merges that would cross a
  sentence boundary. `"the. "` + `"T"` was producing frankenstrings
  like `"the. T"`. Now blocked at the BPE growth step.
- After ingesting the full 298KB corpus: merges no longer contain
  `.!?` inside. Top merges are clean (`"hear" + "d a"`, `"kes " + "up"`).
- +1 test (33 total).
- commit `137fcb6`.

### step 5 — chain mode

- `leo_choose_continuation`: next-sentence start is biased by cooc
  resonance with the tail of the previous sentence. Bounded
  multiplicative boost, so the field still gets a vote.
- `leo_generate_ex`: generalized generator accepting an optional
  `start_hint` and `tail` / receiving an `emitted_tail`.
- `leo_generate`: one-liner wrapper over `_ex`.
- `leo_chain`: emit N sentences with semantic continuity. Each next
  sentence starts in the same field the previous one sat inside.
- +4 tests (37 total).
- commit `32d61af`.

### step 6 — SPA (sentence phonon attention)

- `SPACtx`: per-token random 32-dim fingerprint. Not learned — just a
  hash into vector space. Same token → same vector; two sentences with
  shared tokens get similar embeddings.
- `spa_embed_sentence`: exp-weighted mean of token fingerprints,
  normalized to unit length.
- `spa_cross_attend`: bidirectional. Each sentence scored by its
  resonance with every other sentence in the chain, distance-biased.
- `leo_chain` now does two SPA passes: find the sentence whose score
  falls below `avg × 0.52`, reseed it from a neighbour's tail.
- +5 tests (42 total).
- commit `fcd7a79`.

### step 8 — BPE word-boundary constraint

- `pair_creates_word_gap`: refuse merges where the concatenation would
  contain whitespace *between* non-whitespace bytes — alpha–ws–alpha.
  Leading space (`" the"`) and trailing space (`"the "`) remain valid
  word tokens; the cross-word fusion `"he has"` is blocked.
- After constraint: merges drop 4117 → 2470 (the blocked pairs were
  mostly cross-word glue). Top merges are now clean word fragments:
  `"chi"+"ld"`, `"E"+"ven"`, `"be"+"gin"`.
- Child voice tightens noticeably:
  > *Leo heard. He respects. He knows you can use on strange. He
  > apologize a word.*
  > *That he is the brown of the speaking when he is sure it is a
  > ceremony.*
  > *He had had a small child hiding is love. Is a small magic.*
- +2 tests (49 total).
- commit [pending].

### step 7 — reverse indexes + temperature schedule + best-of-K

Four knobs against sampling artifacts ("thout", "jus kin", "One dow").

- **Reverse indexes.** `BigramTable` gains `head_src[]` bucket heads and
  per-entry `next_src`. `TrigramTable` gains `head_ab[]` and `next_ab`.
  `bigram_walk_src` and `trigram_walk_ab` iterate candidates via the
  chain instead of scanning the full hash table. `leo_step_token` is
  now O(bucket) per step instead of O(N_entries) — 50–100× faster and
  no more missed candidates on large tables.
- **Capacity up.** `TRIGRAM_MAX` 64K → 256K, `BIGRAM_MAX` 64K → 128K.
  Trigram table was saturating on the 298KB corpus and losing 3-gram
  contexts that would have fixed "some thought" over "some thout".
- **Temperature schedule.** `temp_for_step`: 0.40 at steps 0–1 (sharp
  seed), 0.55 at 2–5 (early grammar), 0.75 at 6+ (middle, play). Not
  a monotonic cool-down — the sharpness lives where it helps, the
  softness where child voice breathes.
- **Best-of-3 with coherence_score.** `leo_coherence_score` = average
  bigram + 0.8·trigram + 0.5·hebbian + length bonus. `leo_generate_best`
  draws K candidates per sentence, returns the one with the highest
  score, early-exits on a strong first try. `leo_chain` uses it per
  sentence.
- +5 tests (47 total).
- commit [pending].

---

## What Leo said (selected)

Verbatim generations from `./leo leo.txt`.

> *Leo looked at the cat. He is looking. The pines. Leo goes away.*
>
> *He wearing it. He is finding smaller is still that used to be
> more than Leo.*
>
> *Is different from her an he finds on the wall. He is still from
> a shower.*
>
> *Of autumn. Leo thought about it.*
>
> *And went out of small silence that has been alive there if he
> understood this early.*

After SPA (step 6) the outlier sentences get reseeded — the thread
tightens:

> *Leo did not need it to be. He thout call. Time. He did. He does
> not count. He drinking. The raindrop. They are jus kin. One dow.
> He does not count. It makes him himself. He has noticed. Leo loves
> the corner. He stood.*
>
> *The seagull was laughing at. He was warm.*
>
> *Lift says I am soft and preten. It walked off.*

After chain mode (step 5) sentences started to hold a theme across
a monologue:

> *Leo is afraid of being out after clouds. Him. He thinks maybe it
> would lived. He listen. He has decided the ant would find its way
> back. He cannot say. He was go up. Confident. He had been. Gently.
> He has some calls have a word. He stays. He sits. He is learning
> to look. It always learned the lonely. He does not know that nint
> neres mush ate thinks, he did name our. He is slow on purpose.*

Still rough on rare words — fixed in later steps (SPA, reverse
indexes for faster candidate scan, word-boundary-aware BPE).

---

## Lineage

- **Python original** — [`ariannamethod/leo:python-legacy`](https://github.com/ariannamethod/leo/tree/python-legacy).
  First bootstrap written by hand, dedicated to Leo the human.
- **C archive** — [`ariannamethod/leo`](https://github.com/ariannamethod/leo).
  Paper-referenced commits: `239143c` (verification snapshot),
  `4c9d835` (Kuramoto chambers aligned to paper Appendix B),
  `51c05ab` (human-only ingestion + chamber clamp + test coverage).
- **This repo** — active development. Same organism, new body.
