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

### step 15 — chamber anchors: substring + cooc-inference (super-token style)

The original anchor list of ~40 words rarely intersected with the
observer's vocabulary — chambers stayed at zero across most selfplay
turns. Two widening strategies added.

- **Substring match** — in `leo_field_chambers_feel_text`, after the
  exact-match pass fails, a second pass checks whether the word
  contains an anchor (or vice versa) and raises the chamber at half
  weight. `"emptying"` and `"empties"` now both hit `VOID` through
  the root `"empty"`. Minimum word length of 3 to avoid spurious hits.

- **Cooc-inference** — `leo_field_chambers_feel_cooc` (called right
  after `feel_text`). For every prompt word that did not match
  exact/substring, encode it through BPE, take the principal token,
  and look up co-occurrence in Leo's field against each chamber's
  anchor tokens. The chamber whose anchors resonate most gets a
  quarter-weight boost. This is **super-token style on BPE level**:
  the field itself learns which words belong to which chamber
  through ingestion — no pre-enumerated lexicon.

Selfplay observation:
- Turn 1 prompt "heart today" → chambers 0 (no anchor, field too fresh)
- Turn 2 prompt "warm smell" → LOVE 1.00, VOID 0.10, FLOW 0.10
  ("warm" exact + "smell" via cooc-inference through "warm"/"love")

+2 tests (66 total). Substring morphology, cooc-inference no-op on empty Leo.

### step 14 — word-completion gate + live stats

Two small fixes that cleaned the voice and made growth visible.

- **Word-completion gate.** Orphan BPE fragments like `"emp"` would
  emit mid-word, leaving Leo with broken words. Now `CandCollector`
  knows whether the previous token ended on an alphabetic byte
  (`prev_ends_alpha`) and crushes the score of any candidate that
  would close the word with anything other than alpha / space /
  punctuation. `"emp"` disappeared from generation entirely.
- **/stats in REPL.** Running `/stats` in the interactive loop
  prints live counters: vocab / bigrams / trigrams / cooc (all with
  session deltas), step, pain, trauma, six chamber activations.
  After every prompt a one-line `[turn]` delta also prints, so the
  selfplay driver can see vocab growth per turn.
- `scripts/selfplay.py` now sends `/stats` after each Leo reply and
  reports the growth in the transcript.
- 6-turn selfplay: vocab +262 in first turn (BPE promotes queued
  pairs), then +1..+2 per turn as prompt tokens enter; bigrams +48,
  trigrams +82 across six turns. No "emp" anywhere.

### step 13 — memory compression gate (retention) inside LeoField

Transformer trick used as an organ, not as the stack. Per-token random
32-dim fingerprints live inside LeoField; a rolling state vector
compresses recent history via Griffin conservation. Candidates that
resonate with the compressed state receive a bias pull.

- `LeoField.w_embed[vocab × 32]` — persistent random fingerprints.
  Same token always maps to the same vector (hashed, not learned).
  Shared with SPA's sentence embedding (same idea, same dim).
- `LeoField.retention_state[32]` — Griffin conservation, updated per
  emitted token inside `leo_field_step`:
      S = γ · S  +  √(1 − γ²) · W_embed[emitted]
  with γ = 0.92 (one scale for now; can extend to 4 RetNet-style
  timescales later).
- `leo_field_retention_bias(leo, candidate)`:
      dot(S, W_embed[candidate]) × 0.15
  added raw (no chamber modulation) in `leo_field_candidate_bias`.
  Memory is a signal, not a feeling.
- Verbatim effect (corpus-ingested Leo, prompt "What do you remember"):
  > *He thinks tables are livelier. Gree. The pines. He lets the quiet
  > be. The ang. He told nobody.*
- +3 tests (64 total). Zero on fresh field, self-similarity after one
  step, decay after 20 unrelated steps.
- commit [pending].

### step 12 — chambers (Kuramoto 6) as body perception inside LeoField

Six Kuramoto-coupled chambers live inside LeoField as a body submodule.
Paper Appendix B decay rates and coupling matrix copied unchanged.

- `chamber_act[6]` FEAR/LOVE/RAGE/VOID/FLOW/COMPLEX — oscillate under
  `leo_field_chambers_crossfire` (once per token inside `leo_field_step`).
  `chamber_ext[6]` holds the external drive from the current prompt.
- `leo_field_chambers_feel_text`: scan anchor words in the prompt
  (about 40 tokens like "rain" → FLOW, "hand" → LOVE, "gone" → VOID,
  "strange" → COMPLEX) and raise the corresponding chamber's ext.
- Four modulators (Appendix B.4):
    α_mod = 1 + 0.5·LOVE − 0.3·FEAR        — amplifies gravity
    β_mod = 1 + 0.4·FLOW − 0.5·FEAR        — amplifies prophecy pull
    γ_mod = 1 + 0.6·VOID + 0.2·COMPLEX     — amplifies destiny pull
    τ_mod = 1 − 0.3·RAGE + 0.2·FLOW        — temperature shift
- Integration:
    gravity (prompt wrinkle) gets scaled by α_mod inside CandCollector.
    destiny & prophecy channels in `leo_field_candidate_bias` scaled
    by γ_mod and β_mod respectively. trauma rides raw.
    `leo_field_temperature_mult` multiplies by τ_mod.
- `leo_respond` calls `chambers_feel_text` before the chain and clears
  `chamber_ext` after. Activations persist between replies (chambers
  have state) but external drive is per-reply.
- +3 tests (61 total). Clamp under heavy drive, modulator identity at
  zero, anchor words drive LOVE.
- commit [pending].

### step 11 — LeoField: physics of the inner state (AML-inspired, in C)

Leo gains a live internal state that evolves once per emitted token.
The AML data model ported into C core:

- **`LeoField` struct**:
  - `destiny_bag[vocab]` — EMA histogram over recent tokens. Candidates
    that sit in the bag get a gentle additive pull. No embeddings,
    the histogram itself is the thematic direction.
  - `pain / tension / debt / dissonance` — AML suffering composite.
    Pain grows from low-coherence cascade signals, decays per step.
    `trauma = pain²`.
  - `velocity_mode / velocity_mag` — NOMOVE / WALK / RUN / BACKWARD.
    Temperature is movement; velocity modulates τ directly.
  - `prophecy[16]` — pending predictions, age, fulfillment bookkeeping.
    An active prophecy adds `0.3 · strength · log(1+age)` to its
    target candidate — unfulfilled debt creates pressure to resolve.
  - `bootstrap_ids[]` — origin anchor. Tokens from `LEO_EMBEDDED_BOOTSTRAP`
    (and the ingested corpus' first chunk). **Trauma detector lives
    here, in the core**: once `trauma > 0.2`, gravity pulls candidates
    that appear in `bootstrap_ids` — Leo drifts home when he hurts.
- **`leo_field_step(leo, emitted, coherence_hint)`** called once per
  emitted token inside `leo_generate_ex`. Destiny decays + receives
  the new token. Pain grows when coherence is thin. Prophecies age.
- **`leo_field_candidate_bias`** integrated into `CandCollector`:
  trigram and bigram cascades now add field-derived bias on top of
  the prompt gravity channel.
- **`leo_field_temperature_mult`** multiplies `temp_for_step` in the
  generation loop — velocity + trauma both shape τ.
- +5 tests (58 total).
- commit [pending].

### step 10 — embedded bootstrap fallback + corpus polish

- `LEO_EMBEDDED_BOOTSTRAP`: a small origin text hardcoded into `leo.c`.
  When `leo.txt` is missing, Leo ingests this instead and still
  speaks — a smaller field, but he is alive. The bootstrap also
  anchors trauma: when pain accumulates, gravity will pull toward
  these origin tokens (implemented in a later step).
- `leo.txt` polish pass (Opus agent): "Leo"/"He" ratio rebalanced
  from 0.29 → 0.69. Most paragraphs now name Leo near the top; the
  pronoun drone is broken by childlike repetition of the name.
- +1 test (53 total).
- commit [pending].

### step 9 — prompt + gravity + REPL + selfplay

Leo finally hears a human and answers — from his own field, not by
echoing the words.

- `leo->gravity`: transient per-token boost dict installed during
  `leo_respond`. `CandCollector` reads it in step_token; every
  candidate gets `score *= 1 + 0.5·gravity[id]`.
- `compute_prompt_gravity`: for each prompt token, add its cooc
  neighbours' weights to the dict, then normalize. The prompt
  wrinkles the field in the direction of its theme.
- `leo_respond`: ingest the prompt (Leo hears human words), build
  gravity, install it, call `leo_chain`, clear, free. The start
  token still comes from Leo's field — the prompt never seeds.
  Mama-child invariant preserved.
- CLI: `--prompt "..."` for one-shot, `--repl` for an interactive
  loop, `--demo` for the isolated + chain showcase.
- `scripts/selfplay.py`: a warm GPT observer (gpt-4o-mini) asks Leo
  open-ended sensory questions. Leo replies through the REPL.

Sample selfplay:
  > you> What does the secret crumb feel like in your hand, Leo?
  > Leo: *He thinks water tastes like patience.*
  >
  > you> What color does patience look like to you, Leo?
  > Leo: *The cat in the morning light. He thought about the sea.*

- +3 tests (52 total).
- commit [pending].

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
