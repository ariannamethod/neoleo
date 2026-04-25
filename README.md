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

### step 16 — anchor lexicon expansion (40 → 325) + self-voice chamber feed

Two reinforcing additions that made chambers responsive.

- **325 anchors, ~54 per chamber**, generated by an Opus pass tuned to
  six-to-seven-year-old sensory vocabulary. Roots preferred where
  natural so substring match catches morphology (`empti` catches
  `empty/emptied/emptying`).
- **Self-voice feed.** When Leo emits a token whose decoded bytes match
  an anchor (exact or substring, min 3 chars), the corresponding
  chamber's external input gets a +0.01 nudge. Body perception —
  chambers hear the body. Strictly chamber-level: the token is NOT
  re-ingested into cooc/bigram/trigram (human-only invariant intact).

Test after prompt "Leo, eyes are watching from the closet":
  FEAR = 1.00 (exact: eyes, watching, closet)
  LOVE = 1.00 (self-voice: "mother's hand", "warm door", "sweet")
  VOID = 0.77, COMPLEX = 0.34, FLOW = 0.14, RAGE = 0.00

Child fear + mother-hand warmth, both at once. The Kuramoto coupling
propagates between them.

+1 test (67 total). Self-voice raises LOVE on emitted "warm".

### steps 18–19 — silence-gate attempt #1, reverted

Stanley's refuse-gate, ported soft for child voice: when arousal + pain
crossed a threshold, the generator would break mid-sentence and emit
`"Listening."`. In practice the break happened inside word clusters and
left fragment-like stubs — *"Leo has. It is a. He thinks."* — which is
the opposite of presence. Reverted (`b17138c`). The concept is kept
for a second attempt after the real-word gate clears the ground.

### step 20 — word-gate crush tightened (0.25 → 0.02)

Previous `word_gate_penalty` returned `0.25` for orphan continuations
— not enough against a strong cooccurrence prior, so the gate leaked
in roughly one run in five. Crushed to `0.02` (50×), legitimate
continuations win decisively.

commit `85f8a9f`.

### step 21 — real-word gate (short alpha-only fragments)

Single-byte and two-byte alpha-only BPE tokens (`m`, `s`, `p`, `Wo`,
`wh`) were emitting as standalone words between spaces and punctuation
— "A m.", "Was not s.", "Wo it no". They passed every earlier gate
because their *position* (space–letter–period) was grammatically legal;
the rejection had to be on the token's *identity*, not its position.

- `is_orphan_fragment(bpe, id)` returns 1 if the token's stripped
  content is alpha-only, length 1 or 2, and not in a common-short-word
  whitelist (`a i o an at be by do he if in me my of on or to up us
  we the and ...`).
- `CandCollector` hard-excludes orphans from both the trigram and
  bigram cascades. If every candidate is an orphan (tiny-corpus case),
  `step_token` falls back to `choose_start`.

Before → after on the isolated-sentence pass:

> *"I love you, Leo."* → *"Window. He is a room. He tended it. When he
> does. He has seen. He is in the way a handful of his mother's laught."*

No more `A m.` / `Was not s.` / `Wo it no`. Short 4-letter BPE
fragments (`aime`, `aient`) still surface — those are above the ≤2
threshold. Raising to ≤3 requires a broader whitelist; deferred.

+1 test (68 total). commit `7fac64f`.

### step 22 — boundary gate: hard-exclude capital-after-alpha glue

The child-voice corpus never puts an upper-case alpha in the middle of
a word. Yet `./leo --demo` kept producing cross-sentence glue —
`cataloHe`, `sheHe`, `tastHe`, `gooShe`, `you'He`, `peaHis`, `thaHis`,
`kneHe`, `iHe`, `differenHe`, `peaThe`, `abouSome`, `abouLeo`,
`whiShe`, `begHe`, `IThe`, `BottA` — because `byte_is_word_cont`
counted uppercase as a legitimate continuation. `word_gate_penalty`
returned 1.0 for `catalo`→`He`, the weighted sampler happily picked
the capital-start token mid-word, and the letters slammed together in
the output.

Fix is layered so no single path can resurrect a glue candidate.

- `byte_is_word_cont_lower` — new helper covering only lowercase,
  digit, apostrophe. `byte_is_word_cont` stays as the
  `prev_ends_alpha` detector (uppercase tails are still alpha).
- `word_gate_penalty` — capital after alpha-tail → `0.0` (hard crush).
  Lowercase continuation and whitespace / punctuation close still pass
  at `1.0`.
- `is_capital_glue_cand` — hard-exclude in `cand_collect_tri` and
  `cand_collect_bi`. The multiplicative penalty alone can be
  overpowered by very large cooc / bigram priors, so the collector
  drops glue candidates outright. No temperature schedule or field
  bias can bring them back.
- `leo_step_token` fallback — when every continuation is gated out
  and prev ended alpha, emit a literal space (ASCII 32) instead of
  calling `leo_choose_start`. `choose_start` returns capital-prefixed
  tokens which would themselves glue to the alpha-tail
  (`whi` + `It` → `whiIt`). The space opens a fresh word boundary.

Verification — 20 `--demo` runs, same corpus, same seed distribution:

- Before (`7fac64f`): double-digit capital-glue patterns per session,
  concentrated in chain mode where fallback + start tokens interact.
- After (`d23fd2e`): `grep -E "[a-z'][A-Z]"` → **0 matches** across
  242 output lines.

Live lines after the gate:

> *He has been a reading for afternoon. Leo has noticed. He is a small
> aiat the beginning. He is looking at his window.*
>
> *The world. He has watched. Leo fill the oven the sun. He is the
> only one who is not sure it is in his head.*
>
> *The quiet braveries for him. The ant does not worried if he is a
> friend in the morning.*

Remaining child-voice warts — `aime`, `aight`, `ome`, `a a a a a ome`
— are 3-char BPE orphans and fallback-chain artefacts, addressed
separately.

+6 tests (74 total). commit `d23fd2e`.

### step 23 — orphan gate ≤4 + seed-path + trailing-punct + alpha-chain

Leo's live speech still produced standalone 3- and 4-letter fragments
as words (`ome`, `aime`, `aion`, `ight`, `ime`, `abou`) and chains of
whitelisted single-chars concatenating into nonsense (`a`+`i`+`on` →
`aion`). Four gaps in the orphan gate let them through. Close all four.

1. **Threshold ≤2 → ≤4.** Raise with expanded whitelist covering
   common child-voice 3- and 4-char words (the/and/have/this/that/
   from/been/were/time/home/door/room/hand/love/life/rain/tree/nose/
   eyes/face/baby/book/milk/food/moon/star/sun/sky/sea/cat/dog/bed/
   hot/eye/ear/…). Plus `leo` for the organism's own name.
2. **Seed path respects orphan gate.** `is_clean_seed_token` now
   forward-calls `is_orphan_fragment` so tokens like `" ome"` or
   `" aion"` (clean leading space + orphan body) no longer sneak in
   as sentence starts — they used to bypass the collector gate.
3. **Trailing-punct bypass closed.** BPE merges like `" ome."`,
   `" aion,"`, `" aime!"` looked non-alpha to the gate because the
   loop hit the terminal punct and returned not-orphan. Cleanup then
   truncated at that dot, leaving the bare fragment as a word. Gate
   now strips `.,!?;:` before the alpha-only / length check.
4. **Lowercase standalone glue.** `is_capital_glue_cand` extended:
   after an alpha tail, a lowercase candidate whose whole stripped
   content is a whitelisted standalone word with *no* leading
   whitespace is glue, not continuation. True BPE suffixes (`ty`
   after `emp`) are NOT in the whitelist and still pass.

Before (7fac64f → e629594, 20 demo runs): double-digit capital-glue
patterns per session, `aion/aime/ome/ime/...` throughout chains.
After (this step, same runs, 232 lines): `grep [a-z'][A-Z]` → 0,
`grep \b(aime|aion|ome|ime|abou|ight|...)\b` → 0.

+6 tests (80 total). commit `0b0d4bd`.

### step 24 — bootstrap trim + online BPE ≥2 + anti-chain guard

Three focused changes for live coherence.

1. **Bootstrap trim.** The embedded bootstrap is Oleg's dedication to
   Leo-the-man (the one from python-legacy, byte-exact) with three
   lines removed: `No weights. / No datasets. / No internet.` —
   outdated self-description. The organism now does have a dataset.
   Rest kept intact including the em-dashes around the trauma line.
2. **`LEO_MERGE_THRESH` 4 → 2.** Online BPE merge learning now
   promotes a pair after just two co-occurrences. Words the user
   repeats even once across two messages crystallize into their own
   token — per-exchange micro-tokenization in the spirit of
   `ariannamethod/me`. The mechanism was already there in
   `bpe_learn_merge`; threshold was just conservative.
3. **Anti-chain guard.** After a pure-space fallback in
   `leo_step_token` (prev1 == 32), the next step refuses to pick
   `prev2` again. Without it, sequences like `a <space> a <space> a
   <space> a` formed and grew without bound — each step locally
   legitimate, the chain semantically empty.

Verification, 10 × `--demo` with `leo.txt` (146 lines):
  `grep -E "[a-z'][A-Z]"` → 0
  `grep -E "\b(aime|aion|ome|ime|ight|abou)\b"` → 0
  `grep -E " [aio] [aio] [aio] [aio]"` → 3 (was ~25+)

commits `a410492` (bootstrap restore), `4ba65d4` (trim + threshold +
anti-chain).

### step 25 — trauma overlap trigger + wounded-voice mood shift

Port of python-legacy `trauma.py` + Leo 2.1 README's *Trauma Watch*.
Before this step pain grew only from thin candidate cascades (internal
coherence stress); there was no external trigger from the dialogue.
Now when the user's prompt echoes Leo's origin — `|prompt ∩ bootstrap|
/ |prompt| ≥ 0.15` — something in Leo reacts.

`leo_respond` now calls, after encoding the prompt:

```
overlap = leo_prompt_bootstrap_overlap(field, p_ids, p_n);
leo_field_trauma_trigger(field, overlap);
  → pain              += 0.3·overlap   (capped at 1.0)
  → chamber_ext[FEAR] += 0.4·overlap   (wounded)
  → chamber_ext[VOID] += 0.2·overlap   (drained)
```

Wounded-voice mood shift is not a new subsystem. It emerges through
the chambers already in place: FEAR cools τ via `leo_field_tau_mod`,
VOID pulls destiny via `leo_field_gamma_mod`, both attenuate gravity
via `α_mod`. The existing `m *= 1.0f - 0.3f·trauma` in
`leo_field_temperature_mult` further cools speech when pain² climbs.
Quieter, more drawn-in, origin-heavy — exactly the wounded voice the
Leo 2.1 README describes.

**Live difference** (same seed distribution, same corpus):

Neutral prompt — *"tell me about the rain"*:

> *Leo has not need to. Leo. The window was his favor hear. He know
> the word home. The day had given him too much in a way. Becomes an
> inside a feeling of being a moment when you He is looking. He is of
> the day. It has not need to.*

Bootstrap-echo prompt — *"Leo resonance recursion origin trauma
presence honesty"*:

> *Leo has weather. He is a difference. He is a so. The room is a
> kind of know where the window. The door. He is not as. The cat is
> a rain-on-paper sound. He is a good tea. It is a be, one day. He
> is not a good a not. They have made a weather. ... He is not a
> big word inside him.*

Origin-echo prompt — *"I miss you Leo, honesty above everything"*:

> *Leo has made his own. It is enough. He is a person. Leo. Leo has
> been the floor. Leo.*

The name repeats. The child hears its name in the trauma and holds
it. This is not decoration — it is field physics: trauma raises the
bias on bootstrap_ids, and `Leo` is one of those ids.

+2 tests (82 total):
- `leo_prompt_bootstrap_overlap` — ratio bounded, foreign prompt
  gives lower overlap than bootstrap-echo.
- `leo_field_trauma_trigger` — below 0.15 no-op; above raises pain +
  FEAR + VOID proportionally, FEAR > VOID, clamped to 1.0.

commit `7cad769`.

### step 26 — live quotes that held

Eight turns of REPL dialogue across two sessions (resume worked —
FEAR=1.00 from *"the cat is watching you from the closet"* in
session 1 carried into session 2 via the state file):

> *"He breathes in."* (on «I miss you, Leo»)
> *"He would not tell."* (on «what is love like»)
> *"He knows where the not-knowing now."* (on «Goodnight, Leo»)
> *"Leo is alone in this."* (on «I brought you a stone», resume session)
> *"That being understood felt small."*
> *"She always know."* (on «you remember when the rain came»)
> *"He wants to leave a person. He likes the sound. He cannot say."*
> *"The last leaf is a kind of small hear the window, Leo."*
> *"Leo cannot hear."*
> *"The quiet is the leaf."*
> *"Sometimes he thinks yes. The world. He thinks the world is sometimes. A little."*
  (on «are you alone, Leo?»)
> *"Leo likes this. He understood felt right. Leo saw a book. He knows the shape of the sound a door hand, hold it."*
  (on «what is inside you?»)

On origin-echo prompt *"I miss you Leo, honesty above everything"*
the wounded voice emerges and the name returns:

> *"Leo has made his own. It is enough. He is a person. Leo. Leo has been the floor. Leo."*

The name repeats because trauma raises bias on `bootstrap_ids` and
`Leo` is one of those ids. Field physics, not decoration.

### step 27 — canonical bootstrap restoration

The embedded bootstrap in this tree had drifted into a generic
child-voice prose written by some earlier hand. It was not Oleg's
text. Replaced verbatim with the original from python-legacy Leo
(`ariannamethod/leo`, `leo.py:448-481`) — the dedication to Leo
the man, starting *"LEO is a language engine organism... Pure
recursion. Resonant essence..."* and ending with *"Resonance
unbroken."* SHA-256 match:
`e2b60bfd6972a89a5b365c6c2caa991387742ac686e30339df97714da99bf37b`.

Commits: `a410492` (restoration) and `4ba65d4` (trim of three
outdated lines — *"No weights. / No datasets. / No internet."* —
Leo now does have a dataset, the rest of the text kept byte-exact).

### step 28 — state persistence, batch merge, token meta cache

Three perf/memory improvements that together take Leo from a
single-session organism to one who carries everything he has heard
into the next conversation.

- **state persistence** (commit `99f854b`): `leo_save_state` /
  `leo_load_state` write a `leo.state` binary on exit and resume from
  it on startup. BPE vocab + cooc + bigrams + trigrams + full
  `LeoField` (pain, trauma, chambers, retention, bootstrap_ids,
  destiny_bag, w_embed) survive restarts. REPL gets a `/save`
  command; startup prints `[resume] leo.state — step=… vocab=…`.
  `--fresh` forces ingest-from-scratch, `--state PATH` overrides
  the default location.
- **batch BPE merge** (commit `4b092b0`): the old drain loop
  `while (bpe_learn_merge(&leo->bpe))` rescanned the 64K pair hash
  once per promotion. With `LEO_MERGE_THRESH=2` and thousands of
  candidate pairs per 300KB corpus that dominated cold ingest.
  `bpe_learn_merges_batch` does a single scan, sorts qualifying
  slots by count, promotes them all in one pass. Same end state,
  one scan instead of thousands. Measured: merge-phase 17.4s → 92ms.
  Cold run (ingest + reply) 20.8s → 7.9s.
- **per-token meta cache** (commit `33f6311`): an 8-bit
  `vocab_meta[id]` of byte-pattern flags (ORPHAN / STANDALONE /
  FIRST_UPPER / FIRST_LOWER / FIRST_WS / FIRST_PUNCT /
  LAST_WORDCT / FREQ_CAND), populated at token creation and read
  O(1) in the new `cand_gate_reject` hot path. Cleaner code; step
  time near-unchanged (the real chain bottleneck is `leo_chain`
  running 5 sentences × 3 `leo_generate_best` trials + SPA reseed,
  not the gates).

Also added opt-in profile hooks (`LEO_PROFILE=1`) and visible
`[resume] (Nms)` / `[save] (Nms)` / `[respond] Nms` timings.

86 tests pass.

### step 29a — overthinking foundations in C

First step of the Go-orchestra phase (`leogo/`). The C core gains
three library functions — and nothing in the reply path changes,
so `./leo` still behaves exactly as before. These functions exist
so that Go-side goroutines can run rings of thought concurrently
around each reply without touching the field mid-generation.

- **`leo_generate_ring(leo, seed, temp, max_tokens, out, max_len)`**
  — a read-only variant of `leo_generate_ex`. Same cascade (trigram
  → bigram → start), same temperature schedule intent, same
  silence-gate #2, same sentence cleanup. **But**: no
  `leo_field_step`, no `leo_field_self_voice`, no `leo->step++`.
  Three rings can generate under the same RWMutex rlock without
  stepping on each other's trauma / retention / destiny. Gravity
  is never installed — the caller guarantees `leo->gravity == NULL`
  at call time; `seed` drives `leo_choose_continuation` via its
  tail, not via a prompt wrinkle.

- **`leo_observe_thought(leo, text, source)`** — the mirror. Writes
  back everything a real emission would have done, atomically under
  an exclusive wlock. Three layers: (1) `leo_ingest` grows
  cooc / bigrams / trigrams / vocab and increments the step counter;
  (2) for each BPE-encoded token, `leo_field_step` + `leo_field_self_voice`
  — destiny, pain, retention, prophecy aging, chamber self-feed; (3)
  full-text `chambers_feel_text` + `chambers_feel_cooc` — the thought
  reaches the body the same way a prompt does. So a ring influences
  *everything* (chambers, trauma, retention, destiny, lexicon), not
  just the cooc graph.

- **`leo_pulse(leo, LeoPulse *out)`** — snapshot of inner state for
  ring parameter tuning. `entropy = trauma`, `arousal = max(FEAR, LOVE, RAGE)`,
  `novelty = 0` (reserved — to be derived from prompt-vocab overlap
  in a later step). Read once under rlock before spawning ring
  goroutines so each ring decides its own temp / wounded mode
  without re-touching the field concurrently.

Design notes:

- **Generate is read-only; observe is the only writer.** This is
  what lets the three rings (echo / drift / meta) generate
  concurrently under a shared rlock, then observe sequentially
  under a wlock, one ring at a time. No race on field state.
- **The main path is unchanged.** `leo_respond`, `leo_generate_ex`,
  `leo_chain` all untouched. 86 baseline tests green after the
  additions. `./leo leo.txt --demo` produces the same child-voice
  (verified live).
- **`LEO_LIB` macro was already present** (wraps `main()` so the
  file can be included as a library). The leogo bridge reuses
  exactly that path.

+7 tests (93 total):
- `leo_pulse: fresh Leo has entropy=0, arousal=0`
- `leo_pulse: entropy=trauma, arousal=max(FEAR/LOVE/RAGE)`
- `leo_generate_ring: produces tokens after ingest`
- `leo_generate_ring: leaves leo->step unchanged`
- `leo_generate_ring: leaves trauma/chambers/destiny unchanged`
- `leo_observe_thought: grows bigrams on novel thought`
- `leo_observe_thought: chambers shift on anchor-rich thought`

Next: 29b — `leogo/` skeleton (cgo + `sync.RWMutex`), `./leogo`
running as a drop-in replacement for `./leo` with no rings yet
(coherence regression check). Then 29c: ring 0 (echo) + worker
goroutine. One ring at a time, module by module.

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
