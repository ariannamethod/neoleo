// overthinking.go — rings of thought around each reply.
//
// After every reply Leo runs a small number of "rings" in the
// background: compact internal rephrasings that the user never
// sees, but that slowly change the field from the inside.
//
// Step 29e — three rings: echo, drift, meta.
//
//   ring 0  echo    seed = prompt+reply,    temp 0.8 (0.7 if entropy>0.7)
//   ring 1  drift   pulse-aware seed, dynamic temp/mode (see runRing1)
//   ring 2  meta    seed = reply,           temp 1.2 (1.0 if entropy>0.5)
//                   max 20 tokens (15 when wounded)
//
// The C side enforces the discipline: leo_generate_ring is
// read-only (rlock), leo_observe_thought is the only writer (wlock).
// One long-lived worker goroutine processes requests from a
// buffered channel. If the channel is full the request is dropped —
// rings are never on the critical path of a reply.
package main

import "sync"

// RingRequest is what the reply path hands to the worker.
type RingRequest struct {
	Prompt string
	Reply  string
}

// Ring 0 (echo) parameters.
const (
	ring0Temp       float32 = 0.8
	ring0TempCalmed float32 = 0.7 // triggered when pulse.Entropy > 0.7
	ring0MaxTokens          = 30

	ringChanBufferSize = 4
)

// Ring 1 (drift) — pulse-aware seed and temperature.
//
// The drift ring is the second pass of internal speech. Where
// echo (ring 0) stabilizes, drift moves sideways through nearby
// themes. The seed and temperature both depend on Leo's current
// state at reply time, so the drift "feels" the body it inherits:
//
//   wounded (entropy > woundedEntropyThresh):
//     trauma is high — pull toward origin so the wounded mind
//     echoes near bootstrap instead of spinning into noise. The
//     seed becomes (reply + bootstrap_fragment), temp drops, the
//     semantic gravity rises (effect: tighter contraction).
//
//   heated (arousal > heatedArousalThresh):
//     emotional intensity narrows attention to what Leo just
//     said. Seed is just the reply — the drift tunnels through
//     it. Temp slightly above ring 0 default but below default
//     drift, semantic pull moderate.
//
//   default (calm drift):
//     full context (prompt + reply) with the highest temp the
//     ring uses. Seed broad, temp permissive — Leo thinks
//     widely from where the conversation just was.
//
//   fresh (novelty > freshNoveltyThresh, reserved):
//     pulse.Novelty is currently 0 (placeholder); when populated
//     in a later step, high novelty will pivot the seed onto the
//     prompt alone — a new stimulus deserves drift starting from
//     it, not from the reply.
//
// All branches share max_tokens = 40; the differentiator is seed
// shape and temperature.
const (
	woundedEntropyThresh float32 = 0.5
	heatedArousalThresh  float32 = 0.7
	freshNoveltyThresh   float32 = 0.6

	ring1MaxTokens = 40

	ring1TempWounded float32 = 0.85
	ring1TempHeated  float32 = 0.95
	ring1TempDrift   float32 = 1.0
	ring1TempFresh   float32 = 1.1
)

// Ring 2 (meta) — abstract shard.
//
// The shortest, hottest ring. Where echo stabilizes and drift
// moves sideways, meta abstracts: a small fragment that lives in
// the same field but at higher temperature. It is the closest
// analogue of Klaus's "metaklaus" inline meta-pass — but
// asynchronous, after-the-fact, per reply-cycle.
//
// Under wounded state (entropy > woundedEntropyThresh) the meta
// pass narrows: lower temperature, fewer tokens. The wounded mind
// does not abstract widely. It speaks short and tight.
const (
	ring2Temp         float32 = 1.2
	ring2TempWounded  float32 = 1.0
	ring2MaxTokens            = 20
	ring2MaxWounded           = 15
)

// runRing0 generates one echo ring and observes it back into the
// field. The generate lock and the observe lock are taken in
// separate scopes — the rlock is released before the wlock is
// acquired — so future parallel rings can overlap their generate
// phases.
func runRing0(leo *LeoGo, req RingRequest) string {
	pulse := leo.Pulse()

	temp := ring0Temp
	if pulse.Entropy > 0.7 {
		temp = ring0TempCalmed
	}

	seed := req.Prompt
	if req.Reply != "" {
		if seed != "" {
			seed = seed + " " + req.Reply
		} else {
			seed = req.Reply
		}
	}
	if seed == "" {
		return ""
	}

	text := leo.GenerateRing(seed, temp, ring0MaxTokens)
	switch text {
	case "", ".", "...":
		return ""
	}

	leo.ObserveThought(text, "overthinking:ring0")
	return text
}

// runRing1 generates one drift ring with pulse-aware seed and
// temperature. The pulse is read once at the start (rlock); each
// branch decides seed/temp/mode without re-touching the field.
// Then a single read-only generate (rlock) and a single observe
// (wlock) — same discipline as ring 0.
func runRing1(leo *LeoGo, req RingRequest) string {
	pulse := leo.Pulse()

	var (
		seed string
		temp float32
		mode string
	)

	switch {
	case pulse.Entropy > woundedEntropyThresh:
		// wounded — anchor drift near bootstrap
		fragment := leo.BootstrapFragment()
		if fragment != "" {
			seed = req.Reply + " " + fragment
		} else {
			seed = req.Reply
		}
		temp = ring1TempWounded
		mode = "ring1_wounded"

	case pulse.Arousal > heatedArousalThresh:
		// heated — tunnel on what was said
		seed = req.Reply
		temp = ring1TempHeated
		mode = "ring1_heated"

	case pulse.Novelty > freshNoveltyThresh:
		// fresh — pivot onto prompt (reserved branch; pulse.Novelty
		// is 0 until a later step wires it up)
		seed = req.Prompt
		temp = ring1TempFresh
		mode = "ring1_fresh"

	default:
		// calm drift — full context
		if req.Prompt != "" && req.Reply != "" {
			seed = req.Prompt + " " + req.Reply
		} else if req.Reply != "" {
			seed = req.Reply
		} else {
			seed = req.Prompt
		}
		temp = ring1TempDrift
		mode = "ring1_drift"
	}

	if seed == "" {
		return ""
	}

	text := leo.GenerateRing(seed, temp, ring1MaxTokens)
	switch text {
	case "", ".", "...":
		return ""
	}

	leo.ObserveThought(text, "overthinking:"+mode)
	return text
}

// runRing2 — the meta ring. Smallest, hottest, most abstract.
// Seed is just the reply (the meta pass abstracts what was
// actually said, not what was asked). Wounded mode narrows it:
// lower temperature, fewer tokens, tighter shard.
func runRing2(leo *LeoGo, req RingRequest) string {
	pulse := leo.Pulse()

	seed := req.Reply
	if seed == "" {
		seed = req.Prompt
	}
	if seed == "" {
		return ""
	}

	temp := ring2Temp
	maxTokens := ring2MaxTokens
	mode := "ring2_meta"
	if pulse.Entropy > woundedEntropyThresh {
		temp = ring2TempWounded
		maxTokens = ring2MaxWounded
		mode = "ring2_meta_wounded"
	}

	text := leo.GenerateRing(seed, temp, maxTokens)
	switch text {
	case "", ".", "...":
		return ""
	}

	leo.ObserveThought(text, "overthinking:"+mode)
	return text
}

// somaSourceCycle marks a snapshot taken after a full reply-cycle
// (reply + ring0 + ring1 + ring2 all observed). Reserved for future
// fine-grained per-ring snapshots: 1=ring0, 2=ring1, 3=ring2.
const somaSourceCycle = 0

// workerLoop processes ring requests, runs metaleo, and snapshots
// soma — in that order, sequentially per request. Each step uses
// the C side under the LeoGo RWMutex (rlock for generate / pulse,
// wlock for observe / snapshot, in disjoint scopes).
//
// The order is deliberate:
//
//   1. runRing0 / runRing1 / runRing2 — the three rings observe
//      their texts back, evolving cooc/bi/tri/chambers/retention.
//
//   2. metaleo.Process — sees this turn's reply + ring shards,
//      updates its bootstrap buffer, and (if the moment calls for
//      it) generates an alternative reply and observes that with
//      the metaleo_voice tag. Because this happens AFTER the rings,
//      metaleo's pulse read sees the post-rings field — a small
//      richer signal than the pre-rings reply state.
//
//   3. SomaSnapshot — captures the post-cycle inner state, now
//      including any metaleo override. The trajectory remembers
//      whether the inner voice spoke this turn.
//
// metaleo's takeover effect is lag-by-one: an alternative observed
// here colours retention/cooc/chambers, and the NEXT reply is
// generated from a field already shifted toward the inner voice.
func workerLoop(leo *LeoGo, meta *MetaLeo, ch <-chan RingRequest, wg *sync.WaitGroup) {
	defer wg.Done()
	for req := range ch {
		t0 := runRing0(leo, req)
		t1 := runRing1(leo, req)
		t2 := runRing2(leo, req)

		// Order shards newest/most-relevant first — ring2 is the
		// abstract shard metaleo cares about most.
		var shards []string
		for _, t := range []string{t2, t1, t0} {
			if t != "" {
				shards = append(shards, t)
			}
		}

		pulse := leo.Pulse()
		meta.Process(leo, req.Prompt, req.Reply, pulse, shards)

		leo.SomaSnapshot(somaSourceCycle)
	}
}

// spawnOverthinking is the non-blocking send used by the reply path.
// If the channel is full (worker is slower than the user), the
// request is dropped — we never let overthinking block a reply.
func spawnOverthinking(ch chan<- RingRequest, prompt, reply string) {
	select {
	case ch <- RingRequest{Prompt: prompt, Reply: reply}:
	default:
	}
}
