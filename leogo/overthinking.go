// overthinking.go — rings of thought around each reply.
//
// After every reply Leo runs a small number of "rings" in the
// background: compact internal rephrasings that the user never
// sees, but that slowly change the field from the inside.
//
// Step 29c — ring 0 (echo) only.
//
//   ring 0  echo    seed = prompt+reply,    temp 0.8 (0.7 if entropy>0.7)
//   ring 1  drift   → 29d (wounded-aware, seed drifts toward bootstrap)
//   ring 2  meta    → 29e (higher temp, shorter)
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

// runRing0 generates one echo ring and observes it back into the
// field. The generate lock and the observe lock are taken in
// separate scopes — the rlock is released before the wlock is
// acquired — so future parallel rings can overlap their generate
// phases.
func runRing0(leo *LeoGo, req RingRequest) {
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
		return
	}

	text := leo.GenerateRing(seed, temp, ring0MaxTokens)
	switch text {
	case "", ".", "...":
		return
	}

	leo.ObserveThought(text, "overthinking:ring0")
}

// workerLoop is the long-lived goroutine that processes ring
// requests. Receives from ch until it is closed; each request
// already buffered is processed before exit.
func workerLoop(leo *LeoGo, ch <-chan RingRequest, wg *sync.WaitGroup) {
	defer wg.Done()
	for req := range ch {
		runRing0(leo, req)
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
