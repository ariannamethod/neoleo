// leo.go — Go wrapper around the C core of neoleo.
//
// Owns an opaque C pointer plus a sync.RWMutex. The mutex is the
// discipline that lets future ring goroutines share the field:
//
//   - reply path (Respond / Ingest / Save / Load / Observe) takes wlock
//   - read-only access (Stats / Step / Vocab / GenerateRing) takes rlock
//
// In step 29b only the reply path is wired; ring accessors arrive
// in 29c.
package main

/*
#cgo CFLAGS: -O2 -Wall -I..
#cgo LDFLAGS: -lm

#include <stdlib.h>

extern void *leo_bridge_create(void);
extern void  leo_bridge_destroy(void *);

extern void  leo_bridge_ingest(void *, const char *);
extern int   leo_bridge_respond(void *, const char *, char *, int);

extern int   leo_bridge_save(void *, const char *);
extern int   leo_bridge_load(void *, const char *);

extern void  leo_bridge_stats(void *);
extern long  leo_bridge_step(void *);
extern int   leo_bridge_vocab(void *);
extern int   leo_bridge_bigrams(void *);
extern int   leo_bridge_trigrams(void *);

extern int   leo_bridge_generate_ring(void *, const char *, float, int, char *, int);
extern void  leo_bridge_observe_thought(void *, const char *, const char *);
extern void  leo_bridge_pulse(void *, float *, float *, float *);
extern int   leo_bridge_bootstrap_fragment(void *, char *, int);

extern void  leo_bridge_soma_snapshot(void *, int);
extern void  leo_bridge_soma_dump(void *);
extern void  leo_bridge_soma_velocity(void *, float *);
extern int   leo_bridge_soma_n(void *);

extern float leo_bridge_mathbrain_step(void *, float);
extern float leo_bridge_mathbrain_tau_nudge(void *);
extern void  leo_bridge_mathbrain_dump(void *);
extern int   leo_bridge_mathbrain_train_count(void *);

extern int   leo_bridge_islands_assign(void *);
extern int   leo_bridge_islands_n(void *);
extern int   leo_bridge_islands_current(void *);
extern void  leo_bridge_islands_dump(void *);

extern int   leo_bridge_bridges_record(void *);
extern int   leo_bridge_bridges_n(void *);
extern void  leo_bridge_bridges_dump(void *);
*/
import "C"

import (
	"sync"
	"unsafe"
)

type LeoGo struct {
	ptr unsafe.Pointer
	mu  sync.RWMutex
}

func NewLeoGo() *LeoGo {
	p := C.leo_bridge_create()
	if p == nil {
		return nil
	}
	return &LeoGo{ptr: p}
}

func (lg *LeoGo) Free() {
	if lg == nil || lg.ptr == nil {
		return
	}
	C.leo_bridge_destroy(lg.ptr)
	lg.ptr = nil
}

func (lg *LeoGo) Ingest(text string) {
	if text == "" {
		return
	}
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	lg.mu.Lock()
	defer lg.mu.Unlock()
	C.leo_bridge_ingest(lg.ptr, ctext)
}

func (lg *LeoGo) Respond(prompt string) string {
	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))
	out := make([]byte, 4096)
	lg.mu.Lock()
	n := C.leo_bridge_respond(
		lg.ptr, cprompt,
		(*C.char)(unsafe.Pointer(&out[0])), C.int(len(out)),
	)
	lg.mu.Unlock()
	_ = n
	for i, b := range out {
		if b == 0 {
			return string(out[:i])
		}
	}
	return string(out)
}

func (lg *LeoGo) Save(path string) bool {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	lg.mu.Lock()
	defer lg.mu.Unlock()
	return C.leo_bridge_save(lg.ptr, cpath) != 0
}

func (lg *LeoGo) Load(path string) bool {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	lg.mu.Lock()
	defer lg.mu.Unlock()
	return C.leo_bridge_load(lg.ptr, cpath) != 0
}

func (lg *LeoGo) Stats() {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	C.leo_bridge_stats(lg.ptr)
}

func (lg *LeoGo) Step() int64 {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int64(C.leo_bridge_step(lg.ptr))
}

func (lg *LeoGo) Vocab() int {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int(C.leo_bridge_vocab(lg.ptr))
}

func (lg *LeoGo) Bigrams() int {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int(C.leo_bridge_bigrams(lg.ptr))
}

func (lg *LeoGo) Trigrams() int {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int(C.leo_bridge_trigrams(lg.ptr))
}

// ---- overthinking (rings of thought) -----------------------------
//
// GenerateRing is read-only — holds rlock during the C call. Multiple
// ring goroutines can call this concurrently without racing on the
// Leo field because leo_generate_ring never writes.
//
// ObserveThought is the only writer — wlock. Every ring is observed
// sequentially under an exclusive lock, one at a time.
//
// Pulse is a fast read-only snapshot used by the ring worker to
// decide each ring's temp / wounded mode before spawning generate.

type Pulse struct {
	Entropy float32 // trauma (pain²)
	Arousal float32 // max(FEAR, LOVE, RAGE)
	Novelty float32 // reserved — populated in a later step
}

func (lg *LeoGo) GenerateRing(seed string, temp float32, maxTokens int) string {
	cseed := C.CString(seed)
	defer C.free(unsafe.Pointer(cseed))
	out := make([]byte, 2048)
	lg.mu.RLock()
	n := C.leo_bridge_generate_ring(
		lg.ptr, cseed,
		C.float(temp), C.int(maxTokens),
		(*C.char)(unsafe.Pointer(&out[0])), C.int(len(out)),
	)
	lg.mu.RUnlock()
	_ = n
	for i, b := range out {
		if b == 0 {
			return string(out[:i])
		}
	}
	return string(out)
}

func (lg *LeoGo) ObserveThought(text, source string) {
	if text == "" {
		return
	}
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	csource := C.CString(source)
	defer C.free(unsafe.Pointer(csource))
	lg.mu.Lock()
	defer lg.mu.Unlock()
	C.leo_bridge_observe_thought(lg.ptr, ctext, csource)
}

func (lg *LeoGo) Pulse() Pulse {
	var e, a, n C.float
	lg.mu.RLock()
	C.leo_bridge_pulse(lg.ptr, &e, &a, &n)
	lg.mu.RUnlock()
	return Pulse{
		Entropy: float32(e),
		Arousal: float32(a),
		Novelty: float32(n),
	}
}

// ---- soma (Klaus-style numeric memory) ----------------------------
//
// Snapshot writes a slot — wlock. Dump and velocity read — rlock.

// SomaSnapshot stores the current inner-state into the soma ring
// buffer. Called once per reply-cycle by the worker, after all
// rings have observed.
//
// source: 0 = post-cycle (the only value used for now).
//         Other values are reserved for future fine-grained snapshots.
func (lg *LeoGo) SomaSnapshot(source int) {
	lg.mu.Lock()
	defer lg.mu.Unlock()
	C.leo_bridge_soma_snapshot(lg.ptr, C.int(source))
}

func (lg *LeoGo) SomaDump() {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	C.leo_bridge_soma_dump(lg.ptr)
}

func (lg *LeoGo) SomaVelocity() [6]float32 {
	var v [6]C.float
	lg.mu.RLock()
	C.leo_bridge_soma_velocity(lg.ptr, &v[0])
	lg.mu.RUnlock()
	var out [6]float32
	for i := 0; i < 6; i++ {
		out[i] = float32(v[i])
	}
	return out
}

func (lg *LeoGo) SomaN() int {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int(C.leo_bridge_soma_n(lg.ptr))
}

// ---- mathbrain (body-perception advisor) -------------------------
//
// MathbrainStep is the convenience writer: extracts features from
// current Leo state, runs a forward pass, and trains one SGD step
// on the supplied target quality. Returns the predicted quality
// from the forward pass (post-train it is unchanged in m.last_y).
//
// MathbrainTauNudge reads the most recent advisor output —
// boredom / overwhelm / stuck logic distilled into a single
// temperature shift. The next ring cycle reads this and adjusts.
//
// Both calls are short C ops; rlock for read, wlock for step.

func (lg *LeoGo) MathbrainStep(targetQuality float32) float32 {
	lg.mu.Lock()
	defer lg.mu.Unlock()
	return float32(C.leo_bridge_mathbrain_step(lg.ptr, C.float(targetQuality)))
}

func (lg *LeoGo) MathbrainTauNudge() float32 {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return float32(C.leo_bridge_mathbrain_tau_nudge(lg.ptr))
}

func (lg *LeoGo) MathbrainDump() {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	C.leo_bridge_mathbrain_dump(lg.ptr)
}

func (lg *LeoGo) MathbrainTrainCount() int {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int(C.leo_bridge_mathbrain_train_count(lg.ptr))
}

// ---- phase4 islands ----------------------------------------------
//
// Islands cluster the soma stream — "where in inner-state space am
// I right now?" with a stable answer. Filled by the worker calling
// IslandsAssign after each soma snapshot. Phase4 bridges (next step)
// will track A→B transitions on top of these.

func (lg *LeoGo) IslandsAssign() int {
	lg.mu.Lock()
	defer lg.mu.Unlock()
	return int(C.leo_bridge_islands_assign(lg.ptr))
}

func (lg *LeoGo) IslandsN() int {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int(C.leo_bridge_islands_n(lg.ptr))
}

func (lg *LeoGo) IslandsCurrent() int {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int(C.leo_bridge_islands_current(lg.ptr))
}

func (lg *LeoGo) IslandsDump() {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	C.leo_bridge_islands_dump(lg.ptr)
}

// ---- phase4 bridges ----------------------------------------------
//
// Bridges record island-to-island transitions. The worker calls
// BridgesRecord after IslandsAssign — it is a no-op when the island
// did not change. Pure recording: phase4 advisor (later step) will
// read the transition graph for "where the field tends to flow".

func (lg *LeoGo) BridgesRecord() int {
	lg.mu.Lock()
	defer lg.mu.Unlock()
	return int(C.leo_bridge_bridges_record(lg.ptr))
}

func (lg *LeoGo) BridgesN() int {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	return int(C.leo_bridge_bridges_n(lg.ptr))
}

func (lg *LeoGo) BridgesDump() {
	lg.mu.RLock()
	defer lg.mu.RUnlock()
	C.leo_bridge_bridges_dump(lg.ptr)
}

// BootstrapFragment returns a random sentence from the embedded
// bootstrap (Oleg's dedication to Leo). Read-only — used by ring 1
// in wounded mode to anchor the drift seed near origin when trauma
// crosses threshold.
func (lg *LeoGo) BootstrapFragment() string {
	out := make([]byte, 512)
	lg.mu.RLock()
	C.leo_bridge_bootstrap_fragment(
		lg.ptr,
		(*C.char)(unsafe.Pointer(&out[0])), C.int(len(out)),
	)
	lg.mu.RUnlock()
	for i, b := range out {
		if b == 0 {
			return string(out[:i])
		}
	}
	return string(out)
}
