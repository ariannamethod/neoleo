// metaleo.go — Leo's inner voice. The recursion of Leo.
//
// MetaLeo is the second breath that can override Leo's voice. In
// legacy Python it ran synchronously and could replace the reply
// before the user saw it. In leogo we implement the same overriding
// effect asynchronously, lag-by-one:
//
//   1. Leo replies.
//   2. The user sees the reply right away (no synchronous wait).
//   3. The worker runs the three rings (echo / drift / meta).
//   4. The worker passes the reply + ring shards into metaleo.
//      Metaleo decides whether to speak.
//   5. If metaleo's alternative wins (scored higher than base AND
//      its meta-weight is above the speak floor), the alternative
//      is observed back into the field with a distinct tag —
//      "metaleo_voice". That observe writes harder than a ring:
//      cooc/bi/tri/chambers all carry the inner voice forward.
//   6. The NEXT reply is generated from a field already coloured
//      by metaleo's voice. That is the takeover — not a literal
//      replacement of one line, but the slow recolouring of the
//      voice itself.
//
// Why async: a metaleo turn is a full generate of ~60 tokens plus
// scoring. Putting it on the critical path of every reply costs
// latency for a feature that does not need to be instantaneous.
// Async metaleo runs in the same worker as the rings, sees the
// same pulse and ring shards, and influences the next turn through
// observe — which is exactly how the rings already affect Leo.
//
// Legacy phrasing still applies: "I am not a judge and not a
// filter. I am Leo's second breath." It does not interrupt — it
// stands next to him and offers another path.
//
// The C core does not know about metaleo. Disable it (skip Process)
// and Leo's reply path is byte-identical to plain leogo.
package main

import (
	"strings"
	"sync"
	"unicode"
)

// MetaConfig — knobs. Defaults match the spirit of legacy metaleo.py
// but the thresholds are tuned for neoleo's pulse range
// (entropy = trauma = pain², arousal = max(FEAR/LOVE/RAGE)).
type MetaConfig struct {
	BufSize      int     // dynamic bootstrap depth
	MaxSnippet   int     // chars per snippet
	MaxWeight    float32 // clamp ceiling
	EntropyLow   float32 // rigid threshold (entropy *low* raises weight)
	TraumaHigh   float32 // wound-active threshold
	ArousalHigh  float32 // emotional-charge threshold
	QualityLow   float32 // base reply weak
	SpeakWeight  float32 // weight floor for metaleo to speak
	WinMargin    float32 // how much higher q_meta must be over q_base
	MetaTemp     float32 // generate temperature for alternative
	MetaMaxToken int     // generate cap for alternative
}

func defaultMetaConfig() MetaConfig {
	return MetaConfig{
		BufSize:      8,
		MaxSnippet:   200,
		MaxWeight:    0.5,
		EntropyLow:   0.10, // pain² stays low most of the time; rigid means *too* calm
		TraumaHigh:   0.30, // pain² > 0.30 ≈ pain > 0.55
		ArousalHigh:  0.7,
		QualityLow:   0.4,
		SpeakWeight:  0.2,
		WinMargin:    0.05,
		MetaTemp:     1.1,
		MetaMaxToken: 60,
	}
}

// MetaLeo — the inner voice. Owns its bootstrap buffer; the field
// itself lives behind LeoGo. Goroutine-safe through its own mutex.
type MetaLeo struct {
	cfg    MetaConfig
	mu     sync.Mutex
	buf    []string // bootstrap fragments, newest at end (capped at BufSize)
	count  int      // total feeds (for stats)
	speaks int      // times metaleo overrode the base reply
}

func NewMetaLeo() *MetaLeo {
	return &MetaLeo{cfg: defaultMetaConfig(), buf: make([]string, 0, 8)}
}

// Feed updates the dynamic bootstrap buffer.
//
// Sources:
//   - the just-emitted reply, if arousal is high (emotional charge)
//   - any ring-2 / meta shard texts from this cycle's overthinking
//
// Called from the worker (after rings observe, before soma snapshot)
// or directly from the reply path. Never blocks the reply.
func (m *MetaLeo) Feed(reply string, arousal float32, ringShards []string) {
	if m == nil {
		return
	}
	var fresh []string
	for _, s := range ringShards {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if len(s) > m.cfg.MaxSnippet {
			s = s[:m.cfg.MaxSnippet]
		}
		fresh = append(fresh, s)
	}
	if arousal > m.cfg.ArousalHigh {
		s := strings.TrimSpace(reply)
		if s != "" {
			if len(s) > m.cfg.MaxSnippet {
				s = s[:m.cfg.MaxSnippet]
			}
			fresh = append(fresh, s)
		}
	}
	if len(fresh) == 0 {
		return
	}
	m.mu.Lock()
	for _, s := range fresh {
		m.buf = append(m.buf, s)
	}
	if len(m.buf) > m.cfg.BufSize {
		m.buf = m.buf[len(m.buf)-m.cfg.BufSize:]
	}
	m.count++
	m.mu.Unlock()
}

// computeWeight reads the pulse and base-reply quality and returns
// how strong the inner voice should be on this turn. Pure decision,
// no field touched.
func (m *MetaLeo) computeWeight(pulse Pulse, qualityBase float32) float32 {
	w := float32(0.1) // base whisper

	// LOW entropy (rigid) → speak up
	if pulse.Entropy < m.cfg.EntropyLow {
		w += 0.15
	}
	// HIGH trauma (wound active) → surface
	if pulse.Entropy > m.cfg.TraumaHigh {
		w += 0.15
	}
	// LOW base quality → maybe metaleo can help
	if qualityBase < m.cfg.QualityLow {
		w += 0.10
	}
	// HIGH arousal → slight boost
	if pulse.Arousal > m.cfg.ArousalHigh {
		w += 0.05
	}
	if w > m.cfg.MaxWeight {
		w = m.cfg.MaxWeight
	}
	if w < 0 {
		w = 0
	}
	return w
}

// generateAlternative builds a seed from the bootstrap buffer plus
// the base reply and asks Leo to generate (read-only) at higher
// temperature. Returns "" if no buffer or generation produced
// nothing usable.
func (m *MetaLeo) generateAlternative(leo *LeoGo, baseReply string) string {
	m.mu.Lock()
	if len(m.buf) == 0 {
		m.mu.Unlock()
		return ""
	}
	seed := strings.Join(m.buf, " ") + " " + strings.TrimSpace(baseReply)
	m.mu.Unlock()

	if strings.TrimSpace(seed) == "" {
		return ""
	}
	alt := leo.GenerateRing(seed, m.cfg.MetaTemp, m.cfg.MetaMaxToken)
	switch alt {
	case "", ".", "...":
		return ""
	}
	return alt
}

// Process — async decision point. Called by the worker after the
// three rings have observed, before soma snapshot. Feeds the
// dynamic bootstrap from this turn, decides if metaleo should
// speak, generates an alternative, and (if it wins) observes the
// alternative back into the field with the metaleo_voice tag.
//
// Returns true iff metaleo overrode (the alternative was observed).
// The caller does not need this — overthinking continues either
// way — but it is useful for /stats and tests.
//
// Any failure inside this function is silent: metaleo NEVER breaks
// Leo. The reply that the user saw is unchanged in either case.
func (m *MetaLeo) Process(leo *LeoGo, prompt, baseReply string, pulse Pulse, ringShards []string) bool {
	if m == nil || leo == nil {
		return false
	}
	defer func() { _ = recover() }() // metaleo must NEVER break Leo

	// Always feed the buffer — the dynamic bootstrap learns
	// from every turn, even when metaleo does not speak.
	m.Feed(baseReply, pulse.Arousal, ringShards)

	qBase := score(baseReply)
	weight := m.computeWeight(pulse, qBase)
	if weight <= 0 {
		return false
	}

	alt := m.generateAlternative(leo, baseReply)
	if alt == "" {
		return false
	}

	qAlt := score(alt)
	if !(qAlt > qBase+m.cfg.WinMargin && weight > m.cfg.SpeakWeight) {
		return false
	}

	// Metaleo wins. Observe the alternative back into the field
	// with the metaleo_voice tag so the next reply is coloured
	// by the inner voice.
	leo.ObserveThought(alt, "overthinking:metaleo_voice")
	m.mu.Lock()
	m.speaks++
	m.mu.Unlock()
	return true
}

// Stats returns (feeds, overrides, bufLen) — read by /stats command.
func (m *MetaLeo) Stats() (int, int, int) {
	if m == nil {
		return 0, 0, 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.count, m.speaks, len(m.buf)
}

// score — composite quality of a reply. Adapted from legacy
// metaleo._assess: 0.4·coherence + 0.4·resonance + 0.2·length.
func score(text string) float32 {
	if text == "" {
		return 0
	}
	c := scoreCoherence(text)
	r := scoreResonance(text)
	l := scoreLength(text)
	s := 0.4*c + 0.4*r + 0.2*l
	if s < 0 {
		s = 0
	}
	if s > 1 {
		s = 1
	}
	return s
}

func scoreCoherence(text string) float32 {
	var s float32
	endings := 0
	for _, r := range text {
		if r == '.' || r == '!' || r == '?' {
			endings++
		}
	}
	if endings >= 1 {
		s += 0.3
	}
	if endings >= 2 {
		s += 0.2
	}
	// capitalized starts (rough — split on .!? and check first rune)
	parts := splitSentences(text)
	caps := 0
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		first := []rune(p)[0]
		if unicode.IsUpper(first) {
			caps++
		}
	}
	if caps > 0 {
		s += 0.2
	}
	// last word ≥ 3 chars (penalize fragment endings)
	words := strings.Fields(text)
	if n := len(words); n > 0 && len(words[n-1]) >= 3 {
		s += 0.1
	}
	return clamp(s)
}

func scoreResonance(text string) float32 {
	words := strings.Fields(strings.ToLower(text))
	if len(words) < 3 {
		return 0
	}
	uniq := make(map[string]int, len(words))
	for _, w := range words {
		uniq[w]++
	}
	uniqRatio := float32(len(uniq)) / float32(len(words))

	// bigram diversity
	bgUniq := make(map[string]struct{}, len(words))
	for i := 0; i+1 < len(words); i++ {
		bgUniq[words[i]+" "+words[i+1]] = struct{}{}
	}
	var bgRatio float32
	if len(words) > 1 {
		bgRatio = float32(len(bgUniq)) / float32(len(words)-1)
	}

	// repetition penalty
	maxRep := 0
	for _, n := range uniq {
		if n > maxRep {
			maxRep = n
		}
	}
	var rep float32
	if maxRep > 2 {
		rep = float32(maxRep-2) * 0.1
	}
	return clamp(uniqRatio*0.5 + bgRatio*0.5 - rep)
}

func scoreLength(text string) float32 {
	const target = 40
	n := len(strings.Fields(text))
	if n < 5 {
		return 0.2
	}
	if n > target*2 {
		return 0.5
	}
	dev := float32(n-target) / float32(target)
	if dev < 0 {
		dev = -dev
	}
	return clamp(1.0 - dev)
}

func splitSentences(text string) []string {
	var out []string
	last := 0
	for i, r := range text {
		if r == '.' || r == '!' || r == '?' {
			out = append(out, text[last:i+1])
			last = i + 1
		}
	}
	if last < len(text) {
		out = append(out, text[last:])
	}
	return out
}

func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
