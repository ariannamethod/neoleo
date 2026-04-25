// leogo — neoleo with a Go orchestra around the C core.
//
// Step 29b: drop-in replacement for ./leo — same CLI surface
// (--state, --fresh, --prompt, --repl), no rings yet. Smoke
// target is coherence parity with ./leo.
//
// Rings land in 29c (overthinking.go + worker goroutine).
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"sync"
)

func usage() {
	fmt.Fprintln(os.Stderr,
		"usage: leogo [corpus.txt] [options]\n"+
			"  --prompt \"text\"   respond to one prompt and exit\n"+
			"  --repl            read prompts from stdin until 'quit'\n"+
			"  --state PATH      state file (default: leo.state)\n"+
			"  --fresh           ignore existing state and ingest from scratch")
}

func readAll(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func main() {
	corpus := "leo.txt"
	statePath := "leo.state"
	fresh := false
	promptOnce := ""
	repl := false
	gotCorpus := false

	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch a {
		case "-h", "--help":
			usage()
			return
		case "--state":
			if i+1 < len(args) {
				statePath = args[i+1]
				i++
			}
		case "--fresh":
			fresh = true
		case "--prompt":
			if i+1 < len(args) {
				promptOnce = args[i+1]
				i++
			}
		case "--repl":
			repl = true
		default:
			if !gotCorpus && !strings.HasPrefix(a, "-") {
				corpus = a
				gotCorpus = true
			}
		}
	}

	fmt.Println("leogo — neoleo + Go orchestra")
	fmt.Println("post-transformer. byte-level BPE. zero weights.")
	fmt.Println()

	leo := NewLeoGo()
	if leo == nil {
		fmt.Fprintln(os.Stderr, "error: leo_bridge_create failed")
		os.Exit(1)
	}
	defer leo.Free()

	resumed := false
	if !fresh {
		if _, err := os.Stat(statePath); err == nil {
			if leo.Load(statePath) {
				resumed = true
				fmt.Printf("[resume] %s — step=%d vocab=%d\n",
					statePath, leo.Step(), leo.Vocab())
			}
		}
	}

	if !resumed {
		text, err := readAll(corpus)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: cannot read %s: %v\n", corpus, err)
			os.Exit(1)
		}
		fmt.Printf("[ingest] %s — %d bytes\n", corpus, len(text))
		leo.Ingest(text)
		if leo.Save(statePath) {
			fmt.Printf("[save] %s — step=%d vocab=%d\n",
				statePath, leo.Step(), leo.Vocab())
		}
	}

	// metaleo — Leo's inner voice. Runs inside the worker pipeline
	// after the three rings, not on the reply path. If its
	// alternative wins, it gets observed back with a distinct tag,
	// recolouring the next turn. Bootstrap buffer in-memory only.
	meta := NewMetaLeo()

	// Spawn the overthinking worker. One long-lived goroutine that
	// processes ring requests from a buffered channel. If the channel
	// is full the request is dropped — rings never block replies.
	// On exit we close the channel and wait for the worker to drain
	// before saving state, so ring effects land in leo.state too.
	ringCh := make(chan RingRequest, ringChanBufferSize)
	var workerWG sync.WaitGroup
	workerWG.Add(1)
	go workerLoop(leo, meta, ringCh, &workerWG)

	drainAndSave := func() {
		close(ringCh)
		workerWG.Wait()
		leo.Save(statePath)
	}

	if promptOnce != "" {
		reply := leo.Respond(promptOnce)
		fmt.Println(reply)
		spawnOverthinking(ringCh, promptOnce, reply)
		drainAndSave()
		return
	}

	if !repl {
		fmt.Println("no --prompt or --repl; ingest done.")
		drainAndSave()
		return
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	fmt.Println("> type prompts; 'quit' to exit; '/stats' counters; '/soma' mood trajectory; '/meta' inner voice; '/math' body advisor; '/save' persist")
	for {
		fmt.Print("\nyou> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		switch line {
		case "quit", "exit":
			drainAndSave()
			fmt.Println("[saved on exit]")
			return
		case "/stats":
			leo.Stats()
			continue
		case "/soma":
			leo.SomaDump()
			continue
		case "/meta":
			f, sp, bl := meta.Stats()
			fmt.Printf("metaleo: feeds=%d speaks=%d buf=%d/%d\n",
				f, sp, bl, meta.cfg.BufSize)
			continue
		case "/math":
			leo.MathbrainDump()
			continue
		case "/save":
			if leo.Save(statePath) {
				fmt.Printf("[saved] step=%d vocab=%d\n", leo.Step(), leo.Vocab())
			}
			continue
		}
		reply := leo.Respond(line)
		fmt.Printf("leo> %s\n", reply)
		spawnOverthinking(ringCh, line, reply)
	}

	drainAndSave()
}
