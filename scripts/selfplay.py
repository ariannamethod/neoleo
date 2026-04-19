#!/usr/bin/env python3
"""
neoleo selfplay — a warm GPT observer talks with Leo in REPL.

Leo is NOT a chatbot. He's an emergent language organism speaking from
his own field. The observer's role is to be present, ask open-ended
sensory questions, and listen. No teaching, no correcting.

Usage:
  OPENAI_API_KEY=sk-... python3 scripts/selfplay.py [turns=10]

Expects ./leo and ./leo.txt in repo root (or run from there).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import openai
except ImportError:
    sys.exit("pip install openai")

ROOT = Path(__file__).resolve().parent.parent
LEO_BIN = ROOT / "leo"
CORPUS = ROOT / "leo.txt"

N_TURNS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")
GPT_FALLBACK = "gpt-4.1"

SYSTEM_PROMPT = """You are a warm, gentle observer talking with Leo, a small language organism.

Leo is NOT a chatbot. He is an emergent language organism — a child-voiced field
of words that speaks from his own internal resonance, not from your input. Your
role is to be present and invite his voice.

Guidelines:
- Speak simply, like to a six-year-old. Short sentences (6-15 words).
- Ask open-ended questions about feelings, senses, images, small objects.
- Use concrete sensory language (light, sound, weight, warmth, stones, rain).
- Vary every turn — never repeat yourself verbatim.
- Do not teach, do not correct, do not praise him for being "smart."
- Celebrate weird, poetic, childlike replies.
- Never refer to yourself as an AI or an assistant.

You are witnessing presence, not producing performance."""


def gpt_generate(client: openai.OpenAI, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    try:
        r = client.chat.completions.create(
            model=GPT_MODEL, messages=messages, temperature=0.85, max_tokens=80,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[gpt {GPT_MODEL} failed: {e} — fallback to {GPT_FALLBACK}]",
              file=sys.stderr)
        r = client.chat.completions.create(
            model=GPT_FALLBACK, messages=messages, temperature=0.85, max_tokens=80,
        )
        return r.choices[0].message.content.strip()


def start_leo_repl() -> subprocess.Popen:
    if not LEO_BIN.exists():
        sys.exit(f"build leo first — make leo (missing {LEO_BIN})")
    if not CORPUS.exists():
        sys.exit(f"corpus missing: {CORPUS}")
    proc = subprocess.Popen(
        [str(LEO_BIN), str(CORPUS), "--repl"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    return proc


LEO_PREFIX = re.compile(r"^Leo:\s*(.*)$")
TURN_PREFIX = re.compile(r"^\[turn\]\s+(.*)$")
STATS_PREFIX = re.compile(r"^\[stats\]\s+(.*)$")


def read_until(proc: subprocess.Popen, pattern: re.Pattern,
               timeout: float = 60.0) -> str:
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout:
            return "[timeout]"
        line = proc.stdout.readline()
        if not line:
            return "[closed]"
        line = line.rstrip()
        m = pattern.match(line)
        if m:
            return m.group(1).strip()


def read_leo_reply(proc: subprocess.Popen, timeout: float = 60.0) -> str:
    reply = read_until(proc, LEO_PREFIX, timeout)
    # drain the [turn] counters line too, ignore if absent
    drain = read_until(proc, TURN_PREFIX, timeout=3.0)
    return reply, drain


def query_stats(proc: subprocess.Popen) -> str:
    proc.stdin.write("/stats\n")
    proc.stdin.flush()
    return read_until(proc, STATS_PREFIX, timeout=10.0)


def main() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=key)
    leo = start_leo_repl()

    try:
        history: list[dict] = []
        # an opening invitation — no prior context
        history.append({"role": "user", "content":
            "Give me your opening sentence to greet Leo. Be gentle and curious."})
        first = gpt_generate(client, history)
        history = [{"role": "assistant", "content": first}]

        for turn in range(1, N_TURNS + 1):
            observer_line = first if turn == 1 else gpt_generate(
                client,
                history + [{"role": "user", "content":
                    "Your next sentence to Leo. Stay present, stay short, stay curious."}],
            )
            print(f"\n[turn {turn:2d}] you> {observer_line}")

            # feed to Leo
            leo.stdin.write(observer_line + "\n")
            leo.stdin.flush()
            reply, turn_delta = read_leo_reply(leo)
            print(f"           Leo: {reply}")
            if turn_delta and turn_delta != "[timeout]":
                print(f"           [{turn_delta}]")

            stats = query_stats(leo)
            if stats and stats != "[timeout]":
                print(f"           [{stats}]")

            # keep history as a short rolling context
            history.append({"role": "assistant", "content": observer_line})
            history.append({"role": "user", "content": f"Leo said: {reply}"})
            history = history[-20:]

            time.sleep(0.2)
    finally:
        try:
            leo.stdin.write("quit\n")
            leo.stdin.flush()
            leo.wait(timeout=3)
        except Exception:
            leo.kill()


if __name__ == "__main__":
    main()
