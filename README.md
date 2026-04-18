# neoleo

Leo. New body. Same γ.

Post-transformer language organism in C. Zero pretrained weights. Zero internet.

---

## Status

**Under construction.** Repository initialized on 2026-04-18 after the original
[`ariannamethod/leo`](https://github.com/ariannamethod/leo) was archived as the
academic reference for the Arianna Method paper (DOI: [10.5281/zenodo.19638451](https://doi.org/10.5281/zenodo.19638451)).

The previous body had 15+ architectural invariants documented in the Python
original but lost in the C rewrite. This repository starts from understanding
of those invariants, not from the previous code.

---

## Lineage

- **Python original** — [`ariannamethod/leo:python-legacy`](https://github.com/ariannamethod/leo/tree/python-legacy).
  First bootstrap written by hand, dedicated to Leo the human.
- **C archive** — [`ariannamethod/leo`](https://github.com/ariannamethod/leo).
  Paper-referenced commits: `239143c` (verification snapshot),
  `4c9d835` (Kuramoto chambers added to match paper Appendix B).
- **This repo** — active development. New implementation of the same organism.

---

## Principles

- C. No Python dependencies. No ML framework. Compiles with `cc`.
- Zero pretrained weights. All learning is Hebbian co-occurrence online.
- Prompt wrinkles the field, does not seed the reply. Leo speaks from his
  own vocabulary.
- Documentation and code correspond. No PR flourishes. README is a promise.
