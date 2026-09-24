# tokenforge — a BPE tokenizer trained from scratch

**The piece of infrastructure every LLM uses — trained from scratch in ~300 lines.**
Byte-pair encoding: learn merge ranks from a corpus, encode any text to token ids,
decode back losslessly. Byte-level base vocabulary, so nothing is ever out of
vocabulary — emoji, Georgian, code, anything.

## Why this matters

Every LLM breathes through a tokenizer, yet almost every developer only ever
*downloads* one. This is the whole algorithm — greedy merge training, rank-ordered
encoding, deterministic ties, special tokens, lossless persistence — understood at
the level where it can be implemented and debugged, not imported.

## Quickstart

```bash
python -m pytest -q                                   # 15 tests
python -m tokenforge demo                             # train/encode/round-trip walkthrough
python -m tokenforge train --corpus file.txt --vocab 500 --out tf.json
python -m tokenforge encode --vocab tf.json --text "the quick brown fox"
python -m tokenforge decode --vocab tf.json --ids "272,289,…"
python -m tokenforge stats  --corpus file.txt --vocab tf.json
```

## API

| call | what it does |
|---|---|
| `Tokenizer().train(text, vocab_size)` | learn merges greedily (most frequent pair first; ties deterministic) |
| `encode(text, allow_special=True)` | text → ids; special tokens claim their spans whole |
| `decode(ids)` | ids → text (lossless; `errors="replace"` only on impossible bytes) |
| `register_special(name)` | allocate an id above the merge vocab |
| `save(path)` / `load(path)` | JSON persistence — round-trip equality guaranteed |
| `stats(text)` / `vocab()` | compression numbers; every token as a printable string |

## Guarantees

- **Deterministic** — same corpus, same merges, every time. No randomness.
- **Lossless** — `decode(encode(x)) == x` for any unicode, byte-level base vocab.
- **Honest training stops** — if no pair repeats (tiny corpus), training stops early
  rather than emitting meaningless merges; the vocab test documents both branches.

## Honest scope

Byte-level BPE, deliberately minimal: no regex pre-splitting (GPT-2 style), no
SentencePiece, no streaming API. Those are all add-ons to the same core loop.
Part of an eight-project from-scratch AI systems portfolio.
