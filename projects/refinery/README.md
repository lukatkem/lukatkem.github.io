# Refinery — raw text in, training-grade corpus out

A from-scratch **data refinery for LLM training corpora**. Every algorithm
inside — MinHash, banded LSH, union-find, heuristic quality scoring — is
implemented in this package with **zero dependencies** (pure standard
library). No models are downloaded and no models are needed.

Data curation is the least glamorous and most decisive part of applied AI:
a model can only be as good as the corpus it trains on, and real corpora
arrive full of near-duplicates, crawler junk, other-language noise, and —
more often than anyone admits — leaked API keys and personal data.

## The pipeline

| stage | what it does |
|---|---|
| `normalize` | unicode NFKC, unified newlines, collapsed whitespace (idempotent) |
| `pii` | redacts emails, phone numbers, card/ID digits, API keys, JWTs → `[REDACTED]` |
| `language` | stopword-coverage + letter-share filter (built-in English profile) |
| `quality` | Gopher/C4-style heuristics: symbol soup, repeated lines, shouting, link farms, lorem — every drop carries a named reason |
| `exact dedup` | md5 of normalized text |
| `near dedup` | word 5-gram shingles → 64-perm MinHash signatures → 16-band LSH candidates → Jaccard estimate → union-find clusters; keeps one representative |

Near-duplicate math: two documents sharing a band bucket (16 bands × 4 rows)
are compared; a true 0.8-duplicate escapes all 16 bands with probability
≈ 0.4%.

## Usage

```bash
python -m refinery.cli --input DIR --out corpus.clean.txt --report report.md
python -m refinery.cli --input a.txt --input b.txt --threshold 0.8
```

The **report** is the product: a stage-by-stage funnel, drop counts by
reason, every scrubbed PII/secret occurrence, and sample drops with
receipts — the questions every corpus builder asks a thousand times.

## Demo

```bash
python demo/make_dirty_corpus.py     # 100 documents: dups, junk, PII leaks, foreign text
python -m refinery.cli --input demo/dirty --out demo/clean.txt --report demo/report.md
```

Last run's funnel: **100 → 81** (junk + foreign rejected) **→ 60** (exact
duplicates removed) **→ 53** (near-duplicates clustered away), with
4 emails, 4 phone numbers and 4 leaked secrets redacted — in 0.05 s.

## Tests

```bash
python -m pytest tests/ -q     # 6 tests: every stage gets dirty synthetic input
```
