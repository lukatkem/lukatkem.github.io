# evalforge — documents → golden-set eval pairs, offline

**Deterministic question generation from your own docs — no LLM calls, no cost,
no randomness.** Definition, numeric, and cloze patterns over sentence structure;
quality gates on every pair; dedup; versioned JSON output with a margin-compatible
adapter for the sibling copilot's golden set.

## Quickstart

```bash
python -m pytest -q                                     # 12 tests
python -m evalforge demo                                # 3 documents → pairs walkthrough
python -m evalforge generate --docs docs.json --name myset --out evalset.json
```

## Output shape

```json
{"name": "myset", "n": 42, "cases": [
  {"id": "myset-001", "doc": "faq", "kind": "numeric",
   "question": "How many days does the policy have?", "answer": "30", "source": "…"}]}
```

`to_margin_cases()` emits `{question, expected, type}` — drop it straight into
the sibling margin copilot's golden set and wire it into CI.

## Honest scope

Patterns are structural (definitions, numbers, capitalized-entity cloze) — a
small deterministic starter set, not an LLM question generator. Filters drop
short questions, leaking answers, and duplicates.
