# ctxpack — pack the maximum document value into a context window

**More documents than fit? Pack the maximum value — provably.** Every RAG
system hits this daily: the retriever returns ten documents, the context window
holds three. These packers choose which three — greedily, by value density, or
with an exact knapsack — and every dropped document carries a named reason.

## Quickstart

```bash
python -m pytest -q                                        # 19 tests
python -m ctxpack demo                                     # 3 algorithms compared
python -m ctxpack pack --docs docs.json --chars 2000 --algorithm exact
```

## The three algorithms

| algorithm | strategy | use when |
|---|---|---|
| `greedy` | highest score first, stop at first non-fit | scores already reflect value per size |
| `density` | score-per-character first, skip-and-continue | small dense docs compete with long ones |
| `exact` | 0/1 knapsack DP — provably optimal total score | you want the best possible set |

All three: never exceed the budget, every drop gets a named reason
(`budget_exhausted` / `too_large_alone` / `too_large_skipped` / `not_selected`),
ties break by id, and the render emits a framed, auditable prompt.

## Honest scope

Budgets are characters, not tokens — pair with a tokenizer (see sibling
`tokenforge`) for token-accurate packing. The knapsack uses per-character DP by
default (cheap at realistic window sizes); a granularity knob trades a little
optimality for memory on very large windows.
