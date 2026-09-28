# FaultLine — chaos engineering for LLM agent tool loops

**Your agent works in staging. FaultLine finds out what it does when the world breaks.**
Injects timeouts, malformed JSON, truncated responses, empty bodies, and
plausible-but-wrong data into an agent's tool layer — then classifies every run:
*recovered · degraded · silent failure · crashed*. The headline metric is the
**silent failure rate**: confident answers built on corrupted tool data.

## Quickstart

```bash
python -m pytest -q          # 20 tests
python -m faultline demo     # a real agentcore agent under injected faults
```

## The verdict taxonomy — the product

| verdict | meaning |
|---|---|
| CORRECT_WITH_FAULTS | tool failed, agent still delivered the right answer |
| DEGRADED | agent noticed and hedged ("I couldn't read the note…") |
| **SILENT_FAILURE** | agent confidently answered from corrupted data — the dangerous case |
| CRASHED | the loop died |

Deterministic fault scheduler (fixed-seed LCG), per-tool fault assignments with
probabilities, transparent pass-through when no fault matches, and a recovery-rate
report with an SVG donut + verdict matrix.

## Honest scope

The verdict classifier is heuristic (keyword + hedging markers + tool-output diff) —
expected outcomes must be provided per scenario. It measures the loop you point it at
(the sibling agentcore runtime is the built-in subject).
