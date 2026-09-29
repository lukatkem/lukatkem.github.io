# Pareto — route each request to the cheapest model that provably meets your quality floor

**Vendors claim 80% cost cuts with zero quality proof. Pareto uses *your own
golden set* as the oracle** — measure which models actually pass for prompts
like yours, route to the cheapest that works, and track the savings.

## Quickstart

```bash
python -m pytest -q    # 10 tests
```

```python
from pareto.router import QualityRouter, ModelTier
router = QualityRouter(
    models=[ModelTier("cheap", 0.001), ModelTier("premium", 0.10)],
    quality_floor=0.9)
router.add_eval_data(eval_records)      # from your golden set
decision = router.route(embedding)      # cheapest model that passes
```

## Honest scope

Cosine similarity over provided embeddings — no built-in embedding model.
The quality floor is per-router, not per-domain. Premium escalation is
all-or-nothing (no partial responses).
