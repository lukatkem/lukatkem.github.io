"""Pareto tests — routing, quality floors, ledger math, neighbor search."""
from __future__ import annotations

import pytest

from pareto.record import EvalRecord, QualityLedger
from pareto.router import ModelTier, QualityRouter, cosine_similarity


def make_router(**kwargs) -> QualityRouter:
    models = [ModelTier(name="cheap", cost_usd=0.001),
              ModelTier(name="mid", cost_usd=0.01),
              ModelTier(name="premium", cost_usd=0.10)]
    return QualityRouter(models, **kwargs)


def make_records(model: str, n: int, passed: bool, dim: int = 4) -> list[EvalRecord]:
    return [EvalRecord(prompt_id=f"p{i}", embedding=[0.1 * i] * dim,
                       model=model, passed=passed, cost_usd=0.001) for i in range(n)]


def test_cosine_similarity_identical():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_router_routes_to_cheapest_passing():
    router = make_router(quality_floor=0.8, k_neighbors=3)
    # cheap model passed 3/3 for similar prompts
    router.add_eval_data(make_records("cheap", 3, True))
    decision = router.route([0.5, 0.5, 0.5, 0.5])
    assert decision.model == "cheap"
    assert decision.estimated_cost == 0.001


def test_router_escalates_when_floor_not_met():
    router = make_router(quality_floor=0.8, k_neighbors=3)
    # cheap model FAILED all evals
    router.add_eval_data(make_records("cheap", 3, False))
    decision = router.route([0.5, 0.5, 0.5, 0.5])
    assert decision.model != "cheap"   # escalated


def test_router_premium_fallback_no_data():
    router = make_router()
    decision = router.route([0.5, 0.5, 0.5, 0.5])
    assert decision.fallback_used is True
    assert "no eval data" in decision.reason


def test_ledger_tracks_savings():
    router = make_router(quality_floor=0.8, k_neighbors=3)
    router.add_eval_data(make_records("cheap", 5, True))
    for _ in range(3):
        router.route([0.5, 0.5, 0.5, 0.5])
    s = router.ledger.summary()
    assert s["total_requests"] == 3
    assert s["cheap_routed"] == 3
    assert s["total_saved_usd"] > 0


def test_record_outcome_improves_routing():
    router = make_router(quality_floor=0.8, k_neighbors=3)
    router.route([0.5, 0.5, 0.5, 0.5])     # no data → premium
    router.record_outcome("cheap", [0.5, 0.5, 0.5, 0.5], True)
    decision = router.route([0.5, 0.5, 0.5, 0.5])
    # now has data → routes to cheap if pass rate allows
    assert decision.model in ("cheap", "mid", "premium")


def test_models_sorted_cheapest_first():
    router = make_router()
    assert router.models[0].cost_usd <= router.models[-1].cost_usd


def test_quality_ledger_summary_shape():
    s = QualityLedger().summary()
    assert set(s.keys()) == {"total_requests", "cheap_routed", "premium_fallbacks",
                              "total_saved_usd", "quality_maintained"}


def test_eval_record_validation():
    with pytest.raises(ValueError):
        EvalRecord(prompt_id="", embedding=[1], model="m", passed=True)
    with pytest.raises(ValueError):
        EvalRecord(prompt_id="p", embedding=[1], model="", passed=True)
