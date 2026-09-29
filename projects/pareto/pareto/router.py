"""The quality-grounded router — routes to the cheapest model that provably
meets the quality floor, based on measured eval data (not marketing claims)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .record import EvalRecord, RouteDecision, QualityLedger


@dataclass
class ModelTier:
    """A model with its cost per request."""
    name: str
    cost_usd: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class QualityRouter:
    """Routes prompts to the cheapest model whose nearest eval neighbors pass.

    Uses measured eval data as the quality oracle: for a new prompt, find the
    k nearest neighbors in the eval set (by cosine similarity of embeddings),
    check which models passed for those neighbors, and route to the cheapest
    model with a pass rate above the floor.
    """

    def __init__(self, models: list[ModelTier], quality_floor: float = 0.9,
                 k_neighbors: int = 5, premium_model: str = ""):
        self.models = sorted(models, key=lambda m: m.cost_usd)   # cheapest first
        self.floor = quality_floor
        self.k = k_neighbors
        self.premium = premium_model or (models[-1].name if models else "")
        self.records: list[EvalRecord] = []
        self.ledger = QualityLedger()

    def add_eval_data(self, records: list[EvalRecord]) -> None:
        self.records.extend(records)

    def route(self, embedding: list[float]) -> RouteDecision:
        """Route a new prompt based on its embedding."""
        self.ledger.total_requests += 1
        # find nearest neighbors in the eval set
        scored = []
        for rec in self.records:
            if len(rec.embedding) != len(embedding):
                continue
            sim = cosine_similarity(rec.embedding, embedding)
            scored.append((sim, rec))
        scored.sort(key=lambda x: -x[0])
        neighbors = scored[:self.k]

        if not neighbors:
            # no data — route to premium
            self.ledger.premium_fallbacks += 1
            cost = next((m.cost_usd for m in self.models if m.name == self.premium), 0.0)
            return RouteDecision(model=self.premium, reason="no eval data — premium fallback",
                                 estimated_cost=cost, fallback_used=True)

        # group neighbors by model, compute pass rate per model
        model_stats: dict[str, list[bool]] = {}
        for sim, rec in neighbors:
            model_stats.setdefault(rec.model, []).append(rec.passed)

        # find cheapest model with pass rate >= floor
        for tier in self.models:                     # cheapest first
            if tier.name in model_stats:
                outcomes = model_stats[tier.name]
                pass_rate = sum(outcomes) / len(outcomes)
                if pass_rate >= self.floor:
                    self.ledger.cheap_routed += 1
                    premium_cost = next((m.cost_usd for m in self.models if m.name == self.premium), 0.0)
                    self.ledger.total_saved_usd += max(0, premium_cost - tier.cost_usd)
                    self.ledger.quality_maintained += 1
                    return RouteDecision(model=tier.name,
                                         reason=f"pass rate {pass_rate:.0%} over {len(outcomes)} neighbors",
                                         estimated_cost=tier.cost_usd)

        # no model meets the floor — escalate to premium
        self.ledger.premium_fallbacks += 1
        premium_cost = next((m.cost_usd for m in self.models if m.name == self.premium), 0.0)
        return RouteDecision(model=self.premium,
                             reason="no cheap model meets quality floor — premium escalation",
                             estimated_cost=premium_cost, fallback_used=True)

    def record_outcome(self, model: str, embedding: list[float], passed: bool,
                       cost_usd: float = 0.0) -> None:
        """Feed back the outcome of a routed request to improve future routing."""
        self.records.append(EvalRecord(prompt_id=f"live-{len(self.records)}",
                                       embedding=embedding, model=model,
                                       passed=passed, cost_usd=cost_usd))
