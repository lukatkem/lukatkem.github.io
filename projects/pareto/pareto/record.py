"""Eval records, routing decisions, and the cost-quality ledger."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalRecord:
    """One measured (prompt, model, outcome) triple from your golden set."""
    prompt_id: str
    embedding: list[float]          # simplified: list of floats as the "embedding"
    model: str
    passed: bool
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id is required")
        if not self.model:
            raise ValueError("model is required")


@dataclass
class RouteDecision:
    """The outcome of one routing decision."""
    model: str
    reason: str
    estimated_cost: float
    fallback_used: bool = False


@dataclass
class QualityLedger:
    """Tracks cost savings and quality outcomes across all routed requests."""
    total_requests: int = 0
    cheap_routed: int = 0
    premium_fallbacks: int = 0
    total_saved_usd: float = 0.0
    quality_maintained: int = 0      # cheap route responses that passed eval

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "cheap_routed": self.cheap_routed,
            "premium_fallbacks": self.premium_fallbacks,
            "total_saved_usd": round(self.total_saved_usd, 4),
            "quality_maintained": self.quality_maintained,
        }
