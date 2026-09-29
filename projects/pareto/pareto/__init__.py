"""Pareto — route each request to the cheapest model that provably meets your quality floor.

Uses measured eval data (not marketing benchmarks) to make routing decisions:
record (prompt-embedding, model, pass/fail) tuples from your golden set, then
route new prompts to the cheapest model whose nearest neighbors pass.
"""
from __future__ import annotations

from .record import EvalRecord, RouteDecision, QualityLedger
from .router import QualityRouter

__all__ = ["EvalRecord", "RouteDecision", "QualityLedger", "QualityRouter"]
__version__ = "0.1.0"
