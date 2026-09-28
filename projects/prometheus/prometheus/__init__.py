"""prometheus — the model that writes its own curriculum.

The loop: probe Hydro-1 per domain → rank weaknesses → commission a targeted
textbook (weak domains get double space) → distill via the Academy → retrain →
re-measure. Every iteration is persisted so progress is visible over time.
"""
from __future__ import annotations

from .curriculum import DOMAINS, Probe
from .prober import ProbeRunner, Assessment
from .planner import make_plan, Plan
from .loop import ImprovementLoop

__all__ = ["DOMAINS", "Probe", "ProbeRunner", "Assessment", "make_plan", "Plan", "ImprovementLoop"]
__version__ = "0.1.0"
