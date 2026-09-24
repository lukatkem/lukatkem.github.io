"""ctxpack — pack the maximum document value into a context window.

Every RAG system hits this: more retrieved documents than fit. These packers
assemble the highest-value context that fits a character budget — greedily,
by value density, or with an exact knapsack — and every drop carries a named
reason.
"""
from __future__ import annotations

from .budget import Budget
from .doc import Doc, DocError
from .packer import Packing, pack_density, pack_exact, pack_greedy
from .render import render, render_report

__all__ = ["Budget", "Doc", "DocError", "Packing",
           "pack_greedy", "pack_density", "pack_exact", "render", "render_report"]
__version__ = "1.0.0"
