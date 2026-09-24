"""evalforge — turn documents into golden-set evaluation pairs, offline.

No LLM calls: question generation is deterministic pattern extraction over
sentence structure (definitions, numbers, named entities as cloze). The output
is a versioned JSON eval set with a margin-compatible adapter.
"""
from __future__ import annotations

from .generate import generate_pairs
from .pairs import Pair
from .set import EvalSet
from .quality import QualityFilter

__all__ = ["generate_pairs", "EvalSet", "QualityFilter", "Pair"]
__version__ = "1.0.0"
