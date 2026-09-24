"""The generated QA pair."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pair:
    doc_id: str
    kind: str
    question: str
    answer: str
    source: str
