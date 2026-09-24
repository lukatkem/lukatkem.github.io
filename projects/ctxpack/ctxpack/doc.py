"""Document and budget primitives with validation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


class DocError(ValueError):
    pass


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    score: float

    def __post_init__(self) -> None:
        if not self.id:
            raise DocError("doc id must be non-empty")
        if math.isnan(self.score) or math.isinf(self.score):
            raise DocError(f"doc {self.id}: score must be finite")
        if self.score < 0:
            raise DocError(f"doc {self.id}: score must be >= 0")

    @property
    def cost(self) -> int:
        """Characters charged for this doc: text + framing overhead."""
        return len(self.text)
