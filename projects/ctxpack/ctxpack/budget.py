"""The character budget for a packed context window."""
from __future__ import annotations

from dataclasses import dataclass

from .doc import DocError


@dataclass
class Budget:
    chars: int
    header_chars: int = 0
    per_doc_overhead: int = 40

    def __post_init__(self) -> None:
        if self.chars < 0 or self.header_chars < 0 or self.per_doc_overhead < 0:
            raise DocError("budget values must be non-negative")
        if self.header_chars > self.chars:
            raise DocError("header_chars cannot exceed chars")

    def usable(self) -> int:
        return self.chars - self.header_chars
