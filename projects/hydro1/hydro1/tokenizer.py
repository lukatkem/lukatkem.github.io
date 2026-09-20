"""Character-level tokenizer — deterministic, tiny vocab, zero dependencies.

Char-level is a deliberate choice for a from-scratch demo: no tokenizer model
to download, the vocabulary is just the corpus's observed characters, and the
playground can show per-character attention honestly.
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocab: str):
        # stable order: sorted unique chars
        self.chars = sorted(set(vocab))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        # <unk> handling: any unseen char maps to space (keeps generation sane)
        self.unk = self.stoi.get(" ", 0)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        return cls("".join(data["chars"]))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"chars": self.chars}))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unk) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, " ") for i in ids)
