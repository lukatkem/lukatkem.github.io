"""EvalSet: deduplicated, versioned, save/load, margin-compatible export."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .errors import EvalForgeError
from .generate import Pair


@dataclass
class EvalSet:
    name: str
    pairs: list[Pair] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise EvalForgeError("eval set needs a name")
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add(self, pairs: list[Pair]) -> int:
        """Add pairs, skipping exact duplicate questions. Returns added count."""
        seen = {p.question.lower() for p in self.pairs}
        added = 0
        for p in pairs:
            if p.question.lower() not in seen:
                self.pairs.append(p)
                seen.add(p.question.lower())
                added += 1
        return added

    def to_json(self) -> dict:
        return {"name": self.name, "created_at": self.created_at, "n": len(self.pairs),
                "cases": [{"id": f"{self.name}-{i+1:03d}", "doc": p.doc_id, "kind": p.kind,
                           "question": p.question, "answer": p.answer, "source": p.source}
                          for i, p in enumerate(self.pairs)]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvalSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        es = cls(name=data["name"], created_at=data.get("created_at", ""))
        for c in data["cases"]:
            es.pairs.append(Pair(c["doc"], c["kind"], c["question"], c["answer"], c.get("source", "")))
        return es

    def to_margin_cases(self) -> list[dict]:
        """Adapter for the sibling 'margin' copilot's golden-set format:
        [{"question", "expected", "type"}, …] — evals.py's own shape."""
        return [{"question": p.question, "expected": p.answer, "type": p.kind}
                for p in self.pairs]
