"""Stage 4 — quality scoring, Gopher/C4-style heuristics, no models.

Each rule is a named check producing either a hard rejection or a soft
penalty. The reason list is the product: "why was this document dropped"
is the question every corpus builder asks a thousand times.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_LOREM = re.compile(r"lorem ipsum", re.I)
_URL = re.compile(r"https?://\S+")
_LINE = re.compile(r"^[^\n]{25,}$", re.MULTILINE)


@dataclass
class Quality:
    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _word_stats(text: str) -> tuple[list[str], float]:
    words = re.findall(r"\S+", text)
    if not words:
        return [], 0.0
    mean_len = sum(len(w) for w in words) / len(words)
    return words, mean_len


def quality(
    text: str,
    min_words: int = 25,
    max_symbols: float = 0.15,
    min_score: float = 0.6,
) -> Quality:
    """Score 0..1 with named reasons. Hard rules reject; soft rules subtract."""
    reasons: list[str] = []
    score = 1.0

    words, mean_len = _word_stats(text)
    if len(text) < 200:
        return Quality(0.0, False, ["too-short (<200 chars)"])
    if len(words) < min_words:
        return Quality(0.0, False, [f"too-few-words (<{min_words})"])

    n = len(text)
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    sym_ratio = symbols / n
    if sym_ratio > max_symbols:
        return Quality(0.0, False, [f"symbol-soup ({sym_ratio:.0%} non-text)"])

    # repeated substantial lines — boilerplate / crawler output
    lines = [m.group(0).strip() for m in _LINE.finditer(text)]
    seen: dict[str, int] = {}
    for ln in lines:
        seen[ln] = seen.get(ln, 0) + 1
    max_rep = max(seen.values(), default=1)
    if max_rep >= 3:
        return Quality(0.0, False, [f"repeated-lines (one line ×{max_rep})"])
    if len([ln for ln, c in seen.items() if c > 1]) >= 2:
        score -= 0.4
        reasons.append("repeated-lines (multiple ×2)")

    if _LOREM.search(text):
        score -= 0.5
        reasons.append("lorem-ipsum")

    letters = sum(1 for c in text if c.isalpha())
    if letters / n > 0.3:
        upper = sum(1 for c in text if c.isupper())
        if upper / max(letters, 1) > 0.4:
            score -= 0.3
            reasons.append(f"shouting ({upper / max(letters, 1):.0%} caps)")

    if mean_len > 9:
        score -= 0.3
        reasons.append(f"mean-word-length ({mean_len:.1f})")

    urls = len(_URL.findall(text))
    if urls > max(len(words) / 100, 3):
        score -= 0.3
        reasons.append(f"link-farm ({urls} urls)")

    return Quality(max(score, 0.0), score >= min_score, reasons)
