"""Stage 2 — PII & secret scrubbing.

Two different risks live here: personal data (emails, phone numbers, ID
numbers) and credentials that leaked into text (API keys, JWTs). Training
on either is a real incident — scrub before the bytes ever reach a trainer.
"""
from __future__ import annotations

import re

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # api keys / tokens people actually paste into documents
    ("secret", re.compile(
        r"(?:sk-or-v1-[\w-]{20,}|sk-ant-[\w-]{20,}|sk-[\w]{20,}|nvapi-[\w-]{20,}"
        r"|ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|xox[bap]-[\w-]{10,})"
    )),
    # JWTs: three base64url segments
    ("secret", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b")),
    # long bare digit runs: card numbers, national ids, tracking numbers
    ("digit-id", re.compile(r"(?<![\d-])\d{10,}(?![\d-])")),
    # phone-ish: needs >= 9 digits and must not be a date
    ("phone", re.compile(r"(?<![\d-])(?:\+?\d[\d\s().-]{7,}\d)(?![\d-])")),
]

_DATE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$")
_DIGITS = re.compile(r"\d")


def _is_phone(candidate: str) -> bool:
    if _DATE.match(candidate.strip()):
        return False
    return len(_DIGITS.findall(candidate)) >= 9


def scrub(text: str, placeholder: str = "[REDACTED]") -> tuple[str, dict[str, int]]:
    """Replace PII/secrets with a placeholder. Returns (clean_text, counts_by_kind).

    Date-like strings (2026-09-20) survive; real phone numbers do not."""
    counts: dict[str, int] = {}

    def sub(kind: str, m: re.Match[str]) -> str:
        if kind == "phone" and not _is_phone(m.group(0)):
            return m.group(0)  # a date or a short number — leave it alone
        counts[kind] = counts.get(kind, 0) + 1
        return placeholder

    out = text
    for kind, pattern in PATTERNS:
        out = pattern.sub(lambda m, k=kind: sub(k, m), out)
    return out, counts
