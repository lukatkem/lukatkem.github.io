"""Stage 1 — normalization: make bytes of different provenance comparable."""
from __future__ import annotations

import re
import unicodedata

_MULTISPACE = re.compile(r"[ \t\u00a0]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize(text: str) -> str:
    """NFKC unicode, unified newlines, collapsed runs of space, no control chars.

    Deterministic and idempotent: normalize(normalize(t)) == normalize(t)."""
    t = unicodedata.normalize("NFKC", text)
    t = _CTRL.sub("", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _MULTISPACE.sub(" ", t)
    t = _MULTINEWLINE.sub("\n\n", t)
    return t.strip()
