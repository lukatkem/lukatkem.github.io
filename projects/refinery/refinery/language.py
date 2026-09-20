"""Stage 3 — language filter, without any model.

A tiny from-scratch profile: real text in the target language leans on a
small set of very common words and is mostly letters. Gibberish, symbol
soup, and other-script text fail both signals at once.
"""
from __future__ import annotations

import re
import string

EN_STOPWORDS = frozenset(
    """the be to of and a in that have i it for not on with he as you do at
    this but his by from they we say her she or an will my one all would
    there their what so up out if about who get which go me when make can
    like time no just him know take people into year your good some could
    them see other than then now look only come its over think also back
    after use two how our work first well way even new want because any
    these give day most us was were has had said them more very""".split()
)

_ASCII_LETTERS = set(string.ascii_letters)
_WORD = re.compile(r"[a-z']+")


def language_score(text: str) -> float:
    """0..1 — blend of stopword coverage and ASCII-letter share."""
    if not text:
        return 0.0
    words = _WORD.findall(text.lower())
    if not words:
        return 0.0
    stop_hits = sum(1 for w in words if w in EN_STOPWORDS)
    stop_ratio = min(stop_hits / len(words) / 0.35, 1.0)  # healthy prose ≈ 0.3–0.5
    letters = sum(1 for c in text if c in _ASCII_LETTERS)
    letter_ratio = min(letters / len(text) / 0.75, 1.0)
    return 0.5 * stop_ratio + 0.5 * letter_ratio


def is_language(text: str, lang: str = "en", min_score: float = 0.45) -> bool:
    if lang != "en":
        raise ValueError("built-in profile is English; add profiles for more languages")
    return language_score(text) >= min_score
