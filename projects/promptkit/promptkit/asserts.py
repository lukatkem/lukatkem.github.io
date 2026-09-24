"""Assertions over a model reply — each returns None or raises AssertionFailed."""
from __future__ import annotations

import re

from .errors import PromptError


class AssertionFailed(AssertionError):
    def __init__(self, name: str, detail: str):
        super().__init__(f"[{name}] {detail}")
        self.name = name


def contains(sub: str):
    def check(text: str) -> None:
        if sub.lower() not in text.lower():
            raise AssertionFailed("contains", f"{sub!r} not in reply")
    return check


def not_contains(sub: str):
    def check(text: str) -> None:
        if sub.lower() in text.lower():
            raise AssertionFailed("not_contains", f"{sub!r} must not appear")
    return check


def max_length(n: int):
    def check(text: str) -> None:
        if len(text) > n:
            raise AssertionFailed("max_length", f"reply is {len(text)} chars > {n}")
    return check


def regex(pattern: str):
    compiled = re.compile(pattern)
    def check(text: str) -> None:
        if not compiled.search(text):
            raise AssertionFailed("regex", f"pattern {pattern!r} not matched")
    return check


def parse(checks) -> list:
    """Normalize a user-supplied list/callable into callables."""
    out = []
    for c in checks or []:
        if callable(c):
            out.append(c)
        else:
            raise PromptError(f"assertion must be callable, got {type(c).__name__}")
    return out
