"""promptkit — prompts as versioned, tested code.

A prompt is production logic; treat it like it. Templates with typed slots,
assertion-based regression tests, and an offline runner over any BaseLLM
(mock included) — so changing a prompt is a code change with a test gate.
"""
from __future__ import annotations

from .asserts import AssertionFailed, contains, not_contains, max_length, parse, regex
from .prompt import Prompt, PromptError
from .runner import Case, RunResult, Suite, SuiteReport

__all__ = ["Prompt", "PromptError", "Case", "Suite", "SuiteReport", "RunResult", "parse",
           "contains", "not_contains", "max_length", "regex", "AssertionFailed"]
__version__ = "1.0.0"
