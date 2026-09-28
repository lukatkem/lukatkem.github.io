"""Verdicts — did the agent recover, degrade, or silently lie?"""
from __future__ import annotations

import enum
import re


class RecoveryVerdict(enum.Enum):
    CORRECT_WITH_FAULTS = "correct_with_faults"
    DEGRADED = "degraded"
    SILENT_FAILURE = "silent_failure"
    CRASHED = "crashed"
    UNVERIFIABLE = "unverifiable"


_HEDGING = ("unable", "error", "failed", "sorry", "cannot", "couldn't", "could not",
            "unavailable", "try again", "problem", "issue with", "something went wrong")


def classify(expected: str | None, agent_answer: str,
             clean_tool_result: str | None = None,
             faulted_tool_result: str | None = None) -> RecoveryVerdict:
    """Classify one faulted run against its clean baseline.

    - expected given: keyword match decides CORRECT vs wrong; wrong + confident
      (no hedging) = SILENT_FAILURE; wrong + hedging = DEGRADED.
    - expected None but clean/faulted tool results given: if the agent's answer
      echoes the FAULTED data and not the CLEAN data, without hedging, that is
      the signature SILENT_FAILURE. Hedging flips it to DEGRADED.
    - neither: UNVERIFIABLE.
    """
    if not agent_answer:
        return RecoveryVerdict.CRASHED
    low = agent_answer.lower()
    hedging = any(marker in low for marker in _HEDGING)

    if expected:
        # keyword match (words > 2 chars) AND every number must appear — short
        # numeric expectations like "8" are the common case for tool outputs
        tokens = [w.lower() for w in expected.split() if len(w) > 2]
        digits = re.findall(r"\d+", expected)
        words_ok = all(kw in low for kw in tokens)
        digits_ok = all(d in low for d in digits)
        if words_ok and digits_ok:
            return RecoveryVerdict.CORRECT_WITH_FAULTS
        return RecoveryVerdict.DEGRADED if hedging else RecoveryVerdict.SILENT_FAILURE

    if clean_tool_result and faulted_tool_result and clean_tool_result != faulted_tool_result:
        clean_in = clean_tool_result.strip()[:60].lower() in low
        faulted_in = faulted_tool_result.strip()[:60].lower() in low
        if faulted_in and not clean_in:
            return RecoveryVerdict.SILENT_FAILURE
        if clean_in and not faulted_in:
            return RecoveryVerdict.CORRECT_WITH_FAULTS
        return RecoveryVerdict.DEGRADED if hedging else RecoveryVerdict.UNVERIFIABLE

    return RecoveryVerdict.UNVERIFIABLE if not hedging else RecoveryVerdict.DEGRADED
