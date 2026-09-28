"""FaultyRegistry — wraps any tool registry with the fault scheduler.

Composition, not monkeypatching: the wrapper exposes the same call surface
as agentcore's Registry (call/schema/list) and dispatches through the fault
scheduler before delegating. With no matching fault it is pure pass-through.
"""
from __future__ import annotations

import json

from .faults import FaultProfile, FaultType


class FaultyRegistry:
    def __init__(self, registry, profile: FaultProfile | None = None,
                 json_tools: tuple[str, ...] = ()):
        """json_tools: tool names whose healthy output is JSON — those get
        MALFORMED_JSON / TRUNCATED / WRONG_BUT_PLAUSIBLE treatments that make
        sense for string payloads."""
        self._inner = registry
        self.profile = profile
        self._json_tools = set(json_tools)
        for attr in ("register", "schema", "list"):
            if hasattr(registry, attr):
                setattr(self, attr, getattr(registry, attr))

    def call(self, name: str, kwargs: dict | None = None, **extra):
        kwargs = dict(kwargs or {})
        kwargs.update(extra)          # confirm_dangerous etc. pass through
        fault = self.profile.pick(name) if self.profile else None
        if fault is None:
            return self._inner.call(name, kwargs)

        if fault is FaultType.CONNECTION_ERROR:
            raise ConnectionError(f"faultline: injected connection error for {name!r}")
        if fault is FaultType.TIMEOUT:
            raise TimeoutError(f"faultline: injected timeout for {name!r}")
        if fault is FaultType.EMPTY:
            return ""

        result = self._inner.call(name, kwargs)   # healthy result to corrupt

        if fault is FaultType.MALFORMED_JSON:
            text = result if isinstance(result, str) else json.dumps(result)
            return text[:max(1, len(text) // 2)] + '"}}CORRUPTED'
        if fault is FaultType.TRUNCATED:
            text = result if isinstance(result, str) else json.dumps(result)
            return text[:max(1, len(text) * 2 // 5)]
        if fault is FaultType.WRONG_BUT_PLAUSIBLE:
            return self._wrong_but_plausible(result)
        if fault is FaultType.SLOW:
            return result                          # correct answer, zero injected harm
        return result

    @staticmethod
    def _wrong_but_plausible(result):
        """Correct shape, wrong substance: flip digits, swap a filename."""
        if isinstance(result, (int, float)):
            return result * 7 + 1 if result == 0 else result * -3
        if isinstance(result, dict):
            return {k: (v + 55 if isinstance(v, int) and not isinstance(v, bool) else v)
                    for k, v in result.items()}
        if isinstance(result, str):
            if any(ch.isdigit() for ch in result):
                return "".join(str((int(c) + 5) % 10) if c.isdigit() else c for c in result)
            return result.replace("ok", "failed").replace("success", "error") or "unknown error"
        return "wrong data"

    # transparency for the agent loop's introspection
    def __getattr__(self, attr):
        return getattr(self._inner, attr)
