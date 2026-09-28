"""The fault taxonomy — every failure mode a production tool layer produces."""
from __future__ import annotations

import enum


class FaultType(enum.Enum):
    TIMEOUT = "timeout"                       # handler hangs → the agent's timeout fires
    CONNECTION_ERROR = "connection_error"     # upstream refused / unreachable
    MALFORMED_JSON = "malformed_json"         # returns broken JSON where JSON is expected
    TRUNCATED = "truncated"                   # response cut mid-sentence / mid-JSON
    WRONG_BUT_PLAUSIBLE = "wrong_but_plausible"  # correct shape, WRONG data — the killer
    EMPTY = "empty"                           # 200 OK, empty body
    SLOW = "slow"                             # correct answer, deliberately delayed


class FaultProfile:
    """Deterministic fault schedule: tool → list of (FaultType, probability 0..1).

    No randomness module: a fixed-seed LCG decides injections, so runs are
    reproducible — the same seed always injects the same faults.
    """

    def __init__(self, assignments: dict[str, list] | None = None,
                 global_rate: float = 0.0, seed: int = 20260923):
        self.assignments = assignments or {}
        self.global_rate = global_rate
        self.seed = seed
        self._state = seed
        self.log: list[tuple[str, str, bool]] = []   # (tool, fault, injected)

    def _next(self) -> float:
        self._state = (self._state * 1103515245 + 12345) % (1 << 31)
        return self._state / (1 << 31)

    def pick(self, tool_name: str) -> FaultType | None:
        """Which fault applies to this call, if any. Logs the decision."""
        options = list(self.assignments.get(tool_name, []))
        if self.global_rate > 0 and self.global_rate < 1.0:
            options = options or [(ft, self.global_rate) for ft in FaultType]
        for fault_type, probability in options:
            ft = fault_type if isinstance(fault_type, FaultType) else FaultType(fault_type)
            roll = self._next()
            if roll < probability:
                self.log.append((tool_name, ft.value, True))
                return ft
            self.log.append((tool_name, ft.value, False))
        self.log.append((tool_name, "none", False))
        return None
