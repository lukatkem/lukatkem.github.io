"""Per-provider circuit breaker: CLOSED -> OPEN -> (cooldown) -> HALF_OPEN -> CLOSED."""

from __future__ import annotations

import enum
import threading
import time
from typing import Any, Callable

__all__ = ["CircuitBreaker", "State"]

Clock = Callable[[], float]


class State(str, enum.Enum):
    """Breaker states (values are the JSON strings used by the dashboard)."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker with an injectable clock.

    Semantics:

    * CLOSED: calls allowed; ``failure_threshold`` consecutive failures trip it OPEN.
    * OPEN: calls refused until ``cooldown_seconds`` have elapsed since it opened.
    * HALF_OPEN: a probe call is allowed; one success closes the breaker, one
      failure re-opens it (restarting the cooldown).
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        clock: Clock | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock: Clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state = State.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        """Whether a call may proceed right now (may transition OPEN -> HALF_OPEN)."""
        with self._lock:
            if self._state is State.CLOSED:
                return True
            if self._state is State.OPEN:
                opened_at = self._opened_at or 0.0
                if self._clock() - opened_at >= self._cooldown_seconds:
                    self._state = State.HALF_OPEN
                    return True
                return False
            return True  # HALF_OPEN: probing is allowed

    def record_success(self) -> None:
        """A call succeeded: reset the counter and close the breaker."""
        with self._lock:
            self._state = State.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        """A call failed: count it and trip OPEN at the threshold (or from HALF_OPEN)."""
        with self._lock:
            self._consecutive_failures += 1
            if self._state is State.HALF_OPEN or self._consecutive_failures >= self._failure_threshold:
                self._state = State.OPEN
                self._opened_at = self._clock()

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe state for ``GET /admin/status`` and the dashboard."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self._failure_threshold,
                "cooldown_seconds": self._cooldown_seconds,
            }
