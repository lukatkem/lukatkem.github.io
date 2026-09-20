"""Per-key rate limiting (token bucket) and the USD budget ledger."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

__all__ = ["BudgetLedger", "RateLimiter", "TokenBucket"]

Clock = Callable[[], float]


class TokenBucket:
    """Classic token bucket: starts at ``capacity``, refills continuously."""

    def __init__(self, *, capacity: float, refill_per_second: float, clock: Clock | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = float(capacity)
        self._refill = float(refill_per_second)
        self._clock: Clock = clock or time.monotonic
        self._tokens = self._capacity
        self._last_refill = self._clock()

    def try_acquire(self, amount: float = 1.0) -> bool:
        """Refill for the elapsed time, then take ``amount`` tokens if available."""
        now = self._clock()
        elapsed = max(0.0, now - self._last_refill)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill)
        self._last_refill = now
        if self._tokens >= amount:
            self._tokens -= amount
            return True
        return False


class RateLimiter:
    """One token bucket per API key, created lazily."""

    def __init__(self, *, capacity: float, refill_per_second: float, clock: Clock | None = None) -> None:
        self._capacity = capacity
        self._refill = refill_per_second
        self._clock: Clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, key_id: str) -> bool:
        """Whether ``key_id`` may send another request right now."""
        with self._lock:
            bucket = self._buckets.get(key_id)
            if bucket is None:
                bucket = TokenBucket(capacity=self._capacity, refill_per_second=self._refill, clock=self._clock)
                self._buckets[key_id] = bucket
        return bucket.try_acquire()


class BudgetLedger:
    """Accumulates estimated spend and token usage per API key.

    ``budgets`` maps key id -> USD budget (``None`` = unlimited). A key is
    *exceeded* once its accumulated spend reaches its budget; the gateway
    checks this before routing a request and returns 402 when exceeded.
    """

    def __init__(self, budgets: Mapping[str, float | None] | None = None) -> None:
        self._lock = threading.Lock()
        self._budgets: dict[str, float | None] = dict(budgets or {})
        self._rows: dict[str, dict[str, float]] = {}

    def exceeded(self, key_id: str) -> bool:
        """Whether ``key_id`` has spent up to (or past) its budget."""
        budget = self._budgets.get(key_id)
        if budget is None:
            return False
        with self._lock:
            return self._row(key_id)["usd_spent"] >= budget

    def record(self, key_id: str, *, prompt_tokens: int, completion_tokens: int, usd: float) -> None:
        """Accumulate one successful (non-cached) response against ``key_id``."""
        with self._lock:
            row = self._row(key_id)
            row["requests"] += 1
            row["prompt_tokens"] += prompt_tokens
            row["completion_tokens"] += completion_tokens
            row["usd_spent"] += usd

    def usage(self, key_id: str) -> dict[str, Any]:
        """JSON-safe per-key usage for ``GET /admin/status``."""
        with self._lock:
            row = dict(self._row(key_id))
        budget = self._budgets.get(key_id)
        spent = row["usd_spent"]
        return {
            "requests": int(row["requests"]),
            "prompt_tokens": int(row["prompt_tokens"]),
            "completion_tokens": int(row["completion_tokens"]),
            "total_tokens": int(row["prompt_tokens"] + row["completion_tokens"]),
            "usd_spent": round(spent, 6),
            "usd_budget": budget,
            "budget_remaining": None if budget is None else round(max(0.0, budget - spent), 6),
        }

    def _row(self, key_id: str) -> dict[str, float]:
        return self._rows.setdefault(
            key_id,
            {"requests": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "usd_spent": 0.0},
        )
