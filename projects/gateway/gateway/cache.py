"""Thread-safe exact-match response cache with TTL and maxsize eviction."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

__all__ = ["TTLCache", "cache_key"]

Clock = Callable[[], float]


def cache_key(model: str, messages: list[dict[str, str]]) -> str:
    """SHA-256 of the exact request: the model plus canonicalized messages JSON."""
    canonical = model + json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TTLCache:
    """A small, thread-safe cache with per-entry TTL and maxsize eviction.

    Reads of expired entries count as misses. On ``put`` with a full table,
    expired entries are evicted first, then the oldest entries (FIFO). Stored
    and returned values are deep-copied so callers cannot mutate shared state.
    """

    def __init__(self, *, maxsize: int = 128, ttl_seconds: float = 300.0, clock: Clock | None = None) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._clock: Clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Return the cached value, or None on miss/expiry (counted)."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            stored_at, value = entry
            if now - stored_at > self._ttl:
                del self._entries[key]
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(value)

    def put(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``, evicting if over ``maxsize``."""
        now = self._clock()
        with self._lock:
            if key not in self._entries and len(self._entries) >= self._maxsize:
                self._evict_one(now)
            self._entries[key] = (now, copy.deepcopy(value))
            self._entries.move_to_end(key)

    def clear(self) -> int:
        """Drop every entry; return how many were removed."""
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            return removed

    def stats(self) -> dict[str, Any]:
        """JSON-safe counters for ``GET /admin/status``."""
        with self._lock:
            lookups = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._entries),
                "hit_rate": round(self._hits / lookups, 4) if lookups else 0.0,
            }

    def _evict_one(self, now: float) -> None:
        for key, (stored_at, _) in self._entries.items():
            if now - stored_at > self._ttl:
                del self._entries[key]
                return
        self._entries.popitem(last=False)  # nothing expired: drop the oldest
