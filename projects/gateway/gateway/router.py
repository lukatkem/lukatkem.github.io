"""Provider selection with failover: priority order, breaker gating, retries."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .breaker import CircuitBreaker
from .config import Provider
from .upstream import UpstreamClient, UpstreamError, UpstreamResult

__all__ = ["AllProvidersFailedError", "ProviderStats", "Router", "UnknownModelError"]

Sleep = Callable[[float], None]


class UnknownModelError(Exception):
    """No configured provider serves the requested model."""

    def __init__(self, model: str) -> None:
        super().__init__(f"no configured provider serves model '{model}'")
        self.model = model


class AllProvidersFailedError(Exception):
    """Every candidate provider failed (after retries) or was skipped."""

    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        super().__init__("all upstream providers failed")
        self.attempts = attempts


class ProviderStats:
    """Thread-safe per-provider call counters for the admin dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, dict[str, Any]] = {}

    def record_attempt(self, name: str) -> None:
        with self._lock:
            self._row(name)["requests"] += 1

    def record_failure(self, name: str) -> None:
        with self._lock:
            self._row(name)["failures"] += 1

    def record_success(self, name: str, latency_ms: float) -> None:
        with self._lock:
            self._row(name)["last_latency_ms"] = latency_ms

    def snapshot(self, name: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._row(name))

    def _row(self, name: str) -> dict[str, Any]:
        return self._rows.setdefault(name, {"requests": 0, "failures": 0, "last_latency_ms": None})


class Router:
    """Routes a chat request across candidate providers.

    Candidates are the providers serving the requested model, ordered by
    ``(priority, name)``. Each candidate gets ``max_retries + 1`` attempts with
    exponential backoff between attempts; the per-provider circuit breaker can
    cut a provider short (open = skipped instantly). The first successful
    attempt wins and ``served_by`` is recorded on the result.
    """

    def __init__(
        self,
        providers: Sequence[Provider],
        client: UpstreamClient,
        *,
        breakers: Mapping[str, CircuitBreaker] | None = None,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.25,
        sleep: Sleep | None = None,
    ) -> None:
        if not providers:
            raise ValueError("Router requires at least one provider")
        self._ordered = sorted(providers, key=lambda p: (p.priority, p.name))
        self._client = client
        self._breakers: dict[str, CircuitBreaker] = (
            dict(breakers) if breakers is not None else {p.name: CircuitBreaker(p.name) for p in providers}
        )
        self._max_retries = max_retries
        self._backoff_base = backoff_base_seconds
        self._sleep: Sleep = sleep or time.sleep
        self.stats = ProviderStats()

    @property
    def ordered_providers(self) -> list[Provider]:
        """All configured providers in priority order."""
        return list(self._ordered)

    def providers_for(self, model: str) -> list[Provider]:
        """All providers serving ``model``, in priority order."""
        return [provider for provider in self._ordered if provider.serves(model)]

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> UpstreamResult:
        """Try candidates in order until one answers; raise if all fail."""
        candidates = self.providers_for(model)
        if not candidates:
            raise UnknownModelError(model)
        attempts: list[dict[str, Any]] = []
        for provider in candidates:
            result = self._try_provider(
                provider,
                model,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                attempts=attempts,
            )
            if result is not None:
                return result
        raise AllProvidersFailedError(attempts)

    def _try_provider(
        self,
        provider: Provider,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None,
        temperature: float | None,
        attempts: list[dict[str, Any]],
    ) -> UpstreamResult | None:
        """Run the retry loop for one provider; None means it did not answer."""
        breaker = self._breakers[provider.name]
        for attempt in range(self._max_retries + 1):
            if not breaker.allow():
                if attempt == 0:
                    attempts.append({"provider": provider.name, "skipped": "circuit_open"})
                return None
            self.stats.record_attempt(provider.name)
            try:
                result = self._client.chat(
                    provider, model, messages, max_tokens=max_tokens, temperature=temperature
                )
            except UpstreamError as exc:
                attempts.append({"provider": provider.name, "attempt": attempt + 1, "error": str(exc)})
                breaker.record_failure()
                self.stats.record_failure(provider.name)
                if attempt < self._max_retries:
                    self._sleep(self._backoff_base * (2**attempt))
                continue
            breaker.record_success()
            self.stats.record_success(provider.name, result.latency_ms)
            result.served_by = provider.name
            return result
        return None
