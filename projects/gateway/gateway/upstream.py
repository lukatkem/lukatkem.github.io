"""OpenAI-compatible upstream client (chat completions) with normalized results."""

from __future__ import annotations

import os
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import httpx

from .config import Provider

__all__ = ["UpstreamClient", "UpstreamError", "UpstreamResult", "estimate_tokens"]

# Documented cost model: tokens ~= len(text) // 4.
_CHARS_PER_TOKEN = 4

Clock = Callable[[], float]


class UpstreamError(Exception):
    """An upstream call failed. Messages are safe to surface (no credentials)."""


@dataclass
class UpstreamResult:
    """Normalized chat-completions result from one upstream call."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    served_by: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    """Estimate a token count from text length (the documented cost model)."""
    return len(text) // _CHARS_PER_TOKEN


class UpstreamClient:
    """Minimal OpenAI-compatible ``POST {base_url}/chat/completions`` client.

    ``transport`` is injectable (``httpx.MockTransport`` in tests). Provider API
    keys are resolved from ``env`` (default ``os.environ``) *at call time* via
    the provider's ``api_key_env`` variable name, and are never included in
    error messages.
    """

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_seconds: float = 30.0,
        env: Mapping[str, str] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._client = httpx.Client(transport=transport, timeout=timeout_seconds)
        self._finalizer = weakref.finalize(self, self._client.close)
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._clock: Clock = clock or time.monotonic

    def close(self) -> None:
        """Release the underlying HTTP connection pool (idempotent)."""
        self._finalizer()

    def chat(
        self,
        provider: Provider,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> UpstreamResult:
        """Call one provider; return a normalized result or raise :class:`UpstreamError`."""
        started = self._clock()
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        headers: dict[str, str] = {}
        api_key = self._env.get(provider.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = provider.base_url.rstrip("/") + "/chat/completions"
        try:
            response = self._client.post(url, json=payload, headers=headers)
        except httpx.HTTPError:
            raise UpstreamError(f"{provider.name}: upstream connection failed") from None

        latency_ms = round((self._clock() - started) * 1000.0, 3)
        if response.status_code >= 400:
            # Only the status code is surfaced; response bodies may echo secrets.
            raise UpstreamError(f"{provider.name}: upstream returned HTTP {response.status_code}")
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise UpstreamError(f"{provider.name}: malformed response from upstream") from None

        prompt_tokens, completion_tokens = self._usage(data, messages, str(text))
        return UpstreamResult(
            text=str(text),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _usage(data: Any, messages: list[dict[str, str]], text: str) -> tuple[int, int]:
        """Prefer provider-reported usage; fall back to the length heuristic."""
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            try:
                return int(usage["prompt_tokens"]), int(usage["completion_tokens"])
            except (KeyError, TypeError, ValueError):
                pass
        prompt = sum(estimate_tokens(str(message.get("content", ""))) for message in messages)
        return prompt, estimate_tokens(text)
