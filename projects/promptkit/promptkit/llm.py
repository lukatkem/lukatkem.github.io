"""BaseLLM protocol + a deterministic MockLLM — the offline test double."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str


class BaseLLM:
    def complete(self, messages: list) -> LLMResponse:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """Replies with canned strings in order; records every request."""

    def __init__(self, replies: list[str]):
        if not replies:
            raise ValueError("MockLLM needs at least one reply")
        self._replies = list(replies)
        self.calls: list = []

    def complete(self, messages: list) -> LLMResponse:
        self.calls.append(messages)
        if not self._replies:
            raise RuntimeError("MockLLM script exhausted — more calls than replies")
        return LLMResponse(self._replies.pop(0))
