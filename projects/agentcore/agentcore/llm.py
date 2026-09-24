"""LLM protocol — one method, offline by default.

The agent talks to exactly one surface: ``complete(messages, tools)``.
``BaseLLM`` is the abstract contract; ``MockLLM`` is the offline
implementation that makes the entire loop testable — it replays a SCRIPT
of canned replies in order, each one written in the same wire format a
real model would produce (prose plus fenced `````tool```` blocks, or
structured ``ToolCallRequest`` objects for typed clients).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Union

__all__ = ["Message", "ToolCallRequest", "LLMResponse", "BaseLLM", "MockLLM"]

#: one conversation turn, e.g. {"role": "user", "content": "hi"}.
#: roles: "system" | "user" | "assistant" | "tool".
Message = Dict[str, str]


@dataclass
class ToolCallRequest:
    """A structured tool call — what a typed client (e.g. an
    OpenAI-compatible subclass) would hand the agent directly."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """What a model turn returns.

    ``text`` always carries the raw reply (which may contain fenced
    ```tool blocks for the agent to parse). ``tool_calls`` is the typed
    alternative: when non-empty the agent skips parsing and executes these
    directly. MockLLM leaves it empty; a real-client subclass would not.
    """

    text: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)


class BaseLLM(ABC):
    """The whole LLM protocol: one method."""

    @abstractmethod
    def complete(self, messages: List[Message], tools: List[dict]) -> LLMResponse:
        """Turn a conversation (list of message dicts) plus tool schemas
        into the model's next reply."""


class MockLLM(BaseLLM):
    """A scripted model — the reason the whole runtime needs no network.

    The script is a list of canned replies consumed strictly in order.
    Each entry is either raw wire-format text (prose with zero or more
    fenced ```tool blocks — parsed by the agent, exactly like a real
    model's output) or a ready-made LLMResponse. When the script runs dry
    while the agent is still asking, complete() raises loudly instead of
    looping silently.
    """

    def __init__(self, script: Sequence[Union[str, LLMResponse]]) -> None:
        self._script: List[LLMResponse] = [
            entry if isinstance(entry, LLMResponse) else LLMResponse(text=entry)
            for entry in script
        ]
        self._cursor = 0
        #: every complete() call, recorded: {"messages": [...], "tools": [...]}.
        #: Tests and the demo use this to prove what the model actually saw.
        self.turns: List[Dict[str, Any]] = []

    def complete(self, messages: List[Message], tools: List[dict]) -> LLMResponse:
        self.turns.append(
            {"messages": [dict(m) for m in messages], "tools": [dict(t) for t in tools]}
        )
        if self._cursor >= len(self._script):
            raise RuntimeError(
                "MockLLM script exhausted: the agent is still asking after the last "
                f"scripted reply ({len(self.turns) - 1} turns consumed)"
            )
        response = self._script[self._cursor]
        self._cursor += 1
        return response
