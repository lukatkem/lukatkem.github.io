"""The agent loop — parse, validate, execute, feed back, repeat.

One ``run(user_message)`` is a bounded conversation with the model:

1. append the user message, ask the model;
2. parse the reply for fenced ```tool blocks;
3. tool calls present -> execute each through the registry (results and
   errors alike go back as tool messages) and loop;
4. no tool calls -> the clean text is the final answer, stop.

Every intermediate turn is recorded as a Step — reply, parsed calls, and
each call's outcome — so a run can be replayed and audited line by line.

Guards, in order of firing:
* ParseError  -> the model is shown its own malformed block and may retry;
* unknown tool -> the error goes back as a tool message, the model may
  correct itself;
* dangerous tool -> refused unless a confirm_dangerous callback says yes;
* max_steps   -> the loop stops with stopped=True and a reason, never a
  hang, never an infinite tool-calling spiral.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .llm import BaseLLM, Message, ToolCallRequest
from .parser import ParsedToolCall, ParseError, parse_tool_calls
from .tools import Registry, ToolError

__all__ = ["ToolResult", "Step", "AgentResult", "Agent"]


@dataclass
class ToolResult:
    """The outcome of one tool execution attempt, errors included."""

    name: str
    arguments: Dict[str, Any]
    ok: bool
    result: Any = None
    error: str = ""


@dataclass
class Step:
    """One model turn that issued tool calls — full transparency.

    reply is the raw model text (wire-format blocks included);
    tool_calls are the parsed calls; tool_results are their outcomes in
    the same order; error holds the parse failure for turns whose reply
    could not be parsed at all."""

    index: int
    reply: str
    tool_calls: List[ParsedToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    error: str = ""


@dataclass
class AgentResult:
    """The outcome of a run: the final answer, the full step log, and why
    the loop ended. A run that hit max_steps returns answer="" with
    stopped=True and a reason."""

    answer: str
    steps: List[Step] = field(default_factory=list)
    tool_calls_made: int = 0
    stopped: bool = False
    reason: str = ""


#: a confirmation callback receives the tool name and returns True to allow it.
ConfirmFn = Optional[Callable[[str], bool]]


class Agent:
    """A tool-using agent driven by any BaseLLM and a Tool registry."""

    def __init__(
        self,
        registry: Registry,
        llm: BaseLLM,
        max_steps: int = 8,
        system: str = "You are a helpful agent.",
        confirm_dangerous: ConfirmFn = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.registry = registry
        self.llm = llm
        self.max_steps = max_steps
        self.system = system
        #: None (the default) means dangerous tools are always refused.
        self.confirm_dangerous = confirm_dangerous

    # ------------------------------------------------------------------

    def run(self, user_message: str) -> AgentResult:
        schemas = [tool.schema() for tool in self.registry.list()]
        messages: List[Message] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": user_message})

        steps: List[Step] = []
        tool_calls_made = 0

        for index in range(self.max_steps):
            response = self.llm.complete(messages, schemas)

            # structured calls (typed clients) take precedence over the wire format
            if response.tool_calls:
                calls = [
                    ParsedToolCall(name=tc.name, arguments=dict(tc.arguments or {}), raw="")
                    for tc in response.tool_calls
                    if isinstance(tc, ToolCallRequest)
                ]
            else:
                try:
                    clean_text, calls = parse_tool_calls(response.text)
                except ParseError as exc:
                    steps.append(Step(index=index, reply=response.text, error=str(exc)))
                    messages.append({"role": "assistant", "content": response.text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your last reply could not be parsed: {exc} Re-issue each "
                                "tool call in a ```tool fenced block as one JSON object with "
                                "'name' and 'arguments', or reply with plain text only."
                            ),
                        }
                    )
                    continue

            if not calls:
                return AgentResult(
                    answer=clean_text if not response.tool_calls else response.text,
                    steps=steps,
                    tool_calls_made=tool_calls_made,
                )

            messages.append({"role": "assistant", "content": response.text})
            step = Step(index=index, reply=response.text, tool_calls=calls)
            for call in calls:
                outcome = self._execute(call)
                step.tool_results.append(outcome)
                tool_calls_made += 1
                payload = (
                    {"ok": True, "result": outcome.result}
                    if outcome.ok
                    else {"ok": False, "error": outcome.error}
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": json.dumps(payload, default=str),
                    }
                )
            steps.append(step)

        return AgentResult(
            answer="",
            steps=steps,
            tool_calls_made=tool_calls_made,
            stopped=True,
            reason=(
                f"max_steps ({self.max_steps}) reached while the model was still "
                "issuing tool calls"
            ),
        )

    # ------------------------------------------------------------------

    def _execute(self, call: ParsedToolCall) -> ToolResult:
        """One registry call, every failure mode folded into the result."""
        try:
            value = self.registry.call(
                call.name, call.arguments, confirm_dangerous=self.confirm_dangerous
            )
            return ToolResult(name=call.name, arguments=dict(call.arguments), ok=True, result=value)
        except ToolError as exc:
            return ToolResult(name=call.name, arguments=dict(call.arguments), ok=False, error=str(exc))
