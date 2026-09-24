"""The wire format — how a model's plain text carries tool calls.

The model emits tool calls as fenced blocks inside its normal reply:

    ```tool
    {"name": "calculator", "arguments": {"expression": "2+2*3"}}
    ```

Rules of the format:

* the opening fence is exactly three backticks + ``tool`` + a newline;
* the body is a single JSON object with a string ``name`` and an object
  ``arguments`` (``arguments`` may be omitted for zero-parameter tools);
* a reply may contain any number of blocks, mixed freely with prose;
* everything outside blocks is the model's visible text.

``parse_tool_calls`` returns the prose with every block stripped, plus the
calls in order. Anything malformed inside a block — broken JSON, a
non-object body, a missing name — raises ``ParseError`` with the reason,
so the agent can feed it back to the model instead of crashing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = ["ParseError", "ParsedToolCall", "parse_tool_calls"]


class ParseError(Exception):
    """A ```tool block was present but could not be parsed. Carries the
    reason so the agent can show the model its own mistake."""


@dataclass
class ParsedToolCall:
    """One parsed call: which tool, with what arguments, and the raw
    fenced block it came from (kept for the transparent step log)."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


#: opener: ```tool + whitespace + newline; body non-greedy; closer: ```
_FENCE = re.compile(r"```tool[ \t]*(?:\r?\n)(.*?)(?:\r?\n)?[ \t]*```", re.DOTALL)
_BLANK_RUN = re.compile(r"\n{3,}")


def parse_tool_calls(text: str) -> Tuple[str, List[ParsedToolCall]]:
    """Split a model reply into (clean_text, tool_calls).

    * clean_text — the reply with every ```tool block removed and blank
      runs collapsed;
    * tool_calls — every block, in order.

    Raises ParseError for malformed JSON inside a block, a non-object
    body, a missing/non-string name, non-object arguments, or an
    unterminated block (an opener with no closing fence).
    """
    text = text or ""
    calls: List[ParsedToolCall] = []

    def _capture(match: "re.Match[str]") -> str:
        raw, body = match.group(0), match.group(1)
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON in tool block: {exc}") from None
        if not isinstance(obj, dict):
            raise ParseError(
                f"tool block must be a JSON object, got {type(obj).__name__}"
            )
        name = obj.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ParseError("tool block is missing a string 'name'")
        arguments = obj.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ParseError(
                f"'arguments' must be a JSON object, got {type(arguments).__name__}"
            )
        calls.append(ParsedToolCall(name=name, arguments=arguments, raw=raw))
        return ""  # the block vanishes from the clean text

    clean = _FENCE.sub(_capture, text)
    if "```tool" in clean:
        raise ParseError("unterminated ```tool block (missing closing ```)")
    clean = _BLANK_RUN.sub("\n\n", clean).strip()
    return clean, calls
