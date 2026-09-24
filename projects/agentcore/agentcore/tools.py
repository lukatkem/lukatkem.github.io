"""Tools — the registry, strict argument validation, and the built-in examples.

A Tool is a typed contract: a name, a description, a flat parameter map
(param name -> one of {str, int, float, bool}), and the callable that
implements it. The Registry owns three jobs:

1. schema(name)  — a clean JSON-ish schema dict a model can be shown.
2. call(...)     — validate arguments *before* the handler ever runs.
3. the dangerous gate — a tool marked dangerous=True never executes unless
   a confirm callback returns True for it.

The built-in examples are deliberately small but real: an AST-walk
calculator (no eval), a text statistics tool, and a sandboxed file reader
that refuses every path it does not like.
"""
from __future__ import annotations

import ast
import operator
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "ToolError",
    "Tool",
    "Registry",
    "calculator_tool",
    "text_stats_tool",
    "read_note_tool",
]


class ToolError(Exception):
    """Raised for unknown tools, bad arguments, sandbox escapes, refusals,
    and any crash inside a tool handler — everything the agent loop can
    feed back to the model as an error message."""


# ---------------------------------------------------------------------------
# the Tool contract
# ---------------------------------------------------------------------------

#: parameter type annotation -> JSON-schema-ish type name
_JSON_TYPES: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass
class Tool:
    """A single tool the agent may call.

    parameters maps parameter name -> type annotation, restricted to
    {str, int, float, bool}. Every declared parameter is required.
    """

    name: str
    description: str
    parameters: Dict[str, type]
    handler: Callable[..., Any]
    dangerous: bool = False

    def schema(self) -> dict:
        """Clean JSON-ish schema: what a model would be shown to decide
        whether and how to call this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": _JSON_TYPES[t]} for p, t in self.parameters.items()},
                "required": sorted(self.parameters),
            },
            "dangerous": self.dangerous,
        }


# ---------------------------------------------------------------------------
# the Registry
# ---------------------------------------------------------------------------


class Registry:
    """A map of name -> Tool with strict validation at the call boundary."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    # -- registration --------------------------------------------------

    def register(self, tool: Tool) -> Tool:
        if not isinstance(tool, Tool):
            raise ToolError("register() expects a Tool")
        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ToolError("tool name must be a non-empty string")
        if tool.name in self._tools:
            raise ToolError(f"tool {tool.name!r} is already registered")
        if not callable(tool.handler):
            raise ToolError(f"tool {tool.name!r}: handler must be callable")
        for pname, ptype in tool.parameters.items():
            if ptype not in _JSON_TYPES:
                raise ToolError(
                    f"tool {tool.name!r}: parameter {pname!r} has type {ptype!r}; "
                    f"allowed types are str, int, float, bool"
                )
        self._tools[tool.name] = tool
        return tool

    @classmethod
    def with_builtins(cls, sandbox_dir: Optional[os.PathLike] = None) -> "Registry":
        """A registry preloaded with the example tools: calculator and
        text_stats always, plus the dangerous read_note when a sandbox
        directory is given."""
        reg = cls()
        reg.register(calculator_tool())
        reg.register(text_stats_tool())
        if sandbox_dir is not None:
            reg.register(read_note_tool(sandbox_dir))
        return reg

    # -- lookup ----------------------------------------------------------

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name!r}")
        return tool

    def schema(self, name: str) -> dict:
        return self.get(name).schema()

    def list(self) -> List[Tool]:
        """All registered tools, in registration order."""
        return list(self._tools.values())

    # -- the call boundary -------------------------------------------------

    def call(
        self,
        name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        confirm_dangerous: Optional[Callable[[str], bool]] = None,
    ) -> Any:
        """Validate arguments, then run the handler.

        Raises ToolError for: unknown tool, an unconfirmed dangerous tool,
        missing/extra arguments, wrong argument types, or any exception the
        handler itself raises. The agent loop feeds all of these back to
        the model as error messages — a validation failure is data, not a
        crash.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name!r}")

        if tool.dangerous and not (confirm_dangerous and confirm_dangerous(tool.name)):
            raise ToolError(
                f"dangerous tool {name!r} refused: no confirmation "
                f"(pass confirm_dangerous to allow it)"
            )

        kwargs = dict(kwargs or {})
        missing = sorted(set(tool.parameters) - set(kwargs))
        extra = sorted(set(kwargs) - set(tool.parameters))
        if missing:
            raise ToolError(f"{tool.name}: missing required argument(s): {', '.join(missing)}")
        if extra:
            raise ToolError(f"{tool.name}: unexpected argument(s): {', '.join(extra)}")
        for pname, ptype in tool.parameters.items():
            _validate_value(tool.name, pname, ptype, kwargs[pname])

        try:
            return tool.handler(**kwargs)
        except ToolError:
            raise
        except Exception as exc:  # a buggy tool is feedback, not a crash
            raise ToolError(f"{tool.name} crashed: {type(exc).__name__}: {exc}") from None


def _validate_value(tool_name: str, pname: str, ptype: type, value: Any) -> None:
    """One type check. bool is a subclass of int in Python, so bools are
    rejected where int/float is expected and int is accepted where float
    is expected (3 is a valid number)."""
    if ptype is bool:
        ok = isinstance(value, bool)
    elif ptype is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif ptype is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:  # str
        ok = isinstance(value, str)
    if not ok:
        want = _JSON_TYPES[ptype]
        raise ToolError(
            f"{tool_name}: argument {pname!r} must be {want}, got {type(value).__name__}"
        )


# ---------------------------------------------------------------------------
# built-in 1: calculator — a tiny AST walker, never eval()
# ---------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPR_LEN = 256
_MAX_EXPONENT = 4096  # blocks 9**9**9**9-style blowups


def _calc_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _calc_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(
                f"calculator: only plain numbers are allowed, got {type(node.value).__name__}"
            )
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ToolError(f"calculator: operator {type(node.op).__name__} is not allowed")
        left, right = _calc_node(node.left), _calc_node(node.right)
        if type(node.op) is ast.Pow and abs(right) > _MAX_EXPONENT:
            raise ToolError(f"calculator: exponent too large (|{right}| > {_MAX_EXPONENT})")
        try:
            result = op(left, right)
        except ZeroDivisionError:
            raise ToolError("calculator: division by zero") from None
        if isinstance(result, complex):
            raise ToolError("calculator: negative base with fractional exponent gives a complex number")
        return result
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ToolError(f"calculator: operator {type(node.op).__name__} is not allowed")
        return op(_calc_node(node.operand))
    raise ToolError(f"calculator: disallowed syntax: {type(node).__name__}")


def calculator_tool() -> Tool:
    """Safe arithmetic via ast.parse + a whitelist walk. Names, calls,
    attribute access, strings — everything that is not literal arithmetic —
    is refused before it is ever evaluated."""

    def _calculator(expression: str) -> Any:
        expr = expression.strip()
        if not expr:
            raise ToolError("calculator: empty expression")
        if len(expr) > _MAX_EXPR_LEN:
            raise ToolError(f"calculator: expression longer than {_MAX_EXPR_LEN} characters")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"calculator: invalid syntax: {exc.msg}") from None
        return _calc_node(tree)

    return Tool(
        name="calculator",
        description="Evaluate a pure arithmetic expression: + - * / // % ** and parentheses. "
        "No variables, no functions, no eval.",
        parameters={"expression": str},
        handler=_calculator,
    )


# ---------------------------------------------------------------------------
# built-in 2: text_stats
# ---------------------------------------------------------------------------


def text_stats_tool() -> Tool:
    def _text_stats(text: str) -> dict:
        lines = text.count("\n") + 1 if text else 0
        return {"words": len(text.split()), "chars": len(text), "lines": lines}

    return Tool(
        name="text_stats",
        description="Word, character and line counts of a given string.",
        parameters={"text": str},
        handler=_text_stats,
    )


# ---------------------------------------------------------------------------
# built-in 3: read_note — sandboxed file reader, dangerous by contract
# ---------------------------------------------------------------------------


def read_note_tool(sandbox_dir: os.PathLike) -> Tool:
    """Reads a file, but only from the sandbox directory.

    Every escape hatch is closed before the file is touched:
    * absolute paths (POSIX ``/etc/passwd`` and Windows ``C:\\x`` alike),
    * backslashes (Windows separators are never legitimate here, whatever
      the host OS is),
    * any ``..`` segment,
    * symlinks that resolve outside the sandbox (the final resolved path
      must still be inside it).
    All of the above raise ToolError. Marked dangerous=True: the registry
    refuses to run it without an explicit confirmation callback.
    """
    sandbox = Path(sandbox_dir).resolve()

    def _read_note(path: str) -> str:
        if not path.strip():
            raise ToolError("read_note: path must be a non-empty string")
        if "\\" in path or PureWindowsPath(path).is_absolute() or os.path.isabs(path):
            raise ToolError(f"read_note: absolute paths are not allowed: {path!r}")
        if any(part == ".." for part in path.split("/")):
            raise ToolError(f"read_note: '..' segments are not allowed (path traversal): {path!r}")
        target = (sandbox / path).resolve()
        try:
            target.relative_to(sandbox)  # belt and braces, catches symlink tricks
        except ValueError:
            raise ToolError(f"read_note: path escapes the sandbox: {path!r}") from None
        if not target.exists():
            raise ToolError(f"read_note: no such note: {path!r}")
        if not target.is_file():
            raise ToolError(f"read_note: not a file: {path!r}")
        try:
            return target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"read_note: cannot read {path!r}: {exc}") from None

    return Tool(
        name="read_note",
        description=f"Read a note file by relative path from the sandbox directory.",
        parameters={"path": str},
        handler=_read_note,
        dangerous=True,
    )
