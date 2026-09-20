"""Hand-written JSON-RPC 2.0 + MCP protocol layer.

Implements the Model Context Protocol surface of this server directly on
JSON-RPC 2.0 — no official SDK:

* transport — newline-delimited JSON on stdin/stdout; every response line is
  flushed immediately and stdout carries nothing but protocol messages,
* methods — ``initialize``, ``notifications/initialized`` (notification, no
  reply), ``ping``, ``tools/list``, ``tools/call``,
* errors — standard JSON-RPC codes for protocol problems; tool failures
  follow the MCP convention of ``isError: true`` results instead.

Unknown methods get ``-32601``; undecodable lines get ``-32700`` and the loop
keeps serving — a bad line must not kill the session.
"""

from __future__ import annotations

import json
import os
import sys
from typing import IO, Any

from . import __version__
from .tools import TOOL_MAP, ToolArgumentError

__all__ = [
    "McpProtocol",
    "RpcError",
    "serve",
    "PARSE_ERROR",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    "PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "SERVER_NAME",
]

# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP server metadata
PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2024-11-05")
SERVER_NAME = "mcpserver"


class RpcError(Exception):
    """A JSON-RPC protocol error that must be reported to the client."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _result_response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error_response(
    request_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _debug(message: str) -> None:
    """Write a diagnostic line to stderr — never to stdout.

    Enabled by setting ``MCP_DEBUG=1``. MCP clients read stdout as protocol
    traffic, so even debug output must stay away from it.
    """
    if os.environ.get("MCP_DEBUG"):
        print(f"[mcpserver] {message}", file=sys.stderr)


class McpProtocol:
    """JSON-RPC 2.0 dispatcher for the MCP method surface.

    Pure logic: decoded messages in, response dicts out (``None`` for
    notifications), so the stdio loop in :func:`serve` stays trivial.
    """

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Handle one decoded JSON value; return a response dict, or None."""
        if not isinstance(message, dict):
            return _error_response(None, INVALID_REQUEST, "request must be a JSON object")

        request_id = message.get("id")
        is_notification = "id" not in message
        envelope = message.get("jsonrpc")
        if envelope is not None and envelope != "2.0":
            return self._reject(
                is_notification, request_id, INVALID_REQUEST, "'jsonrpc' must be \"2.0\""
            )

        method = message.get("method")
        if not isinstance(method, str):
            return self._reject(
                is_notification,
                request_id,
                INVALID_REQUEST,
                "request is missing a string 'method'",
            )

        params = message.get("params") or {}
        if not isinstance(params, dict):
            # Every method this server exposes takes object params.
            return self._reject(
                is_notification, request_id, INVALID_PARAMS, "'params' must be an object"
            )

        try:
            result = self._dispatch(method, params)
        except RpcError as exc:
            return self._reject(is_notification, request_id, exc.code, exc.message, exc.data)
        except ToolArgumentError as exc:
            # MCP convention: tool validation failures are isError results.
            if is_notification:
                return None
            return _result_response(request_id, self._tool_error(str(exc)))
        except Exception as exc:  # defensive: a tool bug must not kill the loop
            _debug(f"internal error handling {method!r}: {exc!r}")
            return self._reject(
                is_notification, request_id, INTERNAL_ERROR, f"internal error: {exc}"
            )

        return None if is_notification else _result_response(request_id, result)

    @staticmethod
    def _reject(
        is_notification: bool,
        request_id: Any,
        code: int,
        message: str,
        data: Any = None,
    ) -> dict[str, Any] | None:
        """Build an error response; notifications are never answered."""
        if is_notification:
            return None
        return _error_response(request_id, code, message, data)

    # -- MCP methods --------------------------------------------------------

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return self._initialize(params)
        if method == "notifications/initialized":
            return {}  # acknowledged; notifications get no reply
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [tool.to_dict() for tool in TOOL_MAP.values()]}
        if method == "tools/call":
            return self._tools_call(params)
        raise RpcError(METHOD_NOT_FOUND, f"method not found: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Version negotiation: echo a supported client version, else ours."""
        requested = params.get("protocolVersion")
        version = (
            requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
        )
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
        }

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise RpcError(INVALID_PARAMS, "tools/call requires a string 'name'")
        tool = TOOL_MAP.get(name)
        if tool is None:
            raise RpcError(INVALID_PARAMS, f"unknown tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ToolArgumentError(
                f"tool '{name}' expects 'arguments' to be an object"
            )
        text = tool.handler(arguments)
        return {"content": [{"type": "text", "text": text}], "isError": False}

    @staticmethod
    def _tool_error(message: str) -> dict[str, Any]:
        """MCP convention: tool failures are results with ``isError: true``."""
        return {
            "content": [{"type": "text", "text": f"error: {message}"}],
            "isError": True,
        }


def serve(stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
    """Run the newline-delimited JSON-RPC loop until EOF.

    Reads one JSON message per line from *stdin* and writes one response per
    line to *stdout*, flushed immediately. A line that is not valid JSON
    produces a ``-32700`` parse error and the loop continues; EOF (the client
    closed the stream) ends the server gracefully.
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    protocol = McpProtocol()
    _debug("stdio server ready")
    while True:
        try:
            line = stdin.readline()
        except UnicodeDecodeError:
            response: dict[str, Any] | None = _error_response(
                None, PARSE_ERROR, "parse error: undecodable input"
            )
        else:
            if line == "":  # EOF: the client closed the stream
                break
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                response = _error_response(None, PARSE_ERROR, f"parse error: {exc.msg}")
            else:
                response = protocol.handle_message(message)
        if response is None:
            continue
        try:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
        except BrokenPipeError:  # client stopped reading; shut down quietly
            break
    _debug("stdio server stopped (EOF)")
