"""End-to-end tests for the mcpserver stdio JSON-RPC surface.

Each test spawns the real server as a short-lived subprocess
(``sys.executable -m mcpserver.server``) with pipes on stdin/stdout, speaks
newline-delimited JSON-RPC 2.0 to it, and tears it down with
``terminate()`` + ``wait()`` in the fixture's ``finally`` block. A few
unit-level assertions exercise the tool handlers directly via the registry.
"""

from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # keeps the module importable via bare pytest
    sys.path.insert(0, str(PROJECT_ROOT))

import mcpserver  # noqa: E402
from mcpserver import tools as tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Subprocess harness
# ---------------------------------------------------------------------------


class ServerProcess:
    """A spawned stdio server with line-oriented JSON messaging helpers."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mcpserver.server"],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._responses: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()

    def _pump_stdout(self) -> None:
        """Push every response line onto the queue; push None on EOF."""
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self._responses.put(json.loads(line))
        self._responses.put(None)

    def send(self, message: Any) -> None:
        """Write one message; dicts are JSON-encoded, strings sent verbatim."""
        assert self.proc.stdin is not None
        payload = message if isinstance(message, str) else json.dumps(message)
        self.proc.stdin.write(payload + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout: float = 10.0) -> dict:
        """Block for the next response line (with timeout)."""
        try:
            response = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise AssertionError(f"no response within {timeout}s") from exc
        if response is None:
            raise AssertionError(
                "server closed stdout before responding\nstderr:\n" + self._read_stderr()
            )
        return response

    def request(self, message: dict, timeout: float = 10.0) -> dict:
        """Send one request and return its response."""
        self.send(message)
        return self.recv(timeout)

    def _read_stderr(self) -> str:
        if self.proc.stderr is None:
            return ""
        try:
            return self.proc.stderr.read()
        except (OSError, ValueError):
            return ""

    def close(self) -> None:
        """Shut the server down: close stdin, wait, terminate if it lingers."""
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        finally:
            for stream in (self.proc.stdout, self.proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass


@pytest.fixture()
def server() -> Iterator[ServerProcess]:
    """A fresh server subprocess per test, always reaped afterwards."""
    process = ServerProcess()
    try:
        yield process
    finally:
        process.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def initialize(
    server: ServerProcess,
    protocol_version: str = "2025-06-18",
    request_id: int = 1,
) -> dict:
    """Perform the MCP handshake: initialize + notifications/initialized."""
    response = server.request(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "mcpserver-tests", "version": "0.0.0"},
            },
        }
    )
    server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    return response


def call_tool(
    server: ServerProcess, name: str, arguments: dict, request_id: int = 100
) -> dict:
    """Invoke a tool through tools/call and return the response envelope."""
    return server.request(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )


def tool_text(response: dict) -> str:
    """Extract the text of the first content block of a successful tool call."""
    result = response["result"]
    assert result["isError"] is False, result
    return result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------

# ~220 words of natural prose; the near-copy below swaps exactly one word,
# which keeps ~95% of the 5-gram shingles and an estimated Jaccard ~0.95.
CORPUS_A = (
    "The harbor market opens at dawn when fishing boats return with silver "
    "catches and crates of ice. Vendors arrange lemons, olives, and warm "
    "bread along the stone quay while gulls circle above the awnings. "
    "A violinist plays near the lighthouse while children chase each other "
    "between the carts and barrels of the fishmongers. By noon the crowd "
    "thins and the quadrangle fills with students eating plums and arguing "
    "about poetry. Later, painters set easels facing the bridge and sketch "
    "the amber sunset glow over the water. Night brings lanterns, quiet "
    "conversations, and the scent of salt and cedar drifting through the "
    "alleys. This rhythm repeats each week across every season without fail "
    "or complaint from anyone involved. Merchants wrap coins in linen, tally "
    "ledgers by candlelight, and argue softly about the price of rope. "
    "Somewhere a dog barks at the tide, and the ferryman counts passengers "
    "twice before casting off toward the far pier. The baker wakes first, "
    "firing the oven while stars still hang above the rooftops and chimneys. "
    "Rain arrives sideways in January, yet the market continues under "
    "oilcloth and stubborn umbrellas until the storm passes eastward. In "
    "spring the stalls overflow with peas, mint, and cut flowers bundled in "
    "yesterday's newspaper. Tourists photograph the scales, the nets, and "
    "the old harbor clock that has run slow for decades without repair."
)

CORPUS_B = CORPUS_A.replace("violinist", "cellist")

CORPUS_C = (
    "Reinforcement learning agents struggle with long horizon credit "
    "assignment, sparse rewards, and exploration bonuses. Researchers "
    "mitigate this with replay buffers, curriculum design, and reward "
    "shaping across many training environments."
)

SYMBOL_SOUP = "!@#$%^&*()_+-=[]{}|;:,.<>?/~ " * 10


# ---------------------------------------------------------------------------
# Protocol tests (over stdio)
# ---------------------------------------------------------------------------


def test_initialize_returns_server_info_and_capabilities(server):
    response = initialize(server)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    result = response["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"] == {"name": "mcpserver", "version": mcpserver.__version__}
    assert result["capabilities"] == {"tools": {}}


def test_initialize_echoes_supported_version_and_falls_back_otherwise(server):
    echoed = initialize(server, protocol_version="2024-11-05", request_id=2)
    assert echoed["result"]["protocolVersion"] == "2024-11-05"
    fallback = initialize(server, protocol_version="1999-01-01", request_id=3)
    assert fallback["result"]["protocolVersion"] == "2025-06-18"


def test_tools_list_catalog(server):
    initialize(server)
    response = server.request({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
    listed = response["result"]["tools"]
    assert sorted(tool["name"] for tool in listed) == [
        "corpus_stats",
        "dedupe_corpus",
        "quality_score",
        "scrub_pii",
    ]
    for tool in listed:
        assert tool["description"], tool["name"]
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert schema["properties"], tool["name"]
        assert schema["required"], tool["name"]


def test_scrub_pii_redacts_email_jwt_and_digits_but_keeps_dates(server):
    initialize(server)
    text = (
        "Contact sam@x.io before 2026-09-20, or call 555-123-4567 about card "
        "1234567890123456 and key sk-abc123def456ghi789jkl; token: "
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    response = call_tool(server, "scrub_pii", {"text": text}, request_id=5)
    content = tool_text(response)
    assert "sam@x.io" not in content
    assert "sk-abc123def456ghi789jkl" not in content
    assert "eyJhbGciOiJIUzI1NiJ9" not in content
    assert "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c" not in content
    assert "555-123-4567" not in content
    assert "1234567890123456" not in content
    assert "2026-09-20" in content  # dates survive scrubbing
    assert "emails=1" in content
    assert "phones=1" in content
    assert "long_digit_runs=1" in content
    assert "api_keys=1" in content
    assert "jwts=1" in content


def test_dedupe_corpus_reports_one_duplicate_cluster(server):
    initialize(server)
    response = call_tool(
        server,
        "dedupe_corpus",
        {"texts": [CORPUS_A, CORPUS_B, CORPUS_C], "threshold": 0.8},
        request_id=6,
    )
    content = tool_text(response)
    assert "3 documents → 2 unique (1 duplicate cluster removed)" in content
    assert "docs [0, 1]" in content
    assert "representative: doc 0" in content


def test_quality_score_rejects_symbol_soup(server):
    initialize(server)
    response = call_tool(server, "quality_score", {"text": SYMBOL_SOUP}, request_id=7)
    content = tool_text(response)
    assert content.startswith("score 0.00 · FAIL")
    assert "symbol-ratio" in content


def test_corpus_stats_reports_counts_md5_and_top_words(server):
    initialize(server)
    response = call_tool(
        server, "corpus_stats", {"text": "hello world hello"}, request_id=8
    )
    content = tool_text(response)
    digest = hashlib.md5(b"hello world hello").hexdigest()
    assert "chars: 17" in content
    assert "words: 3" in content
    assert "unique-words: 2" in content
    assert "lines: 1" in content
    assert f"md5: {digest}" in content
    assert "hello (2), world (1)" in content


def test_unknown_method_returns_method_not_found(server):
    initialize(server)
    response = server.request({"jsonrpc": "2.0", "id": 9, "method": "resources/list"})
    assert response["error"]["code"] == -32601
    assert response["id"] == 9


def test_malformed_line_yields_parse_error_and_server_survives(server):
    server.send("{not valid json")
    parse_error = server.recv()
    assert parse_error["id"] is None
    assert parse_error["error"]["code"] == -32700
    response = server.request({"jsonrpc": "2.0", "id": 10, "method": "ping"})
    assert response["result"] == {}


def test_missing_required_argument_is_tool_error_not_protocol_error(server):
    initialize(server)
    response = call_tool(server, "corpus_stats", {}, request_id=11)
    assert "error" not in response  # not a JSON-RPC error
    result = response["result"]
    assert result["isError"] is True
    assert "text" in result["content"][0]["text"]


def test_notification_produces_no_reply(server):
    initialize(server)
    server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    server.send({"jsonrpc": "2.0", "id": 12, "method": "ping"})
    following = server.recv()  # the very next line must be the ping response
    assert following["id"] == 12
    assert following["result"] == {}


# ---------------------------------------------------------------------------
# Unit-level checks against the tool registry (no subprocess)
# ---------------------------------------------------------------------------


def test_scrub_pii_preserves_dates_and_counts_digit_runs():
    scrub = tool_registry.TOOL_MAP["scrub_pii"].handler
    report = scrub({"text": "Moved from 2026-09-20 to 03/10/2026, ref 1234567890."})
    assert "2026-09-20" in report
    assert "03/10/2026" in report
    assert "1234567890" not in report
    assert "long_digit_runs=1" in report


def test_quality_score_passes_healthy_text():
    score = tool_registry.TOOL_MAP["quality_score"].handler
    report = score({"text": CORPUS_A})
    assert report.startswith("score 1.00 · PASS · reasons: none")


def test_dedupe_is_deterministic_across_calls():
    dedupe = tool_registry.TOOL_MAP["dedupe_corpus"].handler
    first = dedupe({"texts": [CORPUS_A, CORPUS_B, CORPUS_C]})
    second = dedupe({"texts": [CORPUS_A, CORPUS_B, CORPUS_C]})
    assert first == second
    assert "1 duplicate cluster removed" in first
