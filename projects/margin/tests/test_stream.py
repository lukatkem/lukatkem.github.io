"""Tests for the NDJSON stream contract used by /api/chat/stream."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root → `margin` package

from margin.streamfmt import ndjson_line, parse_events  # noqa: E402


def test_ndjson_roundtrip_survives_torn_packets():
    lines = (
        ndjson_line({"type": "meta", "model": "m", "citations": [{"n": 1}]})
        + ndjson_line({"type": "delta", "text": "he"})
        + ndjson_line({"type": "delta", "text": "llo"})
        + ndjson_line({"type": "final", "answer": "hello"})
    )
    # the network splits mid-line — the parser must keep the remainder
    a, rem = parse_events(lines[: len(lines) // 2])
    b, rem2 = parse_events(rem + lines[len(lines) // 2 :])
    events = a + b
    assert [e["type"] for e in events] == ["meta", "delta", "delta", "final"]
    assert rem2 == ""


def test_parse_skips_blank_and_bad_lines():
    events, rem = parse_events("\n{bad json}\n" + ndjson_line({"type": "delta", "text": "x"}) + "\n\ntail-without-newline")
    assert len(events) == 1 and events[0]["text"] == "x"
    assert rem == "tail-without-newline"


def test_unicode_is_not_escaped_in_stream():
    line = ndjson_line({"type": "delta", "text": "გამარჯობა"})
    assert "გამარჯობა" in line, "stream must carry unicode as-is (readable in curl)"
