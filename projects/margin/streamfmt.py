"""NDJSON streaming helpers — tiny, pure, and unit-tested.

The stream contract (one JSON object per line, \n-terminated):
  {"type": "meta",  "model": …, "citations": […]}   — retrieval done, badges
  {"type": "delta", "text": …}                       — next text fragment
  {"type": "final", "answer": …, "latency_ms": …, …} — trace id, quota, done
"""
from __future__ import annotations

import json


def ndjson_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def parse_events(buffer: str) -> tuple[list[dict], str]:
    """Parse complete events out of a buffer; return (events, remainder).

    The network may split lines anywhere — callers feed every chunk in and
    keep the remainder for the next round."""
    events: list[dict] = []
    *lines, remainder = buffer.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn line that isn't at the tail is corrupt; skip it
    return events, remainder
