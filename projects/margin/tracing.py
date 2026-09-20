"""Tracing + cost ledger: every route call lands in the traces table.

Local models cost $0 in API spend; we still record token estimates so the
dashboard shows latency/token/cost per run — the observability half of the
LLMOps story.
"""
from __future__ import annotations

import json
import time

from .store import connect, now


def record_trace(
    route: str,
    query: str,
    latency_ms: float,
    hits: list,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    user_id: int | None,
    status: str = "ok",
) -> int:
    conn = connect()
    # cost model: local ollama = $0; extractive = $0. Placeholder API models
    # would price at their per-1k rates here.
    cost_usd = 0.0
    cur = conn.execute(
        """INSERT INTO traces(ts,user_id,route,query,latency_ms,retrieval_json,
                             model,prompt_tokens,completion_tokens,cost_usd,status)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (
            now(),
            user_id,
            route,
            query[:2000],
            round(latency_ms, 1),
            json.dumps(
                [
                    {"doc": h.doc, "heading": h.heading, "score": h.score}
                    for h in hits
                ]
            ),
            model,
            prompt_tokens,
            completion_tokens,
            cost_usd,
            status,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def stats(limit: int = 200) -> dict:
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM traces ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    lat = [r["latency_ms"] for r in rows if r["latency_ms"]]
    lat.sort()
    pct = lambda p: lat[int(len(lat) * p)] if lat else 0  # noqa: E731
    return {
        "count": len(rows),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "recent": [dict(r) for r in rows],
    }
