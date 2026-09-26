"""Answer generation with forced citations, via local Ollama.

Contract with the model: every factual claim must carry a [n] marker mapping
to the numbered context passages. If no Ollama model is reachable we degrade
to an extractive answer built from the top passages — never silently
hallucinate. Two transports: blocking `answer()` and token-streaming
`stream_chunks()` used by /api/chat/stream.
"""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from .config import CHAT_MODEL, OLLAMA_BASE, OLLAMA_TIMEOUT
from .retrieve import Hit

SYSTEM = (
    "You are Margin, a precise assistant that answers ONLY from the provided "
    "rulebook passages. Rules: (1) Cite every claim with a bracketed passage "
    "number formatted exactly like [1] or [2]. (2) If the passages don't "
    "contain the answer, say exactly: I don't find that in the rulebook. "
    "(3) Be concise: lead with the answer, then the details. (4) Quote exact "
    "numbers (%, $, times) when present."
)


def ollama_available() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _chat_payload(messages: list[dict], stream: bool) -> dict:
    return {"model": CHAT_MODEL, "messages": messages,
            "stream": stream, "options": {"temperature": 0.1}}


def _chat(messages: list[dict]) -> str:
    r = httpx.post(
        f"{OLLAMA_BASE}/api/chat",
        json=_chat_payload(messages, stream=False),
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Passages:\n\n{build_context(hits)}\n\nQuestion: {question}"},
    ]


def stream_chunks(messages: list[dict]) -> Iterator[str]:
    """Yield content deltas from Ollama's NDJSON streaming endpoint."""
    with httpx.stream(
        "POST",
        f"{OLLAMA_BASE}/api/chat",
        json=_chat_payload(messages, stream=True),
        timeout=OLLAMA_TIMEOUT,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            piece = (obj.get("message") or {}).get("content") or ""
            if piece:
                yield piece
            if obj.get("done"):
                return


def build_context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] ({h.doc} · {h.heading})\n{h.text}")
    return "\n\n".join(parts)


def _extractive(question: str, hits: list[Hit]) -> str:
    """No-LLM fallback: return the most relevant passages verbatim."""
    lines = ["I couldn't reach a local LLM, so here are the most relevant rulebook passages:"]
    for i, h in enumerate(hits[:3], 1):
        snippet = h.text[:400].replace("\n", " ")
        label = " — ".join(x for x in (h.title, h.heading) if x) or h.doc
        lines.append(f"[{i}] {label}: {snippet}…")
    return "\n\n".join(lines)


def answer(question: str, hits: list[Hit]) -> tuple[str, str, dict]:
    """Returns (answer_text, model_name, usage)."""
    if not hits:
        return "I don't find that in the rulebook.", "none", {}
    if not ollama_available():
        return _extractive(question, hits), "extractive", {}
    try:
        text = _chat(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"Passages:\n\n{build_context(hits)}\n\nQuestion: {question}",
                },
            ]
        )
        est_in = len(question.split()) + len(build_context(hits).split())
        est_out = len(text.split())
        return text, CHAT_MODEL, {"prompt_tokens": est_in, "completion_tokens": est_out}
    except Exception as e:  # model crashed mid-run — degrade, don't fail
        fallback = _extractive(question, hits)
        return f"{fallback}\n\n(_generation error: {type(e).__name__})", "extractive", {}
