"""Unified teacher client — one interface, any provider, nothing downloaded.

Two transport modes:
  * cloud (TEACHER_API_KEY set): POST {TEACHER_BASE_URL}/chat/completions —
    the OpenAI-compatible contract used by Z.AI (GLM 5.3 / GLM 5.3 Flash),
    OpenRouter, vLLM gateways, etc. Inference happens on the provider's
    servers; this Mac only sends and receives text.
  * local fallback: the Ollama daemon's OpenAI-compatible endpoint — used
    when no key is configured so the Arena still works.

Every call returns (text, latency_ms) or raises TeacherError with a message
safe to show in the UI.
"""
from __future__ import annotations

import json
import os
import time

import httpx

from .config import OLLAMA_BASE, TEACHER_TIMEOUT


class TeacherError(RuntimeError):
    pass


def _chat(url: str, headers: dict, payload: dict, model_label: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=TEACHER_TIMEOUT)
    except httpx.HTTPError as e:
        raise TeacherError(f"{model_label}: connection failed ({type(e).__name__})") from e
    latency = (time.perf_counter() - t0) * 1000
    if r.status_code == 401:
        raise TeacherError(f"{model_label}: invalid API key")
    if r.status_code == 402:
        raise TeacherError(f"{model_label}: payment/credits required by provider")
    if r.status_code == 429:
        raise TeacherError(f"{model_label}: rate limited — retry shortly")
    if r.status_code >= 400:
        detail = r.json().get("error", {}).get("message", r.text[:120]) if r.text else r.status_code
        raise TeacherError(f"{model_label}: {detail}")
    try:
        choice = r.json()["choices"][0]
        msg = choice.get("message", {})
        # reasoning models (GLM 5.3, DeepSeek V4…) may answer in reasoning
        # fields or exhaust the budget thinking — handle both shapes
        text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not text.strip() and choice.get("finish_reason") == "length":
            raise TeacherError(f"{model_label}: spent its whole token budget reasoning — raise max_tokens")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise TeacherError(f"{model_label}: unexpected response shape") from e
    return text, latency


def teacher_chat(model_id: str, prompt: str, system: str, temperature: float = 0.9, max_tokens: int = 900) -> tuple[str, float]:
    """Route to the right provider for this model id (each roster entry
    carries its own base_url + key_env), else local Ollama. Reads keys
    through os.environ each call — keys saved from the UI apply live."""
    from .config import DEFAULT_TEACHER_BASE, cloud_teachers

    entry = next((t for t in cloud_teachers() if t["id"] == model_id), None)
    key = ""
    base = ""
    if entry is not None and entry.get("key_env"):
        key = os.environ.get(entry["key_env"], "")
        base = entry.get("base_url") or os.environ.get("TEACHER_BASE_URL", DEFAULT_TEACHER_BASE)
    elif os.environ.get("TEACHER_API_KEY"):
        key = os.environ["TEACHER_API_KEY"]
        base = os.environ.get("TEACHER_BASE_URL", DEFAULT_TEACHER_BASE)
    if key:
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    else:
        base = os.environ.get("OLLAMA_BASE", OLLAMA_BASE)
        url = f"{base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    return _chat(url, headers, payload, model_id)


def probe(model_id: str) -> dict:
    """Quick liveness check used by /api/teachers. Reasoning models need
    headroom to answer even one word, so the budget is small but not tiny."""
    try:
        text, latency = teacher_chat(model_id, "Say OK.", "Reply with one word.", max_tokens=256)
        return {"ok": True, "latency_ms": round(latency)}
    except TeacherError as e:
        return {"ok": False, "error": str(e)}
