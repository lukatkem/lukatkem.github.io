#!/usr/bin/env python3
"""Hydro-1 playground server.

Serves the web UI and a small API over the locally trained model:
  GET  /api/status       — model info, training history, device
  POST /api/generate     — {prompt, temperature, top_k, max_new_tokens}
  POST /api/inspect      — attention maps + per-char predictions + surprisal
  GET  /api/teachers     — configured teacher roster
  POST /api/arena        — same prompt → student + every teacher
  GET/POST /api/academy… — one-click teacher key, distill + retrain jobs
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import academy
from .generate import generate, history, inspect_prompt, load_model
from .train import device as detect_device

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

app = FastAPI(title="Hydro-1")
academy.apply_env()  # a key saved to .env earlier applies on startup
_model = None
_tokenizer = None
_loaded_mtime: float = 0.0


def _ckpt_mtime() -> float:
    """Watch both students — a finished distilled retrain triggers hot-reload."""
    mtimes = []
    for name in ("latest.pt", "latest-distilled.pt"):
        p = ROOT / "checkpoints" / name
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def model():
    """Lazily load — and hot-reload — the active checkpoint, so a running
    training job upgrades the playground live and 'back to base' swaps it
    back: any change in the checkpoint set (newer OR removed) reloads."""
    global _model, _tokenizer, _loaded_mtime
    mtime = _ckpt_mtime()
    if _model is None or mtime != _loaded_mtime:
        _model, _tokenizer = load_model(device=detect_device())
        _loaded_mtime = mtime
    return _model, _tokenizer


class GenIn(BaseModel):
    prompt: str = "Once upon a time"
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 250


class InspectIn(BaseModel):
    text: str
    top_k: int = 8


@app.get("/api/status")
def status():
    if not (Path(__file__).resolve().parent.parent / "checkpoints" / "latest.pt").exists():
        return {"trained": False}
    m, tok = model()
    h = history()
    return {
        "trained": True,
        "params": m.num_params(),
        "vocab": tok.vocab_size,
        "block_size": m.block_size,
        "layers": len(m.blocks),
        "device": detect_device(),
        "history": {
            "config": h.get("config", {}),
            "loss": h["loss"],
            "val_loss": h["val_loss"],
            "samples": h.get("samples", [])[-6:],
        },
    }


@app.post("/api/generate")
def gen(body: GenIn):
    m, tok = model()
    text = generate(
        m, tok, body.prompt,
        max_new_tokens=min(body.max_new_tokens, 800),
        temperature=body.temperature, top_k=body.top_k,
    )
    return {"text": text}


@app.post("/api/inspect")
def inspect(body: InspectIn):
    m, tok = model()
    text = body.text[: m.block_size]
    if not text:
        raise HTTPException(400, "empty text")
    result = inspect_prompt(m, tok, text, top_k=body.top_k)
    # downsample attention for transport: keep every position but cap seq len client-side
    return result


# ---------- arena: student vs frontier teachers ----------

class ArenaIn(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = 160


@app.get("/api/teachers")
def teachers():
    from .config import all_teachers, key_set

    return {
        "cloud_configured": key_set(),
        "teachers": all_teachers(),
    }


@app.post("/api/arena")
def arena(body: ArenaIn):
    """Same prompt → the locally-born student and every configured teacher.
    Teachers run on their provider's servers; this machine only ships text."""
    import time as _time

    from .config import all_teachers
    from .teachers import TeacherError, teacher_chat

    if not body.prompt.strip():
        raise HTTPException(400, "empty prompt")

    contenders = []

    # 1. Hydro-1 — born on this Mac
    m, tok = model()
    t0 = _time.perf_counter()
    text = generate(m, tok, body.prompt, max_new_tokens=min(body.max_new_tokens, 400))
    contenders.append({
        "id": "hydro-1", "label": f"Hydro-1 ({m.num_params()/1e6:.1f}M, trained here)",
        "origin": "local-born",
        "text": text, "latency_ms": round((_time.perf_counter() - t0) * 1000),
    })

    # 2. Teachers — local fallback and/or cloud frontier
    for t in all_teachers():
        try:
            text, latency = teacher_chat(
                t["id"], body.prompt,
                "Continue the given text as a short children's story. Plain prose only.",
                temperature=0.9, max_tokens=min(body.max_new_tokens * 2, 500),
            )
            contenders.append({
                "id": t["id"], "label": t["label"], "origin": "cloud" if TEACHERS_CLOUD(t["id"]) else "local-teacher",
                "text": text.strip(), "latency_ms": round(latency),
            })
        except TeacherError as e:
            contenders.append({"id": t["id"], "label": t["label"], "origin": "error", "text": str(e), "latency_ms": None})

    return {"contenders": contenders}


def TEACHERS_CLOUD(model_id: str) -> bool:
    from .config import cloud_teachers

    return any(t["id"] == model_id for t in cloud_teachers())


# ---------- academy: one-click teacher key + distill + retrain ----------

class KeyIn(BaseModel):
    key: str
    base_url: str | None = None


class StoriesIn(BaseModel):
    stories: int = 200
    model: str | None = None
    grade_model: str | None = None
    chain: bool = False


class StepsIn(BaseModel):
    steps: int = 8000


@app.get("/api/academy")
def academy_overview():
    from .config import cloud_teachers, key_set

    return {
        "key_set": key_set(),
        "base_url": os.environ.get("TEACHER_BASE_URL", "https://api.z.ai/api/paas/v4"),
        "teachers": cloud_teachers(),
        "distill_ready": (academy.CORPUS_DIR / "distilled.txt").exists(),
        "distilled_model": (ROOT / "checkpoints" / "latest-distilled.pt").exists(),
        "job": academy.status(),
    }


@app.post("/api/academy/key")
def academy_set_key(body: KeyIn):
    from .config import cloud_teachers
    from .teachers import probe

    key = body.key.strip()
    if not key:
        raise HTTPException(400, "empty key")
    saved_as = academy.save_key(key, body.base_url)
    probes = [
        {"id": t["id"], "label": t["label"], "provider": t.get("provider", "?"), **probe(t["id"])}
        for t in cloud_teachers()
    ]
    return {"saved": True, "saved_as": saved_as, "ok": any(p["ok"] for p in probes), "probes": probes}


@app.delete("/api/academy/key")
def academy_remove_key():
    academy.clear_key()
    return {"removed": True}


@app.post("/api/academy/distill")
def academy_distill(body: StoriesIn):
    stories = max(5, min(body.stories, 500))
    r = academy.start("distill", stories=stories, model=body.model,
                      grade_model=body.grade_model, chain=body.chain)
    if not r["started"]:
        raise HTTPException(409, r.get("reason", "cannot start"))
    return r


@app.post("/api/academy/train")
def academy_train(body: StepsIn):
    steps = max(100, min(body.steps, 20000))
    r = academy.start("train", steps=steps)
    if not r["started"]:
        raise HTTPException(409, r.get("reason", "cannot start"))
    return r


@app.get("/api/academy/job")
def academy_job():
    return academy.status()


@app.post("/api/academy/stop")
def academy_stop():
    return {"stopped": academy.stop()}


@app.post("/api/academy/reset")
def academy_reset():
    try:
        removed = academy.reset_distilled()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"removed": removed}


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
