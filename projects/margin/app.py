"""Margin API — FastAPI app.

Routes:
  POST /api/chat          — RAG answer with citations (session, API key, or anon demo quota)
  GET  /api/search        — retrieval-only (no generation)
  POST /api/upload        — ingest user docs (md/txt/pdf)
  GET  /api/doc           — doc preview
  POST /api/auth/*        — signup/login/logout/me
  POST /api/keys          — create API key · GET /api/keys · DELETE /api/keys/{prefix}
  GET  /api/usage         — quota state
  POST /api/billing/*     — checkout + Stripe webhook (simulated without keys)
  GET  /admin/traces      — observability JSON (admin token)
  GET  /api/health
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, billing, tracing
from .config import ADMIN_TOKEN, ANON_DAILY_LIMIT, CHAT_MODEL, CORPUS_DIR, PLANS, WEB_DIR
from .generate import answer, build_messages, ollama_available, stream_chunks
from .ingest import MARKDOWN_EXTS
from .retrieve import Retriever
from .store import connect, new_token, now
from .streamfmt import ndjson_line

app = FastAPI(title="Margin", version="0.1.0")
_retriever: Retriever | None = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def reindex() -> dict:
    global _retriever
    from .ingest import ingest

    stats = ingest()
    _retriever = None
    return stats


# ---------- models ----------

class ChatIn(BaseModel):
    question: str
    top_k: int = 5


class SignupIn(BaseModel):
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class CheckoutIn(BaseModel):
    plan: str


# ---------- helpers ----------

def _bearer_user(request: Request):
    """Resolve user from Authorization: Bearer mgn_… API key."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer mgn_"):
        return auth.user_for_api_key(header.split(" ", 1)[1].strip())
    return None


def _session_user(request: Request):
    return auth.user_for_session(request.cookies.get("margin_session", ""))


def _plan_prefix(user) -> tuple[str, str]:
    """(api_key_prefix, plan) for a user row — sqlite3.Row has no .get()."""
    keys = user.keys()
    prefix = user["key_prefix"] if "key_prefix" in keys else "session"
    plan = user["key_plan"] if "key_plan" in keys else user["plan"]
    return prefix, plan


def _anon_blocked(request: Request) -> bool:
    ip = request.client.host if request.client else "?"
    conn = connect()
    row = conn.execute(
        """SELECT COUNT(*) c FROM usage u LEFT JOIN api_keys k ON k.prefix = u.api_key_prefix
           WHERE u.route = '/api/chat' AND u.user_id IS NULL AND u.api_key_prefix LIKE ?
             AND u.month = ?""",
        (f"anon:{ip}%", auth.month_key()),
    ).fetchone()
    return row["c"] >= ANON_DAILY_LIMIT


# ---------- routes ----------

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "chunks": len(retriever().ids),
        "embedder": retriever().kind,
        "llm": ollama_available(),
        "stripe": billing.stripe_enabled(),
    }


# ---------- auth / quota shared by chat + chat/stream ----------

def _authorize_chat(body: ChatIn, request: Request):
    """Common gate for both chat routes: session/API key identity + quota."""
    if not body.question.strip():
        raise HTTPException(400, "empty question")
    user = _bearer_user(request) or _session_user(request)
    prefix = None
    quota = None

    if user:
        prefix, plan = _plan_prefix(user)
        quota = auth.quota_state(prefix, plan)
        if quota["remaining"] <= 0:
            raise HTTPException(402, f"monthly quota exhausted ({quota['quota']}/mo). Upgrade at /pricing.")
    else:
        if _anon_blocked(request):
            raise HTTPException(429, f"anonymous demo limit reached ({ANON_DAILY_LIMIT}/day). Sign up free.")
        ip = request.client.host if request.client else "?"
        prefix = f"anon:{ip}:{new_token()[:8]}"
    return user, prefix, quota


def _citations(hits) -> list[dict]:
    return [
        {"n": i, "doc": h.doc, "title": h.title, "heading": h.heading,
         "score": h.score, "snippet": h.text[:280]}
        for i, h in enumerate(hits, 1)
    ]


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    user, prefix, quota = _authorize_chat(body, request)

    t0 = time.perf_counter()
    hits = retriever().search(body.question, top_k=body.top_k)
    text, model, usage = answer(body.question, hits)
    latency = (time.perf_counter() - t0) * 1000

    trace_id = tracing.record_trace(
        route="/api/chat", query=body.question, latency_ms=latency, hits=hits,
        model=model, prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        user_id=user["id"] if user else None,
    )
    auth.meter(user["id"] if user else None, prefix, "/api/chat")

    return {
        "answer": text,
        "model": model,
        "citations": _citations(hits),
        "latency_ms": round(latency, 1),
        "trace_id": trace_id,
        "quota": quota,
    }


@app.post("/api/chat/stream")
def chat_stream(body: ChatIn, request: Request):
    """Same pipeline as /api/chat, but tokens arrive as they are generated
    (NDJSON: meta → delta… → final). Retrieval happens up front so the UI
    can show citations while the model is still writing."""
    user, prefix, quota = _authorize_chat(body, request)
    hits = retriever().search(body.question, top_k=body.top_k)
    live = ollama_available()

    def generate():
        t0 = time.perf_counter()
        model = CHAT_MODEL if live else "extractive"
        yield ndjson_line({"type": "meta", "model": model, "citations": _citations(hits)})

        pieces: list[str] = []
        usage: dict = {}
        if not hits:
            pieces = ["I don't find that in the rulebook."]
            yield ndjson_line({"type": "delta", "text": pieces[0]})
        elif not live:
            from .generate import _extractive

            pieces = [_extractive(body.question, hits)]
            yield ndjson_line({"type": "delta", "text": pieces[0]})
        else:
            try:
                for piece in stream_chunks(build_messages(body.question, hits)):
                    pieces.append(piece)
                    yield ndjson_line({"type": "delta", "text": piece})
            except Exception as e:  # mid-stream failure → degrade honestly
                from .generate import _extractive

                note = _extractive(body.question, hits)
                pieces = [note]
                yield ndjson_line({"type": "delta", "text": f"\n\n{note}\n\n(_stream error: {type(e).__name__})"})
                model = "extractive"

        text = "".join(pieces)
        latency = (time.perf_counter() - t0) * 1000
        trace_id = tracing.record_trace(
            route="/api/chat/stream", query=body.question, latency_ms=latency, hits=hits,
            model=model, prompt_tokens=0, completion_tokens=len(text.split()),
            user_id=user["id"] if user else None,
        )
        auth.meter(user["id"] if user else None, prefix, "/api/chat")
        yield ndjson_line({
            "type": "final", "answer": text, "model": model,
            "latency_ms": round(latency, 1), "trace_id": trace_id, "quota": quota,
        })

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/search")
def search(q: str, top_k: int = 5):
    hits = retriever().search(q, top_k=top_k)
    return {"hits": [{"doc": h.doc, "title": h.title, "heading": h.heading,
                      "score": h.score, "snippet": h.text[:280]} for h in hits]}


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    user = _bearer_user(request) or _session_user(request)
    if not user:
        raise HTTPException(401, "sign in to upload documents")
    name = Path(file.filename or "upload.md")
    suffix = name.suffix.lower()
    user_dir = CORPUS_DIR / "_user"
    user_dir.mkdir(exist_ok=True)
    dest = user_dir / f"{user['id']}-{name.name}"

    if suffix in MARKDOWN_EXTS:
        dest.write_text((await file.read()).decode("utf-8", "ignore"), encoding="utf-8")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(400, "PDF support requires pypdf (pip install pypdf)")
        reader = PdfReader(await file.read())
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        dest.with_suffix(".md").write_text(f"# {name.stem}\n\n{text}", encoding="utf-8")
    else:
        raise HTTPException(400, "supported: .md .txt .pdf")

    stats = reindex()
    return {"ok": True, "saved": dest.name, **stats}


@app.get("/api/doc")
def doc_preview(path: str):
    full = (CORPUS_DIR / path).resolve()
    if not str(full).startswith(str(CORPUS_DIR.resolve())) or not full.exists():
        raise HTTPException(404, "doc not found")
    return {"path": path, "content": full.read_text(encoding="utf-8")}


@app.post("/api/admin/reindex")
def admin_reindex(request: Request):
    if ADMIN_TOKEN and request.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(403, "bad admin token")
    return reindex()


# ---------- auth ----------

@app.post("/api/auth/signup")
def signup(body: SignupIn):
    if len(body.password) < 8:
        raise HTTPException(400, "password must be 8+ chars")
    if auth.get_user_by_email(body.email):
        raise HTTPException(409, "email already registered")
    uid = auth.create_user(body.email, body.password)
    token = new_token()
    conn = connect()
    conn.execute("INSERT INTO sessions(token,user_id,created_at) VALUES(?,?,?)", (token, uid, now()))
    conn.commit()
    resp = JSONResponse({"ok": True, "email": body.email})
    resp.set_cookie("margin_session", token, httponly=True, samesite="lax", max_age=30 * 86400)
    return resp


@app.post("/api/auth/login")
def login(body: LoginIn):
    token = auth.login(body.email, body.password)
    if not token:
        raise HTTPException(401, "invalid credentials")
    resp = JSONResponse({"ok": True})
    resp.set_cookie("margin_session", token, httponly=True, samesite="lax", max_age=30 * 86400)
    return resp


@app.post("/api/auth/logout")
def logout(request: Request):
    token = request.cookies.get("margin_session", "")
    conn = connect()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("margin_session")
    return resp


@app.get("/api/auth/me")
def me(request: Request):
    user = _bearer_user(request) or _session_user(request)
    if not user:
        return {"user": None}
    prefix, plan = _plan_prefix(user)
    return {"user": {"email": user["email"], "plan": plan, "quota": auth.quota_state(prefix, plan)}}


# ---------- keys ----------

@app.post("/api/keys")
def create_key(request: Request):
    user = _session_user(request)
    if not user:
        raise HTTPException(401, "sign in first")
    key, prefix = auth.create_api_key(user["id"], user["plan"])
    return {"api_key": key, "prefix": prefix, "note": "shown once — store it now"}


@app.get("/api/keys")
def list_keys(request: Request):
    user = _session_user(request)
    if not user:
        raise HTTPException(401, "sign in first")
    conn = connect()
    rows = conn.execute(
        "SELECT prefix, plan, revoked, created_at FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
        (user["id"],),
    ).fetchall()
    return {"keys": [dict(r) for r in rows]}


@app.delete("/api/keys/{prefix}")
def revoke_key(prefix: str, request: Request):
    user = _session_user(request)
    if not user:
        raise HTTPException(401, "sign in first")
    conn = connect()
    conn.execute("UPDATE api_keys SET revoked = 1 WHERE prefix = ? AND user_id = ?", (prefix, user["id"]))
    conn.commit()
    return {"ok": True}


@app.get("/api/usage")
def usage(request: Request):
    user = _bearer_user(request) or _session_user(request)
    if not user:
        raise HTTPException(401, "auth required")
    prefix, plan = _plan_prefix(user)
    return auth.quota_state(prefix, plan)


# ---------- billing ----------

@app.post("/api/billing/checkout")
def checkout(body: CheckoutIn, request: Request):
    user = _session_user(request)
    if not user:
        raise HTTPException(401, "sign in first")
    result = billing.create_checkout(body.plan, user["email"], str(request.base_url).rstrip("/"))
    if result.get("simulated"):
        auth.set_plan(user["id"], body.plan)
        result["note"] = "simulated billing (no STRIPE_SECRET_KEY) — plan upgraded locally"
    return result


@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    result = billing.handle_webhook(payload, request.headers.get("stripe-signature"))
    if isinstance(result, tuple):
        return JSONResponse(result[0], status_code=result[1])
    return result


# ---------- observability ----------

@app.get("/admin/traces")
def admin_traces(request: Request):
    if ADMIN_TOKEN and request.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(403, "bad admin token")
    return tracing.stats()


@app.get("/api/plans")
def plans():
    return {"plans": PLANS, "stripe": billing.stripe_enabled()}


# ---------- static ----------

@app.get("/favicon.ico")
def favicon():
    return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
