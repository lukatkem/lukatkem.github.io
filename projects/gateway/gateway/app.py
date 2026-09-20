"""FastAPI application: the user-facing surface of the gateway.

``build_app`` is a pure factory — every dependency (providers, keys, upstream
transport, clock, sleep function) is injected, so tests run fully in-process
with ``fastapi.testclient.TestClient`` and ``httpx.MockTransport`` fakes: no
ports, no environment mutation, no network.

Request pipeline for ``POST /v1/chat/completions``::

    auth (401) -> rate limit (429) -> budget (402) -> cache (hit? return)
        -> route: priority order, breaker gate, retries w/ backoff (502 if all fail)
        -> budget ledger accumulate -> response {text, model, usage, served_by,
           cached, latency_ms}
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Mapping, Sequence

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .breaker import CircuitBreaker
from .cache import TTLCache, cache_key
from .config import GatewayConfig, GatewayKey, PriceBook, Provider, parse_prices
from .ratelimit import BudgetLedger, RateLimiter
from .router import AllProvidersFailedError, Router, UnknownModelError
from .upstream import UpstreamClient

__all__ = ["app_from_env", "build_app"]

ANON_KEY_ID = "__anon__"

DEFAULT_BREAKER_THRESHOLD = 3
DEFAULT_BREAKER_COOLDOWN_SECONDS = 30.0
DEFAULT_CACHE_TTL_SECONDS = 300.0
DEFAULT_CACHE_MAXSIZE = 128
DEFAULT_RATE_CAPACITY = 60
DEFAULT_RATE_REFILL_PER_SECOND = 1.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE_SECONDS = 0.25

Clock = Callable[[], float]
Sleep = Callable[[float], None]


def _error(
    status_code: int,
    error_type: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    """A structured error response: ``{"error": {"type", "message", ...}}``."""
    error: dict[str, Any] = {"type": error_type, "message": message}
    error.update(extra)
    return JSONResponse(status_code=status_code, content={"error": error}, headers=headers)


def _validate_chat_body(body: Any) -> str | None:
    """Return an error message for an invalid chat body, or None if it is valid."""
    if not isinstance(body, dict):
        return "request body must be a JSON object"
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        return "'model' must be a non-empty string"
    if body.get("stream"):
        return "streaming responses are not supported by this gateway"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "'messages' must be a non-empty list"
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not message["role"].strip()
            or not isinstance(message.get("content"), str)
        ):
            return f"messages[{index}] must be an object with string 'role' and 'content'"
    max_tokens = body.get("max_tokens")
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0):
        return "'max_tokens' must be a positive integer"
    temperature = body.get("temperature")
    if temperature is not None and (
        isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2
    ):
        return "'temperature' must be a number between 0 and 2"
    return None


def build_app(
    providers: Sequence[Provider],
    keys: Mapping[str, GatewayKey],
    *,
    transport: httpx.BaseTransport | None = None,
    env: Mapping[str, str] | None = None,
    prices: PriceBook | None = None,
    allow_anon: bool = False,
    breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
    breaker_cooldown_seconds: float = DEFAULT_BREAKER_COOLDOWN_SECONDS,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    cache_maxsize: int = DEFAULT_CACHE_MAXSIZE,
    rate_capacity: float = DEFAULT_RATE_CAPACITY,
    rate_refill_per_second: float = DEFAULT_RATE_REFILL_PER_SECOND,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    clock: Clock | None = None,
    sleep: Sleep | None = None,
) -> FastAPI:
    """Build the gateway app with every dependency injected.

    ``providers`` is the upstream roster, ``keys`` maps gateway-key prefix ->
    :class:`~gateway.config.GatewayKey` (SHA-256 of the full token, plan, USD
    budget). ``transport`` is the httpx transport for upstream calls
    (``httpx.MockTransport`` in tests), ``env`` the mapping provider API keys
    are read from at call time, ``clock``/``sleep`` the time sources (fakes in
    tests).
    """
    if not providers:
        raise ValueError("build_app requires at least one provider")
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    provider_list = list(providers)
    keys = dict(keys)
    pricebook = prices or PriceBook()

    upstream = UpstreamClient(transport=transport, env=env, clock=clock)
    breakers = {
        provider.name: CircuitBreaker(
            provider.name,
            failure_threshold=breaker_threshold,
            cooldown_seconds=breaker_cooldown_seconds,
            clock=clock,
        )
        for provider in provider_list
    }
    router = Router(
        provider_list,
        upstream,
        breakers=breakers,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        sleep=sleep,
    )
    cache = TTLCache(maxsize=cache_maxsize, ttl_seconds=cache_ttl_seconds, clock=clock)
    limiter = RateLimiter(capacity=rate_capacity, refill_per_second=rate_refill_per_second, clock=clock)
    budgets: dict[str, float | None] = {prefix: key.usd_budget for prefix, key in keys.items()}
    if allow_anon:
        budgets[ANON_KEY_ID] = None
    ledger = BudgetLedger(budgets)

    class AuthFailure(Exception):
        """Raised when the presented gateway key cannot be authenticated."""

    def authenticate(authorization: str | None) -> tuple[str, GatewayKey]:
        """Resolve the Authorization header to ``(key_id, key)``.

        The full token must start with a configured prefix (longest match wins)
        and hash (SHA-256) to the configured digest. Raises AuthFailure.
        """
        if authorization is None or not authorization.strip():
            if allow_anon:
                return ANON_KEY_ID, GatewayKey(sha256="", plan="anon", usd_budget=None)
            raise AuthFailure("missing API key; send 'Authorization: Bearer <gateway key>'")
        scheme, _, token = authorization.strip().partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise AuthFailure("authorization header must use the Bearer scheme")
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        for prefix in sorted(keys, key=len, reverse=True):
            if token.startswith(prefix) and digest == keys[prefix].sha256:
                return prefix, keys[prefix]
        raise AuthFailure("unknown or invalid gateway key")

    def handle_chat(body: Any, authorization: str | None) -> JSONResponse:
        """The blocking half of the chat pipeline (runs in the threadpool)."""
        started = clock()
        try:
            key_id, _key = authenticate(authorization)
        except AuthFailure as exc:
            return _error(401, "invalid_api_key", str(exc), headers={"WWW-Authenticate": "Bearer"})
        if not limiter.allow(key_id):
            return _error(429, "rate_limit_exceeded", "token bucket exhausted for this API key; slow down")
        if ledger.exceeded(key_id):
            usage = ledger.usage(key_id)
            return _error(
                402,
                "budget_exceeded",
                f"USD budget exhausted for key '{key_id}': spent ${usage['usd_spent']:.6f} "
                f"of ${usage['usd_budget']:.6f}",
            )

        validation_error = _validate_chat_body(body)
        if validation_error is not None:
            return _error(400, "invalid_request", validation_error)
        model = str(body["model"])
        messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in body["messages"]]
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")

        key = cache_key(model, messages)
        hit = cache.get(key)
        if hit is not None:
            return JSONResponse({**hit, "cached": True, "latency_ms": round((clock() - started) * 1000.0, 3)})

        try:
            result = router.complete(model, messages, max_tokens=max_tokens, temperature=temperature)
        except UnknownModelError as exc:
            return _error(404, "unknown_model", str(exc))
        except AllProvidersFailedError as exc:
            return _error(502, "upstream_unavailable", "all upstream providers failed for this model", attempts=exc.attempts)
        except Exception:  # pragma: no cover - defensive
            return _error(500, "internal_error", "unexpected gateway failure")

        ledger.record(
            key_id,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            usd=pricebook.cost(result.served_by, result.prompt_tokens, result.completion_tokens),
        )
        payload = {
            "text": result.text,
            "model": result.model,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            },
            "served_by": result.served_by,
            "cached": False,
            "latency_ms": result.latency_ms,
        }
        cache.put(key, payload)
        return JSONResponse(payload)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        upstream.close()

    app = FastAPI(title="LLM Gateway", version="1.0.0", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return _error(400, "invalid_request", "request body must be valid JSON")
        return await run_in_threadpool(handle_chat, body, request.headers.get("authorization"))

    @app.get("/admin/status")
    def admin_status() -> dict[str, Any]:
        """Provider health, per-key usage and cache stats for the dashboard."""
        provider_rows = []
        for provider in router.ordered_providers:
            breaker_snapshot = breakers[provider.name].snapshot()
            stats = router.stats.snapshot(provider.name)
            provider_rows.append(
                {
                    "name": provider.name,
                    "priority": provider.priority,
                    "models": list(provider.models),
                    "state": breaker_snapshot["state"],
                    "consecutive_failures": breaker_snapshot["consecutive_failures"],
                    "requests": stats["requests"],
                    "failures": stats["failures"],
                    "last_latency_ms": stats["last_latency_ms"],
                }
            )
        key_rows = {}
        for prefix, key in keys.items():
            row = ledger.usage(prefix)
            row["plan"] = key.plan
            key_rows[prefix] = row
        if allow_anon:
            anon_row = ledger.usage(ANON_KEY_ID)
            anon_row["plan"] = "anon"
            key_rows[ANON_KEY_ID] = anon_row
        return {
            "providers": provider_rows,
            "keys": key_rows,
            "cache": cache.stats(),
            "settings": {
                "allow_anon": allow_anon,
                "max_retries": max_retries,
                "breaker_threshold": breaker_threshold,
                "breaker_cooldown_seconds": breaker_cooldown_seconds,
            },
        }

    @app.post("/admin/cache/clear")
    def admin_cache_clear() -> dict[str, Any]:
        return {"cleared": True, "entries_removed": cache.clear()}

    bootstrap = json.dumps({"providers": [p.name for p in router.ordered_providers], "allow_anon": allow_anon})
    html = DASHBOARD_HTML.replace("__BOOTSTRAP_JSON__", bootstrap)

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(html)

    return app


def app_from_env(env: Mapping[str, str] | None = None) -> FastAPI:
    """Production entry point: build the gateway from environment variables."""
    config = GatewayConfig.from_env(env)
    return build_app(config.providers, config.keys, prices=config.prices, allow_anon=config.allow_anon)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LLM Gateway Dashboard</title>
<style>
  :root {
    --bg: #0b0f14; --panel: #12171f; --border: #1e2630; --text: #e6edf3;
    --muted: #8b949e; --green: #3fb950; --red: #f85149; --yellow: #d29922; --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; background: var(--bg); color: var(--text);
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); margin: 28px 0 8px; }
  .sub { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; background: var(--green); }
  .dot.err { background: var(--red); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .card .top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .name { font-weight: 700; font-size: 15px; }
  .badge { font-size: 11px; text-transform: uppercase; letter-spacing: .6px; padding: 2px 9px; border-radius: 999px; border: 1px solid; }
  .badge.closed { color: var(--green); border-color: var(--green); background: rgba(63,185,80,.12); }
  .badge.open { color: var(--red); border-color: var(--red); background: rgba(248,81,73,.12); }
  .badge.half_open { color: var(--yellow); border-color: var(--yellow); background: rgba(210,153,34,.12); }
  .kv { display: flex; justify-content: space-between; padding: 2px 0; font-size: 12.5px; }
  .kv span { color: var(--muted); }
  table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 12.5px; }
  th { color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: .6px; }
  tr:last-child td { border-bottom: none; }
  .error { color: var(--red); }
</style>
</head>
<body>
  <h1>LLM Gateway</h1>
  <div class="sub"><span class="dot" id="dot"></span><span id="updated">connecting&hellip;</span> &middot; <span id="settings"></span></div>

  <h2>Providers</h2>
  <div class="grid" id="providers"></div>

  <h2>API keys</h2>
  <table>
    <thead><tr><th>key</th><th>plan</th><th>requests</th><th>tokens</th><th>est. spend</th><th>budget</th><th>remaining</th></tr></thead>
    <tbody id="keys"></tbody>
  </table>

  <h2>Response cache</h2>
  <div class="card" id="cache"></div>

<script>
"use strict";
const BOOTSTRAP = __BOOTSTRAP_JSON__;
const POLL_MS = 2000;

function esc(value) {
  const div = document.createElement("div");
  div.textContent = value === null || value === undefined ? "&mdash;" : String(value);
  return div.innerHTML;
}
function usd(value) { return value === null || value === undefined ? "&mdash;" : "$" + Number(value).toFixed(6); }

function renderProviders(rows) {
  document.getElementById("providers").innerHTML = rows.map(function (p) {
    return `
      <div class="card">
        <div class="top"><span class="name">${esc(p.name)}</span><span class="badge ${esc(p.state)}">${esc(p.state)}</span></div>
        <div class="kv"><span>priority</span><b>${esc(p.priority)}</b></div>
        <div class="kv"><span>models</span><b>${esc((p.models || []).join(", "))}</b></div>
        <div class="kv"><span>requests</span><b>${esc(p.requests)}</b></div>
        <div class="kv"><span>failures</span><b>${esc(p.failures)}</b></div>
        <div class="kv"><span>consecutive failures</span><b>${esc(p.consecutive_failures)}</b></div>
        <div class="kv"><span>last latency</span><b>${p.last_latency_ms === null ? "&mdash;" : esc(p.last_latency_ms) + " ms"}</b></div>
      </div>`;
  }).join("");
}

function renderKeys(rows) {
  document.getElementById("keys").innerHTML = rows.map(function (entry) {
    const id = entry[0], k = entry[1];
    return `
      <tr>
        <td>${esc(id)}</td><td>${esc(k.plan)}</td><td>${esc(k.requests)}</td>
        <td>${esc(k.total_tokens)}</td><td>${usd(k.usd_spent)}</td>
        <td>${usd(k.usd_budget)}</td><td>${usd(k.budget_remaining)}</td>
      </tr>`;
  }).join("") || '<tr><td colspan="7" class="error">no keys configured</td></tr>';
}

function renderCache(c) {
  const lookups = (c.hits || 0) + (c.misses || 0);
  const rate = lookups === 0 ? "0.0" : (100 * c.hits / lookups).toFixed(1);
  document.getElementById("cache").innerHTML = `
    <div class="kv"><span>hits</span><b>${esc(c.hits)}</b></div>
    <div class="kv"><span>misses</span><b>${esc(c.misses)}</b></div>
    <div class="kv"><span>entries</span><b>${esc(c.size)}</b></div>
    <div class="kv"><span>hit rate</span><b>${rate}%</b></div>`;
}

function renderSettings(s) {
  if (!s) return;
  document.getElementById("settings").textContent =
    "max_retries " + s.max_retries +
    " · breaker opens after " + s.breaker_threshold + " failures" +
    " · cooldown " + s.breaker_cooldown_seconds + "s" +
    " · anon " + (s.allow_anon ? "on" : "off");
}

async function poll() {
  try {
    const response = await fetch("/admin/status", { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    renderProviders(data.providers || BOOTSTRAP.providers.map(function (name) { return { name: name }; }));
    renderKeys(Object.entries(data.keys || {}));
    renderCache(data.cache || { hits: 0, misses: 0, size: 0 });
    renderSettings(data.settings);
    document.getElementById("dot").classList.remove("err");
    document.getElementById("updated").textContent = "live · updated " + new Date().toLocaleTimeString();
  } catch (err) {
    document.getElementById("dot").classList.add("err");
    document.getElementById("updated").textContent = "status unavailable: " + err.message;
  }
}

poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
"""
