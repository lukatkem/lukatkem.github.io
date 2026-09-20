# Hydrogen Gateway — a resilient multi-provider LLM gateway

One OpenAI-compatible endpoint in front of several upstream LLM providers, with
priority failover, per-provider circuit breakers, retries with exponential
backoff, an exact-match TTL cache, per-key token-bucket rate limits, USD
budgets, and a live dashboard.

## Why a gateway exists

The true story: every production LLM app eventually learns that **providers die
mid-run**. The vendor you picked has a bad deploy, rate-limits you at 4pm, or
starts returning 502s while your batch job is halfway through. Without a
gateway, every client application has to know the fallback order, the retry
policy, the breaker state, and the billing math — and each one gets it wrong
differently. A gateway puts that resilience in one place:

- applications speak one OpenAI-shaped protocol and never change when you
  switch, add, or demote providers;
- failover, retries and circuit breakers live server-side, uniformly;
- spend is visible and capped per key before the invoice arrives;
- identical repeated requests can be served from cache without burning tokens.

## Architecture

```
              ┌──────────────────────────────────────────────────────────────────┐
              │                             client                               │
              └────────────────────────────┬─────────────────────────────────────┘
                                           │  POST /v1/chat/completions
                                           │  Authorization: Bearer gw-…
                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ gateway/app.py                                                                    │
│                                                                                   │
│   ┌──────┐    ┌───────────┐    ┌────────┐    ┌─────────┐    ┌───────────────────┐ │
│   │ auth │ ─► │ ratelimit │ ─► │ budget │ ─► │  cache  │ ─► │     router.py     │ │
│   │ 401  │    │    429    │    │  402   │    │ hit? -> │    │ providers for the │ │
│   └──────┘    └───────────┘    └────────┘    │ return  │    │ model by priority │ │
│                                              │ cached  │    │ breaker gate ->   │ │
│                                              └─────────┘    │ retries + exp.    │ │
│                                                             │ backoff -> next   │ │
│   GET / and /dashboard ── poll GET /admin/status ──┐         └─────────┬─────────┘ │
│   POST /admin/cache/clear                          │ every 2s          │           │
└────────────────────────────────────────────────────┼───────────────────┼───────────┘
                                                     │                   │ failover
                                              stats: health,         chain
                                              key usage, cache             │
                                                     │        ┌────────────┴────────────┐
                                                     ▼        ▼                         ▼
                                          ┌───────────────────────┐        ┌───────────────────┐
                                          │  upstream: frontier   │ prio 1 │  upstream: cheap  │ prio 2
                                          │  breaker: closed/open │──────► │  breaker: closed  │
                                          └───────────────────────┘ on fail└───────────────────┘
```

Request flow: **auth → rate limit → budget → cache → breaker + retry chain →
providers**, then the ledger accumulates the estimated cost of the response.

### Semantics

| Concern | Behavior |
| --- | --- |
| Failover | Providers serving the model, ordered by `(priority, name)`; first success wins. |
| Retries | `max_retries + 1` attempts per provider with exponential backoff (`0.25s`, `0.5s`, …). |
| Circuit breaker | Per provider: 3 consecutive failures → `open`; after a 30s cooldown → `half_open`; one success → `closed`, one failure → `open` again. An `open` provider is skipped instantly. |
| Cache | Exact match on `sha256(model + json(messages, sort_keys=True))`, TTL 300s, maxsize 128; only successful responses are cached; hits carry `"cached": true`. |
| Rate limit | Token bucket per key (capacity 60, refill 1/s); exhausted → `429`. |
| Budget | Ledger per key accumulating estimated USD; check before routing; exceeded → `402`. |
| Cost model | Tokens ≈ `len(text) // 4` (provider-reported usage is preferred when present); price per provider, default $0.002 / 1k prompt + $0.006 / 1k completion tokens. |
| Auth | Full token must start with a configured prefix (longest match wins) and SHA-256 to the configured digest. |

## Configuration

Everything is env-driven for deployment and injectable for tests
(`build_app(providers, keys, transport=..., clock=..., sleep=...)`).

| Variable | Meaning |
| --- | --- |
| `GATEWAY_PROVIDERS` | JSON list of providers: `[{"name", "base_url", "api_key_env", "models", "priority"}]` |
| `GATEWAY_KEYS` | JSON object of gateway keys: `{"<prefix>": {"sha256": "...", "plan": "pro", "usd_budget": 5.0}}` (`usd_budget: null` = unlimited) |
| `GATEWAY_PRICES` | Optional JSON: `{"<provider|default>": {"prompt_per_1k": x, "completion_per_1k": y}}` |
| `GATEWAY_ALLOW_ANON` | `1` allows keyless requests (they share the `__anon__` bucket and have no budget) |

Upstream credentials are **never** stored in config: `api_key_env` names the
environment variable that is read at call time (e.g. `FRONTIER_API_KEY`).

Default example roster (used when `GATEWAY_PROVIDERS` is unset): a `frontier`
provider (priority 1) and a `cheap` provider (priority 2), both placeholder
`*.example` URLs.

Generate a gateway key (run locally; nothing sensitive is committed):

```bash
python - <<'PY'
import hashlib, secrets
token = "gw-prod-" + secrets.token_urlsafe(18)
print("token :", token)                                   # give this to the client
print("sha256:", hashlib.sha256(token.encode()).hexdigest())  # put this in GATEWAY_KEYS
PY
```

`GATEWAY_KEYS` uses the *prefix* as the object key — pick a distinctive prefix
of the token (here `gw-prod`) so the dashboard shows friendly key ids.

## Running

```bash
export GATEWAY_PROVIDERS='[{"name":"frontier","base_url":"https://api.frontier.example/v1","api_key_env":"FRONTIER_API_KEY","models":["m1"],"priority":1},{"name":"cheap","base_url":"https://api.cheap.example/v1","api_key_env":"CHEAP_API_KEY","models":["m1"],"priority":2}]'
export GATEWAY_KEYS='{"gw-prod":{"sha256":"<64 hex chars>","plan":"pro","usd_budget":5.0}}'
uvicorn gateway.app:app_from_env --factory --host 127.0.0.1 --port 8000
```

## API

### `POST /v1/chat/completions`

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer gw-prod-XXXXXXXXXXXX" \
  -H "Content-Type: application/json" \
  -d '{"model":"m1","messages":[{"role":"user","content":"hello"}],"max_tokens":64,"temperature":0.7}'
```

```json
{
  "text": "hello from frontier",
  "model": "m1",
  "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
  "served_by": "frontier",
  "cached": false,
  "latency_ms": 213.417
}
```

Errors are structured: `{"error": {"type": "...", "message": "...", ...}}` with
`400 invalid_request`, `401 invalid_api_key`, `402 budget_exceeded`,
`404 unknown_model`, `429 rate_limit_exceeded`, `502 upstream_unavailable`
(the 502 body includes the per-attempt log of what failed where).

### `GET /admin/status`

```bash
curl -s http://127.0.0.1:8000/admin/status | python -m json.tool
```

Returns provider health (name, breaker state, consecutive failures, last
latency, requests, failures), per-key usage (requests, tokens, est. USD,
budget remaining) and cache stats (hits, misses, size, hit rate).

### `POST /admin/cache/clear`

```bash
curl -s -X POST http://127.0.0.1:8000/admin/cache/clear
# {"cleared": true, "entries_removed": 3}
```

### `GET /` and `GET /dashboard`

Single-file dark dashboard (inline CSS/JS, no frameworks, no build step). It
renders provider cards — breaker state colored green (`closed`), yellow
(`half_open`), red (`open`) — plus the per-key usage table and cache stats,
polling `/admin/status` every 2 seconds.

## Testing

```bash
cd gateway && python -m pytest tests/ -q
```

All upstreams are `httpx.MockTransport` fakes; clocks and sleeps are injected
fakes (no real sleeps); `fastapi.testclient.TestClient` runs everything
in-process (no ports); gateway keys are SHA-256 digests of test tokens. No
network calls, no real keys, no long-lived processes.

## Project layout

```
gateway/
├── gateway/
│   ├── __init__.py      # package exports
│   ├── config.py        # providers, gateway keys, prices (env or direct)
│   ├── upstream.py      # OpenAI-compatible HTTP client (injectable transport)
│   ├── breaker.py       # per-provider circuit breaker
│   ├── router.py        # priority routing, retries, breaker integration
│   ├── cache.py         # exact-match TTL response cache
│   ├── ratelimit.py     # token-bucket limiter + USD budget ledger
│   └── app.py           # FastAPI factory, admin endpoints, dashboard
├── tests/
│   └── test_gateway.py  # MockTransport fakes, fake clocks, in-process TestClient
└── README.md
```

## Honest limitations

- **Exact-match cache only.** Two requests must serialize to identical
  `(model, messages)` JSON to hit; no semantic matching, no response-level
  normalization, no cross-process sharing.
- **Non-streaming only.** `stream: true` is rejected with `400`; SSE pass-through
  is not implemented.
- **State is in-process.** Breakers, buckets, ledger and cache live in the
  worker's memory: run one worker, or accept per-worker divergence; add Redis
  for shared state.
- **Admin endpoints are unauthenticated.** `/admin/*` is meant for localhost /
  an internal network — put it behind a proxy with its own auth otherwise.
- **Token counting is a length heuristic** (`len // 4`) unless the provider
  reports usage; real tokenizers will differ.
- **The breaker's half-open state admits multiple probes** (no single-flight),
  so a recovering provider may receive a burst of trial calls.
- **No request coalescing**: concurrent identical requests on a cold cache each
  hit upstream (thundering herd).
- **The ledger is not persisted**: restarts reset budgets and usage.
- **Key matching is prefix-based**: the longest configured prefix that is a
  prefix of the presented token wins; rotating a key means publishing a new one.
