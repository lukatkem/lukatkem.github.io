"""Tests for the gateway: every upstream is an httpx.MockTransport fake.

No real network, no ports (``fastapi.testclient.TestClient`` is in-process),
no real sleeps (clock and sleep are injected fakes), and no real API keys:
upstream credentials come from an injected env mapping, gateway keys are
SHA-256 hashes of test tokens computed here at test time.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

import httpx
from fastapi.testclient import TestClient

from gateway.app import build_app
from gateway.config import GatewayConfig, GatewayKey, Provider

TEST_KEY_FULL = "gw-test-0123456789abcdef"
TEST_KEY_PREFIX = "gw-test"
CHAT_URL = "/v1/chat/completions"

FRONTIER_ENV_KEY = "test-frontier-key"
CHEAP_ENV_KEY = "test-cheap-key"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeClock:
    """Injectable monotonic clock that only moves when a test advances it."""

    def __init__(self) -> None:
        self._now = 1_000.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _chat_response(text: str, usage: dict[str, int] | None = None) -> httpx.Response:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "m1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


class UpstreamStub:
    """Fake upstream cluster keyed by host; records calls and auth headers."""

    HOST_FOR = {"frontier.test": "frontier", "cheap.test": "cheap"}

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.auth_headers: dict[str, str | None] = {}
        self._behaviors: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.transport = httpx.MockTransport(self._handle)

    def set_ok(self, name: str, text: str, usage: dict[str, int] | None = None) -> None:
        self._behaviors[name] = lambda _request: _chat_response(text, usage)

    def set_status(self, name: str, status: int) -> None:
        self._behaviors[name] = lambda _request: httpx.Response(status)

    def set_handler(self, name: str, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._behaviors[name] = handler

    def count(self, name: str) -> int:
        return self.calls.count(name)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        name = self.HOST_FOR[request.url.host]
        self.calls.append(name)
        self.auth_headers[name] = request.headers.get("authorization")
        behavior = self._behaviors.get(name)
        if behavior is None:
            return httpx.Response(500)
        return behavior(request)


PROVIDERS = (
    Provider(name="frontier", base_url="https://frontier.test/v1", api_key_env="FRONTIER_API_KEY", models=("m1", "m2"), priority=1),
    Provider(name="cheap", base_url="https://cheap.test/v1", api_key_env="CHEAP_API_KEY", models=("m1", "m2"), priority=2),
)


def _keys(budget: float | None = 5.0) -> dict[str, GatewayKey]:
    return {TEST_KEY_PREFIX: GatewayKey(sha256=_sha256(TEST_KEY_FULL), plan="pro", usd_budget=budget)}


def make_gateway(
    stub: UpstreamStub,
    *,
    clock: FakeClock | None = None,
    providers: Any = None,
    **overrides: Any,
) -> tuple[TestClient, FakeClock, list[float]]:
    """Build an in-process TestClient with everything injected."""
    clock = clock or FakeClock()
    sleeps: list[float] = []
    params: dict[str, Any] = {
        "transport": stub.transport,
        "env": {"FRONTIER_API_KEY": FRONTIER_ENV_KEY, "CHEAP_API_KEY": CHEAP_ENV_KEY},
        "clock": clock,
        "sleep": sleeps.append,
        "backoff_base_seconds": 0.25,
    }
    params.update(overrides)
    keys = params.pop("keys", None) or _keys()
    app = build_app(providers if providers is not None else PROVIDERS, keys, **params)
    return TestClient(app), clock, sleeps


def auth_headers(token: str = TEST_KEY_FULL) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def chat_body(content: str = "hello there", model: str = "m1") -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_primary_provider_wins_by_priority():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    stub.set_ok("cheap", "hello from cheap")
    client, _clock, _sleeps = make_gateway(stub)

    response = client.post(CHAT_URL, headers=auth_headers(), json=chat_body())

    assert response.status_code == 200
    body = response.json()
    assert body["served_by"] == "frontier"
    assert body["text"] == "hello from frontier"
    assert body["cached"] is False
    assert body["usage"]["prompt_tokens"] == len("hello there") // 4
    assert body["usage"]["completion_tokens"] == len("hello from frontier") // 4
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert body["latency_ms"] >= 0
    assert stub.count("cheap") == 0


def test_failover_to_secondary_on_primary_errors():
    stub = UpstreamStub()
    stub.set_status("frontier", 500)
    stub.set_ok("cheap", "hello from cheap")
    client, _clock, sleeps = make_gateway(stub, max_retries=1)

    response = client.post(CHAT_URL, headers=auth_headers(), json=chat_body())

    assert response.status_code == 200
    body = response.json()
    assert body["served_by"] == "cheap"
    assert body["text"] == "hello from cheap"
    assert stub.count("frontier") == 2  # 1 initial attempt + 1 retry
    assert stub.count("cheap") == 1
    assert sleeps == [0.25]  # exponential backoff between retries


def test_provider_routing_by_model():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    stub.set_ok("cheap", "hello from cheap")
    cheap_only = (
        PROVIDERS[0],
        Provider(name="cheap", base_url="https://cheap.test/v1", api_key_env="CHEAP_API_KEY", models=("m3",), priority=2),
    )
    client, _clock, _sleeps = make_gateway(stub, providers=cheap_only)

    response = client.post(CHAT_URL, headers=auth_headers(), json=chat_body(model="m3"))

    assert response.status_code == 200
    assert response.json()["served_by"] == "cheap"
    assert stub.count("frontier") == 0  # frontier does not serve m3


def test_all_providers_down_returns_502():
    stub = UpstreamStub()
    stub.set_status("frontier", 500)
    stub.set_status("cheap", 503)
    client, _clock, _sleeps = make_gateway(stub, max_retries=0)

    response = client.post(CHAT_URL, headers=auth_headers(), json=chat_body())

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "upstream_unavailable"
    assert len(error["attempts"]) == 2
    assert stub.count("frontier") == 1 and stub.count("cheap") == 1


def test_unknown_model_returns_404():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    client, _clock, _sleeps = make_gateway(stub)

    response = client.post(CHAT_URL, headers=auth_headers(), json=chat_body(model="m9"))

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "unknown_model"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_breaker_trips_skips_and_heals():
    stub = UpstreamStub()
    seen = {"count": 0}

    def frontier(request: httpx.Request) -> httpx.Response:
        seen["count"] += 1
        if seen["count"] <= 3:
            return httpx.Response(500)
        return _chat_response("hello from frontier")

    stub.set_handler("frontier", frontier)
    stub.set_ok("cheap", "hello from cheap")
    client, clock, sleeps = make_gateway(stub)  # threshold=3, cooldown=30s, retries=2

    # Unique bodies per request so the response cache never interferes.
    first = client.post(CHAT_URL, headers=auth_headers(), json=chat_body("request-1")).json()
    assert first["served_by"] == "cheap"
    assert stub.count("frontier") == 3  # 3 attempts trip the breaker
    assert sleeps == [0.25, 0.5]

    # Breaker is OPEN: the primary is skipped instantly (no new upstream calls).
    second = client.post(CHAT_URL, headers=auth_headers(), json=chat_body("request-2")).json()
    assert second["served_by"] == "cheap"
    assert stub.count("frontier") == 3
    frontier_row = next(p for p in client.get("/admin/status").json()["providers"] if p["name"] == "frontier")
    assert frontier_row["state"] == "open"
    assert frontier_row["consecutive_failures"] == 3

    # After the cooldown the breaker goes HALF_OPEN; the probe succeeds -> CLOSED.
    clock.advance(30.001)
    third = client.post(CHAT_URL, headers=auth_headers(), json=chat_body("request-3")).json()
    assert third["served_by"] == "frontier"
    assert stub.count("frontier") == 4

    frontier_row = next(p for p in client.get("/admin/status").json()["providers"] if p["name"] == "frontier")
    assert frontier_row["state"] == "closed"
    assert frontier_row["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_serves_repeat_requests():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    client, _clock, _sleeps = make_gateway(stub, cache_ttl_seconds=60)

    first = client.post(CHAT_URL, headers=auth_headers(), json=chat_body("hello there")).json()
    second = client.post(CHAT_URL, headers=auth_headers(), json=chat_body("hello there")).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["text"] == first["text"] == "hello from frontier"
    assert second["served_by"] == "frontier"
    assert stub.count("frontier") == 1  # upstream hit exactly once

    third = client.post(CHAT_URL, headers=auth_headers(), json=chat_body("different")).json()
    assert third["cached"] is False
    assert stub.count("frontier") == 2

    stats = client.get("/admin/status").json()["cache"]
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 2


# ---------------------------------------------------------------------------
# Rate limiting and budget
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_and_recovers():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    client, clock, _sleeps = make_gateway(stub, rate_capacity=1, rate_refill_per_second=0.5)

    assert client.post(CHAT_URL, headers=auth_headers(), json=chat_body()).status_code == 200

    limited = client.post(CHAT_URL, headers=auth_headers(), json=chat_body())
    assert limited.status_code == 429
    assert limited.json()["error"]["type"] == "rate_limit_exceeded"

    clock.advance(2.0)  # 0.5 tokens/s * 2s -> one token back
    assert client.post(CHAT_URL, headers=auth_headers(), json=chat_body()).status_code == 200


def test_budget_returns_402_when_exhausted():
    stub = UpstreamStub()
    stub.set_ok("frontier", "a" * 12, usage={"prompt_tokens": 10, "completion_tokens": 1000})
    client, _clock, _sleeps = make_gateway(stub, keys=_keys(budget=0.001))

    assert client.post(CHAT_URL, headers=auth_headers(), json=chat_body()).status_code == 200

    # Expected spend: 10/1000 * $0.002 + 1000/1000 * $0.006 = $0.00602 > $0.001.
    over = client.post(CHAT_URL, headers=auth_headers(), json=chat_body())
    assert over.status_code == 402
    error = over.json()["error"]
    assert error["type"] == "budget_exceeded"
    assert "budget" in error["message"].lower()

    key_row = client.get("/admin/status").json()["keys"][TEST_KEY_PREFIX]
    assert math.isclose(key_row["usd_spent"], 0.00602, abs_tol=1e-9)
    assert key_row["budget_remaining"] == 0.0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_auth_rejects_missing_and_invalid_keys():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    client, _clock, _sleeps = make_gateway(stub)  # allow_anon=False

    cases = [
        client.post(CHAT_URL, json=chat_body()),  # no header
        client.post(CHAT_URL, headers=auth_headers("gw-test-wrong-secret"), json=chat_body()),  # prefix ok, digest bad
        client.post(CHAT_URL, headers=auth_headers("gw-other-0123456789abcdef"), json=chat_body()),  # unknown prefix
        client.post(CHAT_URL, headers={"Authorization": "Basic dXNlcjpwYXNz"}, json=chat_body()),  # wrong scheme
    ]
    for response in cases:
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "invalid_api_key"

    assert client.post(CHAT_URL, headers=auth_headers(), json=chat_body()).status_code == 200


def test_allow_anon_permits_keyless_requests():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    client, _clock, _sleeps = make_gateway(stub, allow_anon=True)

    anonymous = client.post(CHAT_URL, json=chat_body())
    assert anonymous.status_code == 200
    assert anonymous.json()["served_by"] == "frontier"

    # Anonymous mode does not weaken explicit-key checking.
    assert client.post(CHAT_URL, headers=auth_headers("gw-test-wrong"), json=chat_body()).status_code == 401


def test_upstream_receives_bearer_key_from_env_name():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    client, _clock, _sleeps = make_gateway(stub)

    assert client.post(CHAT_URL, headers=auth_headers(), json=chat_body()).status_code == 200
    assert stub.auth_headers["frontier"] == f"Bearer {FRONTIER_ENV_KEY}"


# ---------------------------------------------------------------------------
# Dashboard, admin, validation, config
# ---------------------------------------------------------------------------


def test_dashboard_and_admin_status():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    client, _clock, _sleeps = make_gateway(stub)
    client.post(CHAT_URL, headers=auth_headers(), json=chat_body())

    for path in ("/", "/dashboard"):
        page = client.get(path)
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert "frontier" in page.text
        assert "cheap" in page.text

    status = client.get("/admin/status").json()
    assert [p["name"] for p in status["providers"]] == ["frontier", "cheap"]
    frontier_row = status["providers"][0]
    assert frontier_row["state"] == "closed"
    assert frontier_row["requests"] == 1
    assert frontier_row["failures"] == 0

    assert status["cache"] == {"hits": 0, "misses": 1, "size": 1, "hit_rate": 0.0}

    key_row = status["keys"][TEST_KEY_PREFIX]
    assert key_row["plan"] == "pro"
    assert key_row["usd_budget"] == 5.0
    assert key_row["usd_spent"] > 0.0
    assert key_row["budget_remaining"] < 5.0

    assert client.post("/admin/cache/clear").json()["cleared"] is True
    assert client.get("/admin/status").json()["cache"]["size"] == 0


def test_invalid_body_returns_400():
    stub = UpstreamStub()
    stub.set_ok("frontier", "hello from frontier")
    client, _clock, _sleeps = make_gateway(stub)

    cases = [
        client.post(CHAT_URL, headers={**auth_headers(), "Content-Type": "application/json"}, content="not json"),
        client.post(CHAT_URL, headers=auth_headers(), json=[1, 2, 3]),
        client.post(CHAT_URL, headers=auth_headers(), json={"model": "m1"}),  # no messages
        client.post(CHAT_URL, headers=auth_headers(), json={"model": "m1", "messages": [{"role": "user"}]}),
    ]
    for response in cases:
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request"


def test_config_parsing_from_env():
    env = {
        "GATEWAY_PROVIDERS": json.dumps(
            [{"name": "cheap", "base_url": "https://cheap.test/v1", "api_key_env": "CHEAP_API_KEY", "models": ["m3"], "priority": 2}]
        ),
        "GATEWAY_KEYS": json.dumps({"gw-x": {"sha256": "a" * 64, "plan": "free", "usd_budget": 2.0}}),
        "GATEWAY_PRICES": json.dumps({"cheap": {"prompt_per_1k": 0.001, "completion_per_1k": 0.002}}),
        "GATEWAY_ALLOW_ANON": "1",
    }
    config = GatewayConfig.from_env(env)

    assert config.providers[0].name == "cheap"
    assert config.allow_anon is True
    assert config.keys["gw-x"].usd_budget == 2.0
    assert math.isclose(config.prices.cost("cheap", 1000, 1000), 0.003, abs_tol=1e-12)
    # Unknown providers fall back to the default price.
    assert math.isclose(config.prices.cost("unknown", 1000, 1000), 0.002 + 0.006, abs_tol=1e-12)
