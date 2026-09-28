"""Model backends: the abstract student, the real Hydro-1 server, and a mock.

`Base` is the one-method interface the prober needs: complete(prompt) -> str.
HttpModel speaks hydro-1's generation API — read from hydro1/hydro1/server.py:
POST /api/generate with {"prompt", "temperature", "top_k", "max_new_tokens"}
and a {"text": …} reply. The request/response mapping lives in two small
methods (build_payload / extract_text) so a different server is a two-line
change. Any connection failure degrades to ModelUnavailable instead of
crashing the loop.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class ModelError(Exception):
    """The model replied, but with something we could not use."""


class ModelUnavailable(ModelError):
    """The model server could not be reached (down, wrong port, timeout)."""


class Base:
    """The one method a student must provide."""

    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class MockModel(Base):
    """Scripted replies keyed by prompt; unknown prompts get a default.

    `script` is held by live reference: tests and the demo mutate it between
    loop iterations to simulate the model studying its commissioned textbook.
    `calls` records every prompt asked (the prober hits all of them)."""

    def __init__(self, script: dict[str, str], default: str = ""):
        self.script = script
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.script.get(prompt, self.default)


class HttpModel(Base):
    """Talks to the Hydro-1 server's generation endpoint over HTTP.

    transport is injectable for tests: any callable(url, data, timeout) that
    returns the raw body (str or bytes). Connection errors degrade to
    ModelUnavailable so a dead server fails soft instead of killing the run."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001",
                 path: str = "/api/generate", timeout: float = 20.0,
                 temperature: float = 0.2, top_k: int = 40,
                 max_new_tokens: int = 140,
                 payload_builder=None, extract=None, transport=None):
        self.base_url = base_url.rstrip("/")
        self.path = path
        self.timeout = timeout
        self.temperature = temperature      # low: probing wants recall, not flourish
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self._payload_builder = payload_builder or self.build_payload
        self._extract = extract or self.extract_text
        self._transport = transport or self._urllib_post   # injectable for tests

    # --- the two small mapping methods to adjust for another server -------

    def build_payload(self, prompt: str) -> bytes:
        """hydro-1 GenIn shape — see hydro1/hydro1/server.py."""
        return json.dumps({"prompt": prompt, "temperature": self.temperature,
                           "top_k": self.top_k,
                           "max_new_tokens": self.max_new_tokens}).encode("utf-8")

    @staticmethod
    def extract_text(body: dict) -> str:
        """hydro-1 replies {"text": …}."""
        if isinstance(body, dict) and "text" in body:
            return str(body["text"])
        raise ModelError(f"unexpected reply shape: {body!r}")

    # --- transport ---------------------------------------------------------

    @staticmethod
    def _urllib_post(url: str, data: bytes, timeout: float) -> bytes:
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def complete(self, prompt: str) -> str:
        url = self.base_url + self.path
        try:
            raw = self._transport(url, self._payload_builder(prompt), self.timeout)
            body = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        except ModelUnavailable:
            raise
        except ModelError:
            raise
        except Exception as e:                                  # noqa: BLE001
            raise ModelUnavailable(f"model at {self.base_url} unreachable: {e}") from e
        return self._extract(body)


def _urllib_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def teacher_status(base_url: str, timeout: float = 5.0,
                   send=None) -> tuple[bool, str]:
    """Ask the hydro-1 server whether a teacher API is configured.

    GET {base_url}/api/teachers → {"cloud_configured": …, "teachers": […]}.
    Soft-fails to (False, …) — teacher probing must never kill an assess run.
    `send` is injectable for tests: callable(url, timeout) -> bytes."""
    send = send or _urllib_get
    url = base_url.rstrip("/") + "/api/teachers"
    try:
        body = json.loads(send(url, timeout))
    except Exception:                                           # noqa: BLE001
        return False, "teacher status unknown (server unreachable)"
    configured = bool(body.get("cloud_configured"))
    note = "teacher API configured" if configured else "no teacher API configured"
    return configured, note
