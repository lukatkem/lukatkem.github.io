"""Academy client — prometheus drives the sibling hydro1 distill/train pipeline.

The plan's domain weights become per-domain distill runs with separate output
files; the concatenated textbook then feeds the retrain. Fully mockable.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


class AcademyError(RuntimeError):
    pass


@dataclass
class AcademyClient:
    base_url: str = "http://127.0.0.1:8001"
    timeout: float = 30.0
    transport: object = None   # injectable: callable(method, url, payload_bytes) -> bytes

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = self._urllib

    @staticmethod
    def _urllib(method: str, url: str, data: bytes | None) -> bytes:
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    def _post(self, path: str, payload: dict) -> dict:
        try:
            raw = self.transport("POST", self.base_url + path, json.dumps(payload).encode())
            return json.loads(raw)
        except Exception as e:                                  # noqa: BLE001
            raise AcademyError(f"{path}: {e}") from e

    def _get(self, path: str) -> dict:
        try:
            raw = self.transport("GET", self.base_url + path, None)
            return json.loads(raw)
        except Exception as e:                                  # noqa: BLE001
            raise AcademyError(f"{path}: {e}") from e

    def status(self) -> dict:
        return self._get("/api/academy/job")

    def start_distill(self, domain: str, stories: int, out: str) -> dict:
        return self._post("/api/academy/distill",
                          {"stories": stories, "domain": domain, "out": out, "chain": False})

    def start_train(self, steps: int = 2500, size: str = "base") -> dict:
        return self._post("/api/academy/train", {"steps": steps, "size": size})

    def wait_for_completion(self, poll_seconds: float = 30, max_wait_hours: float = 8,
                            on_poll=None) -> dict:
        """Block until the academy job finishes. on_poll(status) is called each poll."""
        import time
        deadline = time.time() + max_wait_hours * 3600
        last = {}
        while time.time() < deadline:
            last = self.status()
            if not last.get("active"):
                return last
            if on_poll:
                on_poll(last)
            time.sleep(poll_seconds)
        raise AcademyError("timed out waiting for the academy job")
