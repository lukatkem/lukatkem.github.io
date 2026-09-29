from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from .attestation import AttestationError, Attestor, ProviderProfile, ProviderQuarantined
from .integrity import ReplayCache, sign_and_check
from .scan import ScanVerdict, scan_request

@dataclass
class Provider:
    name: str; url: str; weight: int = 1
    headers: dict = field(default_factory=dict)
    profile: ProviderProfile = field(default_factory=ProviderProfile)

@dataclass
class RouteResult:
    provider: str; response_body: dict; latency_ms: float
    scan: ScanVerdict; attested: bool; logged: list = field(default_factory=list)

class HardenedRouter:
    def __init__(self, providers: list[Provider], secret: bytes,
                 attestor: Attestor | None = None, replay_cache: ReplayCache | None = None):
        self.providers = providers; self.secret = secret
        self.attestor = attestor or Attestor()
        self.replay_cache = replay_cache or ReplayCache()
        self._next = 0; self.log: list[dict] = []

    def _pick(self, exclude: set[str]) -> Provider:
        available = [p for p in self.providers if p.name not in exclude]
        if not available: raise RuntimeError("all providers quarantined or unavailable")
        total = sum(p.weight for p in available)
        self._next = (self._next + 1) % total
        acc = 0
        for p in available:
            acc += p.weight
            if self._next < acc: return p
        return available[-1]

    def route(self, body: dict, timestamp: float | None = None) -> RouteResult:
        body_bytes = json.dumps(body).encode()
        ts = int(timestamp or time.time())
        fresh, reason = self.replay_cache.sign_and_check(body_bytes, self.secret, ts)
        if not fresh: raise RuntimeError(f"request rejected: {reason}")
        scan_result = scan_request(body.get("prompt", body.get("message", "")))
        if scan_result.verdict is ScanVerdict.BLOCKED:
            raise RuntimeError(f"request blocked by router scan: {scan_result.matched_rules}")
        exclude: set[str] = set(); last_error = None
        for _ in range(len(self.providers)):
            provider = self._pick(exclude)
            start = time.time()
            try: raw = self._forward(provider, body_bytes)
            except Exception as e:
                exclude.add(provider.name); last_error = f"{provider.name}: {e}"
                self.log.append({"provider": provider.name, "error": str(e)}); continue
            latency = (time.time() - start) * 1000
            try: parsed = self.attestor.attest(provider.name, raw, latency, len(body_bytes))
            except AttestationError as e:
                exclude.add(provider.name); last_error = str(e)
                self.log.append({"provider": provider.name, "attestation_error": str(e)}); continue
            except ProviderQuarantined as e:
                exclude.add(provider.name); last_error = str(e)
                self.log.append({"provider": provider.name, "quarantined": True}); continue
            self.log.append({"provider": provider.name, "status": "ok",
                             "latency_ms": round(latency, 1), "scan": scan_result.verdict.value})
            return RouteResult(provider=provider.name, response_body=parsed,
                               latency_ms=round(latency, 1), scan=scan_result, attested=True)
        raise RuntimeError(f"all providers failed — last error: {last_error}")

    def _forward(self, provider: Provider, body_bytes: bytes) -> bytes:
        return body_bytes
