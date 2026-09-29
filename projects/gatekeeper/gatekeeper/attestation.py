"""Response attestation — provider sanity checks with quarantine."""
from __future__ import annotations
import json
from dataclasses import dataclass, field

class AttestationError(RuntimeError):
    def __init__(self, rule: str, detail: str):
        super().__init__(f"[{rule}] {detail}"); self.rule = rule

class ProviderQuarantined(RuntimeError):
    def __init__(self, provider: str, evidence: list[str]):
        super().__init__(f"provider {provider!r} quarantined: {len(evidence)} consecutive violations")
        self.provider = provider; self.evidence = evidence

@dataclass
class ProviderProfile:
    typical_body_keys: set = field(default_factory=lambda: {"choices", "model"})
    min_tokens_per_char: float = 0.0
    max_tokens_per_char: float = 1.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 300000.0
    quarantine_threshold: int = 3

class Attestor:
    def __init__(self, profile: ProviderProfile | None = None):
        self.profile = profile or ProviderProfile()
        self._violations: dict[str, list[str]] = {}
        self._quarantined: set[str] = set()

    def attest(self, provider: str, body_bytes: bytes, latency_ms: float,
               input_chars: int = 0) -> dict:
        if provider in self._quarantined:
            raise ProviderQuarantined(provider, self._violations.get(provider, []))
        try: body = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            self._record(provider, "malformed_body")
            raise AttestationError("malformed_body", f"{provider}: body is not valid JSON")
        violations = []
        missing = self.profile.typical_body_keys - set(body.keys())
        if missing: violations.append(f"missing_keys:{sorted(missing)}")
        usage = body.get("usage", {})
        total_tokens = (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
        if input_chars > 0 and total_tokens > 0:
            ratio = total_tokens / max(input_chars, 1)
            if ratio < self.profile.min_tokens_per_char or ratio > self.profile.max_tokens_per_char:
                violations.append(f"token_usage_sanity:{ratio:.3f}")
        if latency_ms > self.profile.max_latency_ms:
            violations.append(f"latency_outlier:{latency_ms:.0f}ms")
        if not body_bytes.strip(): violations.append("empty_body")
        if violations:
            self._record(provider, violations[0].split(":")[0])
            if len(self._violations.get(provider, [])) >= self.profile.quarantine_threshold:
                self._quarantined.add(provider)
                raise ProviderQuarantined(provider, self._violations[provider])
            raise AttestationError(violations[0].split(":")[0], f"{provider}: {violations[0]}")
        self._violations.pop(provider, None)
        return body

    def _record(self, provider: str, rule: str) -> None:
        self._violations.setdefault(provider, []).append(rule)
    def is_quarantined(self, provider: str) -> bool:
        return provider in self._quarantined
    def release(self, provider: str) -> None:
        self._quarantined.discard(provider); self._violations.pop(provider, None)
