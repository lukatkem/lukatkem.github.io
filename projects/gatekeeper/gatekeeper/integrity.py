"""Request integrity — HMAC signing, verification, replay cache, chain signing."""
from __future__ import annotations
import hashlib, hmac, time

def sign_request(body: bytes, secret: bytes, timestamp: int) -> str:
    payload = str(timestamp).encode() + b":" + body
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()

def verify_request(body: bytes, signature: str, secret: bytes, timestamp: int,
                   max_skew_seconds: int = 300) -> tuple[bool, str]:
    expected = sign_request(body, secret, timestamp)
    if not hmac.compare_digest(signature, expected):
        return False, "signature_mismatch"
    skew = abs(time.time() - timestamp)
    if skew > max_skew_seconds:
        return False, f"timestamp_skew_{skew:.0f}s"
    return True, "ok"

class ReplayCache:
    def __init__(self, ttl_seconds: float = 600):
        self.ttl = ttl_seconds; self._seen: dict[str, float] = {}
    def check_and_record(self, signature: str) -> tuple[bool, str]:
        now = time.time()
        expired = [k for k, t in self._seen.items() if now - t > self.ttl]
        for k in expired: del self._seen[k]
        if signature in self._seen: return False, "replay_detected"
        self._seen[signature] = now; return True, "ok"
    def sign_and_check(self, body: bytes, secret: bytes, timestamp: int) -> tuple[bool, str]:
        sig = sign_request(body, secret, timestamp)
        fresh, reason = self.check_and_record(sig)
        if not fresh: return False, reason
        return verify_request(body, sig, secret, timestamp)

def sign_and_check(body: bytes, secret: bytes, timestamp: int) -> tuple[bool, str]:
    sig = sign_request(body, secret, timestamp)
    return verify_request(body, sig, secret, timestamp)

def sign_chain(bodies: list[bytes], secret: bytes) -> list[str]:
    sigs = []; prev = b""
    for body in bodies:
        payload = prev + body
        sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        sigs.append(sig); prev = sig.encode()
    return sigs

def verify_chain(bodies: list[bytes], signatures: list[str], secret: bytes) -> tuple[bool, str]:
    prev = b""
    for i, (body, sig) in enumerate(zip(bodies, signatures)):
        payload = prev + body
        expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False, f"chain_broken_at_{i}"
        prev = sig.encode()
    return True, "ok"
