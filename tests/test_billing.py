"""Stripe webhook verification (HMAC path) and simulated checkout."""
import hashlib
import hmac as hmac_mod
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def billing(monkeypatch, tmp_path):
    monkeypatch.setenv("MARGIN_DB", str(tmp_path / "bill.db"))
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    import importlib

    import margin.config as config
    importlib.reload(config)
    import margin.store as store
    importlib.reload(store)
    import margin.auth as auth
    importlib.reload(auth)
    import margin.billing as billing
    importlib.reload(billing)
    return billing, auth


def _sign(payload: bytes, secret: str) -> str:
    t = str(int(time.time()))
    mac = hmac_mod.new(secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
    return f"t={t},v1={mac}"


def test_webhook_verifies_signature_and_upgrades(billing):
    billing_mod, auth = billing
    uid = auth.create_user("s@example.com", "password123")
    payload = json.dumps({
        "type": "checkout.session.completed",
        "data": {"object": {"client_reference_id": "s@example.com", "amount_total": 1900}},
    }).encode()
    sig = _sign(payload, "whsec_test")
    result = billing_mod.handle_webhook(payload, sig)
    assert result["upgraded"] == "s@example.com"
    assert auth.get_user_by_email("s@example.com")["plan"] == "pro"


def test_webhook_rejects_bad_signature(billing):
    billing_mod, _ = billing
    payload = b"{}"
    result = billing_mod.handle_webhook(payload, "t=1,v1=deadbeef")
    assert result == ({"error": "bad signature"}, 400)


def test_simulated_checkout_without_keys(billing, monkeypatch):
    billing_mod, auth = billing
    monkeypatch.setenv("STRIPE_SECRET_KEY", "")
    import importlib

    import margin.config as config
    importlib.reload(config)
    importlib.reload(billing_mod)
    out = billing_mod.create_checkout("pro", "x@y.z", "http://localhost:8000")
    assert out["simulated"] is True
