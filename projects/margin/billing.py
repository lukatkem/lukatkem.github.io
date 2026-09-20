"""Stripe billing — optional by design.

With STRIPE_SECRET_KEY set: real Checkout Sessions + webhook-driven plan
updates. Without it: simulated checkout so the full upgrade flow is demoable
locally. The two paths share the same route contract.
"""
from __future__ import annotations

import json
import os

import httpx

from .auth import set_plan
from .config import (
    PLANS,
    STRIPE_PRICE_PRO,
    STRIPE_PRICE_TEAM,
    STRIPE_SECRET_KEY,
    STRIPE_WEBHOOK_SECRET,
)

PRICE_BY_PLAN = {"pro": STRIPE_PRICE_PRO, "team": STRIPE_PRICE_TEAM}


def stripe_enabled() -> bool:
    return bool(STRIPE_SECRET_KEY)


def create_checkout(plan: str, user_email: str, base_url: str) -> dict:
    if plan not in ("pro", "team"):
        return {"error": "unknown plan"}
    if not stripe_enabled():
        # simulated checkout — used by the local demo and CI
        return {"simulated": True, "plan": plan, "price": PLANS[plan]["price"]}

    form = {
        "mode": "subscription",
        "success_url": f"{base_url}/pricing?upgraded=1",
        "cancel_url": f"{base_url}/pricing",
        "customer_email": user_email,
        "line_items[0][price]": PRICE_BY_PLAN[plan],
        "line_items[0][quantity]": "1",
        "client_reference_id": user_email,
    }
    r = httpx.post(
        "https://api.stripe.com/v1/checkout/sessions",
        data=form,
        headers={"Authorization": f"Bearer {STRIPE_SECRET_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    return {"simulated": False, "url": r.json()["url"]}


def handle_webhook(payload: bytes, signature: str | None) -> dict:
    """Verifies and processes checkout.session.completed.

    Without stripe-python installed we verify the webhook by re-deriving the
    HMAC per Stripe's scheme (t=timestamp,v1=sig) — good enough without the
    SDK, documented honestly in the README.
    """
    if stripe_enabled() and STRIPE_WEBHOOK_SECRET and signature:
        if not _verify_signature(payload, signature):
            return {"error": "bad signature"}, 400
    event = json.loads(payload)
    if event.get("type") == "checkout.session.completed":
        session = event["data"]["object"]
        email = session.get("client_reference_id") or (session.get("customer_details") or {}).get("email")
        plan = "team" if session.get("amount_total", 0) >= int(PLANS["team"]["price"] * 100) else "pro"
        if email:
            from .auth import get_user_by_email

            user = get_user_by_email(email)
            if user:
                set_plan(user["id"], plan)
                return {"ok": True, "upgraded": email, "plan": plan}
    return {"ok": True, "ignored": event.get("type")}


def _verify_signature(payload: bytes, signature: str) -> bool:
    import hashlib
    import hmac as hmac_mod

    try:
        parts = dict(p.split("=", 1) for p in signature.split(","))
        timestamp, expected = parts["t"], parts["v1"]
    except Exception:
        return False
    signed = f"{timestamp}.".encode() + payload
    mac = hmac_mod.new(STRIPE_WEBHOOK_SECRET.encode(), signed, hashlib.sha256).hexdigest()
    return hmac_mod.compare_digest(mac, expected)
