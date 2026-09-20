"""Auth + metering: users, sessions, API keys, plan quotas.

Zero-dependency security: scrypt password hashing (stdlib), random tokens,
hashed API keys (only the prefix is stored for lookup/display).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from .config import PLANS
from .store import connect, month_key, new_token, now

# ---- passwords (scrypt) ----


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, dk_hex = stored.split("$")
        dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


# ---- users & sessions ----


def create_user(email: str, password: str, plan: str = "free") -> int:
    conn = connect()
    cur = conn.execute(
        "INSERT INTO users(email,password_hash,plan,created_at) VALUES(?,?,?,?)",
        (email.lower(), hash_password(password), plan, now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_user_by_email(email: str):
    return connect().execute(
        "SELECT * FROM users WHERE email = ?", (email.lower(),)
    ).fetchone()


def login(email: str, password: str) -> str | None:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    token = new_token()
    conn = connect()
    conn.execute(
        "INSERT INTO sessions(token,user_id,created_at) VALUES(?,?,?)",
        (token, user["id"], now()),
    )
    conn.commit()
    return token


def user_for_session(token: str):
    if not token:
        return None
    return connect().execute(
        """SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
           WHERE s.token = ?""",
        (token,),
    ).fetchone()


# ---- API keys ----


def create_api_key(user_id: int, plan: str) -> tuple[str, str]:
    """Returns (full_key, prefix). Full key shown once; store hash only."""
    raw = secrets.token_hex(16)
    prefix = raw[:8]
    full = f"mgn_{raw}"
    conn = connect()
    conn.execute(
        "INSERT INTO api_keys(prefix,key_hash,user_id,plan,created_at) VALUES(?,?,?,?,?)",
        (prefix, hashlib.sha256(raw.encode()).hexdigest(), user_id, plan, now()),
    )
    conn.commit()
    return full, prefix


def _key_secret(key: str) -> str | None:
    if not key.startswith("mgn_") or len(key) != 4 + 32:
        return None
    return key[4:]


def user_for_api_key(key: str):
    secret = _key_secret(key)
    if not secret:
        return None
    h = hashlib.sha256(secret.encode()).hexdigest()
    return connect().execute(
        """SELECT u.*, k.prefix AS key_prefix, k.plan AS key_plan
           FROM api_keys k JOIN users u ON u.id = k.user_id
           WHERE k.key_hash = ? AND k.revoked = 0""",
        (h,),
    ).fetchone()


# ---- metering & quotas ----


def meter(user_id: int | None, prefix: str | None, route: str) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO usage(user_id,api_key_prefix,route,ts,month) VALUES(?,?,?,?,?)",
        (user_id, prefix, route, now(), month_key()),
    )
    conn.commit()


def month_usage(prefix: str) -> int:
    return connect().execute(
        "SELECT COUNT(*) c FROM usage WHERE api_key_prefix = ? AND month = ?",
        (prefix, month_key()),
    ).fetchone()["c"]


def quota_for(plan: str) -> int:
    return PLANS.get(plan, PLANS["free"])["quota"]


def quota_state(prefix: str, plan: str) -> dict:
    used = month_usage(prefix)
    quota = quota_for(plan)
    return {"used": used, "quota": quota, "remaining": max(quota - used, 0)}


def set_plan(user_id: int, plan: str) -> None:
    assert plan in PLANS, f"unknown plan {plan}"
    conn = connect()
    conn.execute("UPDATE users SET plan = ? WHERE id = ?", (plan, user_id))
    conn.execute("UPDATE api_keys SET plan = ? WHERE user_id = ?", (plan, user_id))
    conn.commit()


# ---- seed ----


def seed_demo() -> dict:
    """Idempotently create the demo account and a demo API key."""
    if not get_user_by_email("demo@margin.local"):
        uid = create_user("demo@margin.local", "demo1234", plan="pro")
        key, prefix = create_api_key(uid, "pro")
        return {"user_id": uid, "api_key": key}
    return {}
