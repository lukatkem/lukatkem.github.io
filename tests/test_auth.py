"""Auth + metering: hashing, sessions, API keys, quotas."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def db(monkeypatch, tmp_path):
    monkeypatch.setenv("MARGIN_DB", str(tmp_path / "auth.db"))
    import importlib

    import margin.config as config
    importlib.reload(config)
    import margin.store as store
    importlib.reload(store)
    import margin.auth as auth
    importlib.reload(auth)
    return auth


def test_password_roundtrip(db):
    h = db.hash_password("correct horse battery")
    assert db.verify_password("correct horse battery", h)
    assert not db.verify_password("wrong", h)
    assert not db.verify_password("wrong", "garbage")


def test_user_session_flow(db):
    uid = db.create_user("t@example.com", "password123")
    token = db.login("t@example.com", "password123")
    assert token
    user = db.user_for_session(token)
    assert user["id"] == uid
    assert db.user_for_session("bogus") is None
    assert db.login("t@example.com", "wrong") is None


def test_api_key_auth_and_quota(db):
    uid = db.create_user("k@example.com", "password123", plan="pro")
    key, prefix = db.create_api_key(uid, "pro")
    user = db.user_for_api_key(key)
    assert user["id"] == uid and user["key_prefix"] == prefix

    # meter to the quota
    for _ in range(db.quota_for("pro")):
        db.meter(uid, prefix, "/api/chat")
    state = db.quota_state(prefix, "pro")
    assert state["remaining"] == 0 and state["used"] == state["quota"]

    # revoked key stops working
    import margin.store as store

    conn = store.connect()
    conn.execute("UPDATE api_keys SET revoked = 1 WHERE prefix = ?", (prefix,))
    conn.commit()
    assert db.user_for_api_key(key) is None


def test_set_plan_updates_keys(db):
    uid = db.create_user("p@example.com", "password123")
    key, prefix = db.create_api_key(uid, "free")
    db.set_plan(uid, "team")
    user = db.user_for_api_key(key)
    assert user["key_plan"] == "team"
