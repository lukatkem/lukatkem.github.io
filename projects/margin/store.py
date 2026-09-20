"""SQLite storage: chunks, vectors, traces, users, keys, usage.

One database file, WAL mode. All access goes through this module so tests can
point MARGIN_DB at a temp file.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
  path TEXT PRIMARY KEY,
  title TEXT,
  hash TEXT,
  ingested_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  doc TEXT REFERENCES docs(path) ON DELETE CASCADE,
  heading TEXT,
  ord INTEGER,
  text TEXT,
  n_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc);
CREATE TABLE IF NOT EXISTS vectors (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  kind TEXT,
  dim INTEGER,
  vec BLOB
);
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  email TEXT UNIQUE,
  password_hash TEXT,
  plan TEXT DEFAULT 'free',
  stripe_customer_id TEXT,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
  token TEXT PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS api_keys (
  prefix TEXT PRIMARY KEY,
  key_hash TEXT,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  plan TEXT DEFAULT 'free',
  revoked INTEGER DEFAULT 0,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY,
  user_id INTEGER,
  api_key_prefix TEXT,
  route TEXT,
  ts REAL,
  month TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_month ON usage(month, api_key_prefix);
CREATE TABLE IF NOT EXISTS traces (
  id INTEGER PRIMARY KEY,
  ts REAL,
  user_id INTEGER,
  route TEXT,
  query TEXT,
  latency_ms REAL,
  retrieval_json TEXT,
  model TEXT,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  cost_usd REAL,
  status TEXT
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(str(db_path or DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def now() -> float:
    return time.time()


def month_key(ts: float | None = None) -> str:
    return time.strftime("%Y-%m", time.gmtime(ts or time.time()))


def new_token(prefix_len: int = 12) -> str:
    return uuid.uuid4().hex
