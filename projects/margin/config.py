"""Central configuration — env-overridable, zero required secrets to run."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = Path(os.environ.get("MARGIN_CORPUS", ROOT / "corpus"))
GOLDEN_PATH = Path(os.environ.get("MARGIN_GOLDEN", ROOT / "evals" / "golden.jsonl"))
WEB_DIR = ROOT / "web"
DB_PATH = Path(os.environ.get("MARGIN_DB", ROOT / "data" / "margin.db"))

# Local inference
OLLAMA_BASE = os.environ.get("MARGIN_OLLAMA_BASE", "http://localhost:11434")
CHAT_MODEL = os.environ.get("MARGIN_CHAT_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("MARGIN_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.environ.get("MARGIN_OLLAMA_TIMEOUT", "120"))

# Retrieval
CHUNK_TARGET = int(os.environ.get("MARGIN_CHUNK_TARGET", "900"))
CHUNK_OVERLAP = int(os.environ.get("MARGIN_CHUNK_OVERLAP", "150"))
BM25_CANDIDATES = int(os.environ.get("MARGIN_BM25_K", "20"))
VEC_CANDIDATES = int(os.environ.get("MARGIN_VEC_K", "20"))
RRF_K = int(os.environ.get("MARGIN_RRF_K", "60"))

# Evals — CI gate
EVAL_RECALL_THRESHOLD = float(os.environ.get("MARGIN_EVAL_RECALL", "0.80"))
EVAL_ANSWER_THRESHOLD = float(os.environ.get("MARGIN_EVAL_ANSWER", "0.75"))
# Structured eval snapshot served by GET /api/evals (written by save_report;
# lives inside the margin package so it ships with the code).
EVALS_REPORT_PATH = Path(
    os.environ.get("MARGIN_EVAL_REPORT", Path(__file__).resolve().parent / "evals" / "report.json")
)

# SaaS plans: (monthly query quota, price_usd)
PLANS = {
    "free": {"quota": 100, "price": 0.0},
    "pro": {"quota": 2000, "price": 19.0},
    "team": {"quota": 10000, "price": 79.0},
}
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "")
STRIPE_PRICE_TEAM = os.environ.get("STRIPE_PRICE_TEAM", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# Anonymous demo rate limit (queries/day/IP) so the product is try-able signed out
ANON_DAILY_LIMIT = int(os.environ.get("MARGIN_ANON_LIMIT", "10"))

ADMIN_TOKEN = os.environ.get("MARGIN_ADMIN_TOKEN", "")


def ensure_dirs() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
