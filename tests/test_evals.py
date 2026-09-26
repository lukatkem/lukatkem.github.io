"""Eval metric math + answer checking, plus GET /api/evals."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from margin.evals import check_answer, load_golden
from margin.config import GOLDEN_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_REPORT = REPO_ROOT / "margin" / "evals" / "report.json"


def test_check_answer_case_insensitive_all_terms():
    assert check_answer("Max daily loss is 5% of equity.", ["5%", "equity"])
    assert not check_answer("Max daily loss is 5% of equity.", ["5%", "balance"])
    assert check_answer("RESET AT 00:00 UTC", ["00:00 utc"])  # case


def test_golden_dataset_loads_and_is_well_formed():
    golden = load_golden(GOLDEN_PATH)
    assert len(golden) >= 60, "expect a substantial golden set"
    ids = [q["id"] for q in golden]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for q in golden:
        assert q["question"] and q["gold_doc"] and q["must_contain"]


# ---------- GET /api/evals (serves margin/evals/report.json) ----------


@pytest.fixture()
def api(monkeypatch, tmp_path):
    """TestClient against a temp DB with a real user + API key — same
    MARGIN_DB + module-reload pattern as the auth/billing fixtures."""
    monkeypatch.setenv("MARGIN_DB", str(tmp_path / "api.db"))
    import importlib

    import margin.config as config
    importlib.reload(config)
    import margin.store as store
    importlib.reload(store)
    import margin.auth as auth
    importlib.reload(auth)
    import margin.app as app_module

    uid = auth.create_user("evals@example.com", "password123")
    key, _prefix = auth.create_api_key(uid, "free")
    from fastapi.testclient import TestClient

    return TestClient(app_module.app), app_module, {"Authorization": f"Bearer {key}"}


def test_api_evals_requires_auth(api):
    client, _app_module, _headers = api
    assert client.get("/api/evals").status_code == 401
    # well-formed but unknown key is still unauthorized
    bogus = {"Authorization": "Bearer mgn_" + "0" * 32}
    assert client.get("/api/evals", headers=bogus).status_code == 401


def test_api_evals_missing_report_404_then_save_report_serves_200(api, monkeypatch, tmp_path):
    client, app_module, headers = api
    missing = tmp_path / "reports" / "report.json"
    monkeypatch.setattr(app_module, "EVALS_REPORT_PATH", missing)

    r = client.get("/api/evals", headers=headers)
    assert r.status_code == 404
    assert "make eval" in r.json()["detail"]

    # what `make eval` does: evals.save_report writes the structured JSON → 200
    import margin.evals as evals

    monkeypatch.setattr(evals, "EVALS_REPORT_PATH", missing)
    evals.save_report({
        "summary": {
            "n": 1, "recall_at_5": 1.0, "precision_at_3": 1.0, "mrr": 1.0,
            "median_latency_ms": 5.0, "answer_accuracy": 1.0,
            "thresholds": {"recall_at_5": 0.8, "answer_accuracy": 0.75},
            "passed": True,
        },
        "results": [],
    })
    r = client.get("/api/evals", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["example"] is False and body["generated_at"]
    assert body["summary"]["n"] == 1


def test_api_evals_serves_committed_example_report(api, monkeypatch):
    client, app_module, headers = api
    assert EXAMPLE_REPORT.exists(), "example snapshot ships so the endpoint works out of the box"
    monkeypatch.setattr(app_module, "EVALS_REPORT_PATH", EXAMPLE_REPORT)

    r = client.get("/api/evals", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["example"] is True  # committed snapshot is marked as an example
    summary = body["summary"]
    for metric in ("n", "recall_at_5", "precision_at_3", "mrr", "median_latency_ms",
                   "answer_accuracy", "by_type", "thresholds", "passed"):
        assert metric in summary
    assert summary["passed"] is True
    assert summary["n"] == len(body["results"])
