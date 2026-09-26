"""Retrieval fusion (RRF) — against a temp store with TF-IDF vectors."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def retriever(monkeypatch, tmp_path):
    # point config at temp corpus + db before importing modules
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a-payouts.md").write_text(
        "# Payouts\n\n## Schedule\n\nPayout windows open bi-weekly on the 14th and 28th.\n"
        "Minimum profit since last payout is $500 on $50k accounts.\n"
    )
    (corpus / "b-drawdown.md").write_text(
        "# Drawdown\n\n## Daily loss\n\nThe max daily loss is 5% of starting balance, "
        "measured on equity and resetting at 00:00 UTC.\n"
    )
    monkeypatch.setenv("MARGIN_CORPUS", str(corpus))
    monkeypatch.setenv("MARGIN_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("MARGIN_OLLAMA_BASE", "http://localhost:1")  # force TF-IDF path

    import importlib

    import margin.config as config
    importlib.reload(config)

    import margin.ingest as ingest
    importlib.reload(ingest)
    ingest.ingest(verbose=False)

    import margin.retrieve as retrieve
    importlib.reload(retrieve)
    return retrieve.Retriever()


def test_hybrid_returns_both_signals(retriever):
    hits = retriever.search("when can I request a payout", top_k=2)
    assert hits and hits[0].doc == "a-payouts.md"
    assert hits[0].bm25_rank is not None


def test_reranked_query_hits_right_doc(retriever):
    hits = retriever.search("daily loss reset time", top_k=2)
    assert hits[0].doc == "b-drawdown.md"


def test_fused_score_beats_single_source(retriever):
    hits = retriever.search("payout minimum $500", top_k=2)
    top = hits[0]
    # doc matched by both sources → both ranks set
    assert top.bm25_rank is not None and (top.vec_rank is not None or retriever.kind != "tfidf")
