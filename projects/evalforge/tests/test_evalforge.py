"""evalforge tests — generation, filters, dedupe, persistence, margin adapter."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalforge import EvalSet, QualityFilter, generate_pairs

PY = sys.executable
from evalforge.generate import Pair

FAQ = ("The return policy is 30 days with a receipt. The store has 3 branches in Tbilisi. "
       "Sale items are final after 14 days.")
SECURITY = ("Two-factor authentication is an extra layer of security beyond the password. "
            "A firewall filters network traffic.")


def test_definition_extraction():
    pairs = generate_pairs("d", FAQ)
    kinds = {p.kind for p in pairs}
    assert "definition" in kinds
    assert any("return policy" in p.question.lower() for p in pairs)


def test_verb_agreement_and_numeric_routing():
    pairs = generate_pairs("d", FAQ)
    kinds = {p.kind for p in pairs}
    assert "numeric" in kinds                       # "is 30 days…" routes to numeric
    for q, p in ((p.question, p) for p in pairs):
        if "Sale items" in q:
            assert q.startswith("What are ")        # plural verb carried into the question
        if p.kind == "numeric":
            assert q.startswith("How many ")


def test_numeric_extraction():
    pairs = generate_pairs("d", FAQ)
    nums = [p for p in pairs if p.kind == "numeric"]
    assert any(p.answer == "30" for p in nums)


def test_cloze_generation():
    pairs = generate_pairs("d", "Lighthouse keeper Sam walked along the cliff path every evening.")
    cloze = [p for p in pairs if p.kind == "cloze"]
    assert cloze and "_____" in cloze[0].question and cloze[0].answer == "Lighthouse"


def test_determinism():
    a = generate_pairs("d", FAQ + " " + SECURITY)
    b = generate_pairs("d", FAQ + " " + SECURITY)
    assert [(p.kind, p.question) for p in a] == [(p.kind, p.question) for p in b]


def test_quality_filter_rejects_short_questions():
    flt = QualityFilter(min_question_len=200)
    assert generate_pairs("d", FAQ, flt) == []


def test_answer_leak_rejected():
    flt = QualityFilter()
    failed = flt.accept(Pair("d", "cloze", "What is x y?", "x", "src"))
    assert "answer_leaks_into_question" in failed


def test_dedup_on_add():
    es = EvalSet(name="s")
    pairs = generate_pairs("d", FAQ)
    assert es.add(pairs) == len(pairs)
    assert es.add(pairs) == 0   # all duplicates


def test_save_load_round_trip(tmp_path: Path):
    es = EvalSet(name="s")
    es.add(generate_pairs("d", FAQ))
    p = tmp_path / "set.json"
    es.save(p)
    es2 = EvalSet.load(p)
    assert es2.name == "s" and len(es2.pairs) == len(es.pairs)
    assert es2.to_json()["cases"][0]["question"] == es.to_json()["cases"][0]["question"]


def test_margin_adapter_shape():
    es = EvalSet(name="s")
    es.add(generate_pairs("d", FAQ))
    cases = es.to_margin_cases()
    assert cases and set(cases[0]) == {"question", "expected", "type"}


def test_cli_demo_end_to_end():
    assert subprocess.run([PY, "-m", "evalforge", "demo"], capture_output=True).returncode == 0


def test_cli_generate_writes_file(tmp_path: Path):
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps([{"id": "faq", "text": FAQ}]))
    out = tmp_path / "set.json"
    r = subprocess.run([PY, "-m", "evalforge", "generate", "--docs", str(docs),
                        "--name", "t", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and out.exists()
    data = json.loads(out.read_text())
    assert data["n"] >= 1 and data["cases"]
