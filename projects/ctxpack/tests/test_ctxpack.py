"""ctxpack tests — budget guarantees, algorithm behavior, CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ctxpack import Budget, Doc, DocError, pack_density, pack_exact, pack_greedy, render

PY = sys.executable
PACKERS = [pack_greedy, pack_density, pack_exact]


class LCG:
    """Deterministic pseudo-random for test fixtures."""
    def __init__(self, seed: int = 20260923):
        self.s = seed
    def next(self):
        self.s = (self.s * 1103515245 + 12345) % (1 << 31)
        return self.s
    def below(self, n: int):
        return self.next() % n


def lc_docs(n: int, seed: int = 7):
    rng = LCG(seed)
    out = []
    for i in range(n):
        length = 10 + rng.below(200)
        out.append(Doc(f"d{i}", "w" * length, 1.0 + rng.below(90) / 10))
    return out


# ---------- budget guarantee ----------
@pytest.mark.parametrize("algo", PACKERS)
def test_budget_never_exceeded(algo):
    b = Budget(chars=300, header_chars=20, per_doc_overhead=30)
    for seed in (1, 2, 3):
        pk = algo(lc_docs(20, seed), b)
        assert pk.used_chars <= b.usable(), f"{algo.__name__} exceeded budget"


@pytest.mark.parametrize("algo", PACKERS)
def test_empty_input_yields_empty_packing(algo):
    pk = algo([], Budget(chars=500))
    assert pk.included == [] and pk.used_chars == 0 and pk.total_score == 0


# ---------- greedy ----------
def test_greedy_takes_highest_scores_first():
    b = Budget(chars=160, per_doc_overhead=30)   # 60/doc: two fit, the third drops
    docs = [Doc("low", "x" * 30, 1.0), Doc("high", "x" * 30, 9.0), Doc("mid", "x" * 30, 5.0)]
    pk = pack_greedy(docs, b)
    assert [d.id for d in pk.included] == ["high", "mid"]


# ---------- density ----------
def test_density_skips_huge_low_density_keeps_small_dense():
    b = Budget(chars=150, per_doc_overhead=10)   # huge (160) no longer fits
    huge = Doc("huge", "z" * 150, 9.0)          # 0.06 per char
    dense = Doc("dense", "q" * 30, 8.0)          # 0.27 per char
    pk = pack_density([huge, dense], b)
    assert [d.id for d in pk.included] == ["dense"]
    assert any(doc.id == "huge" and reason in ("too_large_skipped", "too_large_alone")
               for doc, reason in pk.dropped)


# ---------- exact knapsack ----------
def test_exact_never_loses_to_greedy():
    for seed in (11, 22, 33):
        docs = lc_docs(12, seed)
        b = Budget(chars=400, per_doc_overhead=20)
        assert pack_exact(docs, b).total_score >= pack_greedy(docs, b).total_score - 1e-9


def test_exact_chooses_optimal_on_teaching_case():
    b = Budget(chars=400, header_chars=30, per_doc_overhead=30)
    docs = [Doc("a", "short doc", 5.0), Doc("b", "x" * 300, 8.0),
            Doc("c", "another doc here", 4.0), Doc("d", "dense", 6.0)]
    pk = pack_exact(docs, b)
    assert {d.id for d in pk.included} == {"a", "c", "d"}
    assert pk.total_score == 15.0


def test_ties_break_by_id():
    b = Budget(chars=28, per_doc_overhead=12)   # room for exactly ONE doc
    docs = [Doc("zz", "same", 5.0), Doc("aa", "same", 5.0)]
    pk = pack_exact(docs, b)
    assert len(pk.included) == 1
    assert [d.id for d in pk.included] == ["aa"]   # equal score → smaller id wins


# ---------- reasons / overhead / utilization ----------
@pytest.mark.parametrize("algo", PACKERS)
def test_every_drop_has_named_reason(algo):
    b = Budget(chars=150, per_doc_overhead=20)
    pk = algo(lc_docs(15, 5), b)
    for doc, reason in pk.dropped:
        assert reason in ("budget_exhausted", "too_large_alone", "too_large_skipped", "not_selected")


def test_per_doc_overhead_counts_toward_usage():
    b = Budget(chars=50, per_doc_overhead=20)   # text 10 + overhead 20 = 30/doc
    docs = [Doc("one", "x" * 10, 1.0), Doc("two", "y" * 10, 1.0)]
    pk = pack_greedy(docs, b)
    assert len(pk.included) == 1 and pk.used_chars == 30


def test_utilization_math():
    b = Budget(chars=100, per_doc_overhead=0)
    pk = pack_greedy([Doc("a", "x" * 60, 1.0)], b)
    assert pk.utilization == 0.6


def test_render_contains_header_and_docs():
    b = Budget(chars=500)
    docs = [Doc("a", "alpha text", 2.0), Doc("b", "beta text", 3.0)]
    out = render(pack_greedy(docs, b), header="ANSWER WITH THESE.")
    assert "ANSWER WITH THESE." in out and "alpha text" in out and "beta text" in out


# ---------- validation ----------
def test_doc_and_budget_validation():
    with pytest.raises(DocError):
        Doc("", "text", 1.0)
    with pytest.raises(DocError):
        Doc("x", "text", float("nan"))
    with pytest.raises(DocError):
        Budget(chars=100, header_chars=200)


# ---------- CLI ----------
def test_cli_demo_and_pack(tmp_path: Path):
    assert subprocess.run([PY, "-m", "ctxpack", "demo"], capture_output=True).returncode == 0
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps([{"id": "a", "text": "alpha " * 10, "score": 3.0},
                                {"id": "b", "text": "beta", "score": 1.5}]))
    r = subprocess.run([PY, "-m", "ctxpack", "pack", "--docs", str(docs),
                        "--chars", "200", "--algorithm", "exact"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "alpha" in r.stdout
