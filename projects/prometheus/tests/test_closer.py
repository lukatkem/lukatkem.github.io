"""Loop-closer tests — the full cycle against a mock Academy, no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prometheus.academy import AcademyClient
from prometheus.curriculum import DOMAINS
from prometheus.closer import LoopCloser
from tests.mock_academy import MockAcademy
from prometheus.loop import ImprovementLoop
from prometheus.model_backends import MockModel
from prometheus.prober import ProbeRunner


def make_weak_model() -> MockModel:
    """A model with ~25% keyword coverage across all domains."""
    base = {}
    for domain, probes in DOMAINS.items():
        for p in probes:
            base[p.prompt] = " ".join(p.keywords[:1])
    return MockModel(base, default="")


def make_loop_with_plan(tmp_path=None):
    loop = ImprovementLoop(make_weak_model(), state_path=None,
                           max_iterations=3)
    rep = loop.run_iteration()
    return loop, rep


# ---------- closing the cycle ----------
def test_full_cycle_distills_per_domain(tmp_path):
    loop, rep = make_loop_with_plan(tmp_path)
    academy = MockAcademy()
    closer = LoopCloser(loop, academy=academy, textbook_dir=str(tmp_path / "tb"))
    Path(str(tmp_path / "tb")).mkdir(parents=True, exist_ok=True)
    result = closer.close_cycle(rep, steps=100, size="base")
    distill_domains = [d for d, _, _ in result.distill_runs]
    assert len(distill_domains) == len({e.domain for e in rep.plan.entries})
    assert result.retrain_triggered
    assert result.concatenated
    assert loop.history[0]["trained"] is True


def test_cycle_plan_order_weakest_first(tmp_path=None):
    loop, rep = make_loop_with_plan(None)
    academy = MockAcademy()
    result = LoopCloser(loop, academy=academy).close_cycle(rep)
    first_distill = result.distill_runs[0][0]
    assert first_distill == rep.assessment.ranked[0][0]


def test_cycle_without_plan_raises(tmp_path=None):
    loop, rep = make_loop_with_plan(None)
    empty_rep = type(rep)(n_iteration=1, assessment=rep.assessment, plan=None,
                          started_at=rep.started_at)
    with pytest.raises(Exception, match="plan has no entries"):
        LoopCloser(loop, academy=MockAcademy()).close_cycle(empty_rep)


def test_cycle_concatenates_textbook(tmp_path):
    loop, rep = make_loop_with_plan(tmp_path)
    td = tmp_path / "textbook"
    td.mkdir()
    for e in rep.plan.entries:
        (td / f"distilled-{e.domain}.txt").write_text(f"content for {e.domain}")
    academy = MockAcademy()
    result = LoopCloser(loop, academy=academy, textbook_dir=str(td)).close_cycle(rep)
    assert result.concatenated
    master = td / "distilled.txt"
    assert master.exists()
    content = master.read_text()
    for e in rep.plan.entries:
        assert f"content for {e.domain}" in content


def test_academy_error_propagates(tmp_path):
    loop, rep = make_loop_with_plan(tmp_path)

    class Refusing:
        def start_distill(self, domain, stories, out):
            return {"started": False, "reason": "job already running"}
    with pytest.raises(Exception, match="refused"):
        LoopCloser(loop, academy=Refusing()).close_cycle(rep)


def test_academy_client_post_shape():
    calls = []
    def fake_transport(method, url, data):
        calls.append((method, url, json.loads(data) if data else None))
        return json.dumps({"started": True}).encode()
    client = AcademyClient(transport=fake_transport)
    client.start_distill("cybersecurity", 50, "distilled-cyber.txt")
    method, url, payload = calls[0]
    assert method == "POST" and url.endswith("/api/academy/distill")
    assert payload["domain"] == "cybersecurity" and payload["stories"] == 50
