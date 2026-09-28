"""prometheus tests — scoring, ranking, planning, the loop, state, backends."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prometheus.curriculum import DOMAINS, textbook_library
from prometheus.loop import ImprovementLoop, LoopComplete
from prometheus.model_backends import HttpModel, MockModel, ModelUnavailable
from prometheus.planner import Planner, TeacherAvailability, make_plan
from prometheus.prober import ProbeRunner, Assessment
from prometheus.report import render_html


def make_model(script_overrides: dict | None = None, default: str = "",
               weak_domains: list[str] | None = None) -> MockModel:
    """Perfect model by default; weak_domains answer with keyword-less filler."""
    base = {}
    for domain, probes in DOMAINS.items():
        for p in probes:
            base[p.prompt] = "" if domain in (weak_domains or []) else " ".join(p.keywords)
    base.update(script_overrides or {})
    return MockModel(base, default=default)


def strong_everywhere() -> Assessment:
    return ProbeRunner(make_model()).run()


# ---------- probing / scoring ----------
def test_perfect_answers_score_1():
    a = strong_everywhere()
    assert all(score == 1.0 for score in a.domain_scores.values())


def test_scoring_is_case_insensitive_and_lenient():
    a = ProbeRunner(make_model(
        {"Explain what a firewall does.": "A FIREWALL BLOCKS bad Network traffic"})).run(
        ["cybersecurity"])
    assert a.domain_scores["cybersecurity"] > 0.5


def test_empty_reply_scores_zero():
    t = ProbeRunner(make_model()).probe_reply(DOMAINS["cybersecurity"][0])
    assert ProbeRunner(make_model()).score_reply(DOMAINS["cybersecurity"][0], "") == 0.0
    a = ProbeRunner(make_model(weak_domains=["cybersecurity"])).run(["cybersecurity"])
    assert a.domain_scores["cybersecurity"] == 0.0


def test_ranking_weakest_first():
    a = ProbeRunner(make_model(weak_domains=["cybersecurity"])).run()
    assert a.ranked[0][1] <= a.ranked[-1][1]
    assert a.weakest == a.ranked[0]


def test_curriculum_has_four_domains_of_twelve():
    assert len(DOMAINS) == 4
    assert all(len(probes) == 12 for probes in DOMAINS.values())


# ---------- planner ----------
def test_weak_domains_get_double_quota():
    a = ProbeRunner(make_model({"Explain what a firewall does.": ""})).run()
    planner = Planner(base_texts=2)
    plan = planner.plan(a)
    quotas = {e.domain: len(e.target_texts) for e in plan.entries}
    ranked_domains = [d for d, _ in a.ranked]
    assert quotas[ranked_domains[0]] == 4            # weakest → 2×2
    assert quotas[ranked_domains[-1]] == 2           # strongest → base


def test_plan_mode_and_reasons():
    a = strong_everywhere()
    plan = Planner(base_texts=2, teachers=TeacherAvailability(available=True)).plan(a)
    assert plan.mode == "textbooks"
    study = Planner(base_texts=2).plan(a)
    assert study.mode == "study"
    assert all(r for r in plan.reason_per_domain.values())


def test_make_plan_wrapper_and_library_seeds():
    a = ProbeRunner(make_model({"Explain what a firewall does.": ""})).run()
    plan = make_plan(a, base_texts=2)
    lib = textbook_library()
    for e in plan.entries:
        for seed in e.target_texts:
            assert seed in lib[e.domain]


# ---------- the loop ----------
def test_loop_records_iterations_and_trend():
    model = make_model(weak_domains=["cybersecurity"])          # one weak domain
    loop = ImprovementLoop(model, state_path=None, max_iterations=5)
    loop.run_iteration()
    loop.run_iteration()
    assert len(loop.history) == 2
    trend = loop.trend()
    assert trend["cybersecurity"] == [loop.history[0]["domain_scores"]["cybersecurity"],
                                      loop.history[1]["domain_scores"]["cybersecurity"]]


def test_loop_raises_when_complete():
    loop = ImprovementLoop(make_model(), state_path=None, max_iterations=1)
    loop.run_iteration()
    with pytest.raises(LoopComplete):
        loop.run_iteration()


def test_stop_reason_target_reached():
    loop = ImprovementLoop(make_model(), state_path=None, max_iterations=5, target=0.9)
    loop.run_iteration()                                # perfect model → 1.0 everywhere
    stop, reason = loop.should_stop()
    assert stop and "target" in reason


def test_loop_run_iterates_until_complete():
    loop = ImprovementLoop(make_model(weak_domains=["cybersecurity"]),
                           state_path=None, max_iterations=2)
    reports = loop.run()
    assert len(reports) == 2


def test_state_persists_and_resumes(tmp_path: Path):
    sp = tmp_path / "state.json"
    loop1 = ImprovementLoop(make_model(weak_domains=["cybersecurity"]),
                            state_path=sp, max_iterations=2)
    loop1.run_iteration()
    loop2 = ImprovementLoop(make_model(weak_domains=["cybersecurity"]),
                            state_path=sp, max_iterations=2)
    assert len(loop2.history) == 1
    loop2.run_iteration()
    loop2.save()
    data = json.loads(sp.read_text())
    assert data["iterations"][-1]["n"] == 2


def test_mark_trained_updates_history(tmp_path: Path):
    sp = tmp_path / "state.json"
    loop = ImprovementLoop(make_model(), state_path=sp, max_iterations=2)
    rep = loop.run_iteration()
    loop.mark_trained(rep.n_iteration)
    assert loop.history[0]["trained"] is True


def test_state_records_model_name(tmp_path: Path):
    sp = tmp_path / "state.json"
    loop = ImprovementLoop(make_model(), state_path=sp, max_iterations=1)
    loop.run_iteration()
    data = json.loads(sp.read_text())
    assert "MockModel" in data["model"]


# ---------- backends ----------
def test_mock_model_script_and_default():
    m = MockModel({"q1": "a1"}, default="d")
    assert m.complete("q1") == "a1" and m.complete("zzz") == "d"


def test_http_model_request_shape():
    captured = {}

    def fake_transport(url, data, timeout):
        captured["url"] = url
        captured["data"] = json.loads(data)
        return json.dumps({"text": "a reply"}).encode()

    m = HttpModel(base_url="http://x", path="/api/arena", transport=fake_transport)
    assert m.complete("hello") == "a reply"
    assert captured["url"] == "http://x/api/arena"
    assert captured["data"]["prompt"] == "hello"


def test_http_model_raises_unavailable():
    def broken(url, data, timeout):
        raise ConnectionError("refused")
    m = HttpModel(transport=broken)
    with pytest.raises(ModelUnavailable):
        m.complete("hi")


# ---------- report ----------
def test_report_html_has_domains_and_bars(tmp_path: Path):
    sp = tmp_path / "state.json"
    loop = ImprovementLoop(make_model(), state_path=sp, max_iterations=2)
    loop.run_iteration()
    html = render_html(loop)
    assert "prometheus" in html and "<svg" in html and "cybersecurity" in html
    assert "<rect" in html and "iteration" in html
