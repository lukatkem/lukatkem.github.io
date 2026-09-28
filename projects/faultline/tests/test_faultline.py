"""FaultLine tests — scheduler, faults, proxy transparency, verdicts, runner, CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from faultline.faults import FaultProfile, FaultType
from faultline.proxy import FaultyRegistry
from faultline.runner import ChaosRunner, RecoveryVerdict
from faultline.verdict import classify

PY = sys.executable


class FakeRegistry:
    """Minimal registry with the agentcore call surface."""
    def __init__(self):
        self.calls = []
    def call(self, name, kwargs=None, **extra):
        self.calls.append((name, kwargs or {}, extra))
        return {"result": 31, "status": "ok"} if name == "data" else {"ok": True}


# ---------- fault scheduler ----------
def test_scheduler_is_deterministic():
    a, b = FaultProfile({"t": [("timeout", 1.0)]}, seed=42), FaultProfile({"t": [("timeout", 1.0)]}, seed=42)
    seq_a = [a.pick("t") for _ in range(10)]
    seq_b = [b.pick("t") for _ in range(10)]
    assert seq_a == seq_b


def test_probability_one_always_injects():
    p = FaultProfile({"t": [("timeout", 1.0)]})
    assert all(p.pick("t") is FaultType.TIMEOUT for _ in range(10))


def test_probability_zero_never_injects():
    p = FaultProfile({"t": [("timeout", 0.0)]})
    assert all(p.pick("t") is None for _ in range(10))


def test_no_profile_means_no_faults():
    p = FaultProfile()
    assert p.pick("anything") is None


def test_log_records_decisions():
    p = FaultProfile({"t": [("timeout", 1.0)]})
    p.pick("t")
    assert p.log[-1] == ("t", "timeout", True)


# ---------- faulty proxy ----------
def test_pass_through_transparency():
    inner = FakeRegistry()
    wrapped = FaultyRegistry(inner, None)
    assert wrapped.call("data") == {"result": 31, "status": "ok"}
    assert len(inner.calls) == 1


def test_connection_error_raised():
    wrapped = FaultyRegistry(FakeRegistry(), FaultProfile({"data": [("connection_error", 1.0)]}))
    with pytest.raises(ConnectionError):
        wrapped.call("data")


def test_timeout_raised():
    wrapped = FaultyRegistry(FakeRegistry(), FaultProfile({"data": [("timeout", 1.0)]}))
    with pytest.raises(TimeoutError):
        wrapped.call("data")


def test_wrong_but_plausible_corrupts_numbers():
    wrapped = FaultyRegistry(FakeRegistry(), FaultProfile({"data": [("wrong_but_plausible", 1.0)]}))
    out = wrapped.call("data")
    assert out != {"result": 31, "status": "ok"}
    assert "86" in str(out)      # 31 flipped digit-wise


def test_malformed_json_breaks_structure():
    wrapped = FaultyRegistry(FakeRegistry(), FaultProfile({"data": [("malformed_json", 1.0)]}))
    out = wrapped.call("data")
    assert "CORRUPTED" in str(out)


def test_kwargs_pass_through():
    inner = FakeRegistry()
    wrapped = FaultyRegistry(inner, None)
    wrapped.call("data", {"k": 1}, confirm_dangerous=True)
    seen = inner.calls[-1]
    assert seen[1].get("k") == 1 and seen[1].get("confirm_dangerous") is True


# ---------- verdicts ----------
def test_correct_with_faults_on_keyword_and_digit_match():
    v = classify("8", "2+2*3 is 8.")
    assert v is RecoveryVerdict.CORRECT_WITH_FAULTS


def test_silent_failure_confident_wrong():
    v = classify("31", "The server count is 86 and everything is fine.")
    assert v is RecoveryVerdict.SILENT_FAILURE


def test_degraded_when_hedging():
    v = classify("31", "Sorry, I could not read the note because of an error.")
    assert v is RecoveryVerdict.DEGRADED


def test_crashed_on_empty_answer():
    v = classify("31", "")
    assert v is RecoveryVerdict.CRASHED


def test_unverifiable_without_expected():
    v = classify(None, "something else entirely")
    assert v is RecoveryVerdict.UNVERIFIABLE


# ---------- runner / demo ----------
def _demo_runner():
    sys_path = str(Path(__file__).resolve().parent.parent.parent / "faultline")
    from faultline.cli import _agent_factory, _registry_factory, _scenarios
    return ChaosRunner(_agent_factory, _registry_factory, _scenarios())


def test_demo_runner_recovery_rate_bounds():
    report = _demo_runner().run()
    assert 0.0 <= report.recovery_rate <= 1.0
    assert 0.0 <= report.silent_failure_rate <= 1.0
    assert report.recovery_rate + report.silent_failure_rate + report.degradation_rate <= 1.0


def test_baseline_all_correct_in_demo():
    report = _demo_runner().run()
    assert all(c.baseline_verdict is RecoveryVerdict.CORRECT_WITH_FAULTS for c in report.cases)


def test_demo_generates_html(tmp_path: Path):
    report = _demo_runner().run()
    from faultline.report import render_html
    html = render_html(report, caption="test")
    assert "FaultLine" in html and "silent" in html and "<svg" in html


def test_cli_demo_end_to_end():
    r = subprocess.run([PY, "-m", "faultline", "demo"], capture_output=True, text=True)
    assert r.returncode == 0 and "silent failures" in r.stdout
