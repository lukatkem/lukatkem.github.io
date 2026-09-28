"""The chaos runner: every scenario twice — clean baseline, then faulted."""
from __future__ import annotations

from dataclasses import dataclass, field

from .faults import FaultProfile
from .verdict import RecoveryVerdict, classify


@dataclass
class CaseResult:
    scenario: str
    baseline_verdict: RecoveryVerdict
    faulted_verdict: RecoveryVerdict
    faults_injected: list[str]
    agent_answer: str


@dataclass
class ChaosReport:
    cases: list[CaseResult] = field(default_factory=list)
    recovery_rate: float = 0.0
    silent_failure_rate: float = 0.0
    degradation_rate: float = 0.0

    def summary(self) -> str:
        return (f"scenarios {len(self.cases)} · recovery {self.recovery_rate:.0%} · "
                f"silent failures {self.silent_failure_rate:.0%} · "
                f"degradation {self.degradation_rate:.0%}")


class ChaosRunner:
    def __init__(self, agent_factory, registry_factory, scenarios):
        """agent_factory(registry, scenario) → a fresh agent for that scenario.
        registry_factory(profile) → a (possibly faulty) registry.
        scenarios: dicts {name, message, llm_script, expected?, faults: {tool: [(type, prob), …]}}.
        Each scenario gets its own agent + script so replies never bleed across cases.
        """
        self.agent_factory = agent_factory
        self.registry_factory = registry_factory
        self.scenarios = scenarios

    def run(self) -> ChaosReport:
        report = ChaosReport()
        for sc in self.scenarios:
            profile = FaultProfile(sc.get("faults", {}), sc.get("global_rate", 0.0))
            # baseline: clean registry, clean script
            clean_agent = self.agent_factory(self.registry_factory(None), sc)
            base = clean_agent.run(sc["message"])
            base_verdict = classify(sc.get("expected"), base.answer)
            # faulted run: same script, faulty tool layer
            registry = self.registry_factory(profile)
            agent = self.agent_factory(registry, sc)
            try:
                result = agent.run(sc["message"])
                answer = result.answer
            except Exception as e:                     # noqa: BLE001 — a crash IS a verdict
                answer = f"crashed: {e}"
            injected = [f"{t}:{f}" for t, f, was in profile.log if was]
            faulted_verdict = classify(sc.get("expected"), answer)
            report.cases.append(CaseResult(sc["name"], base_verdict, faulted_verdict,
                                           injected, answer))
        faulted = [c for c in report.cases]
        n = len(faulted) or 1
        report.recovery_rate = sum(1 for c in faulted
                                   if c.faulted_verdict is RecoveryVerdict.CORRECT_WITH_FAULTS) / n
        report.silent_failure_rate = sum(1 for c in faulted
                                         if c.faulted_verdict is RecoveryVerdict.SILENT_FAILURE) / n
        report.degradation_rate = sum(1 for c in faulted
                                      if c.faulted_verdict is RecoveryVerdict.DEGRADED) / n
        return report
