"""The loop: measure → rank → plan → persist → (externally retrain) → again.

Each run_iteration() probes the student, plans from the assessment, and
appends one entry to the state JSON (iterations so far + per-iteration domain
scores) so the loop resumes across processes and the report can chart the
trend. The distill + retrain steps belong to hydro-1's Academy — the loop
records the commissioned texts and `mark_trained()` flags when they landed.
Stop conditions: every domain >= target (default 0.6), or max_iterations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .model_backends import Base
from .planner import Plan, Planner
from .prober import Assessment, ProbeRunner

STATE_VERSION = 1


class LoopComplete(RuntimeError):
    """Raised by run_iteration() when a stop condition is already satisfied."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class IterationReport:
    """What one pass of the loop produced."""
    n_iteration: int
    assessment: Assessment
    plan: Plan
    started_at: str

    def summary(self) -> str:
        weakest, score = self.assessment.weakest
        return (f"iteration {self.n_iteration}: weakest domain {weakest} ({score:.2f}) — "
                f"{self.plan.mode} plan targets {self.plan.total_texts} texts")


class ImprovementLoop:
    """The self-improvement loop around any Base model and any Planner."""

    def __init__(self, model: Base, planner: Planner | None = None,
                 max_iterations: int = 3, target: float = 0.6,
                 state_path: str | Path | None = None, runner: ProbeRunner | None = None):
        self.model = model
        self.planner = planner or Planner()
        self.max_iterations = max_iterations
        self.target = target
        self.state_path = Path(state_path) if state_path else None
        self.runner = runner or ProbeRunner(model)
        self.history: list[dict] = self._load()

    # ---------- state ------------------------------------------------------

    def _load(self) -> list[dict]:
        if self.state_path and self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return list(data.get("iterations", []))
        return []

    def save(self) -> None:
        if not self.state_path:
            return
        payload = {
            "version": STATE_VERSION,
            "model": self._model_name(),
            "target": self.target,
            "max_iterations": self.max_iterations,
            "iterations": self.history,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    def _model_name(self) -> str:
        name = type(self.model).__name__
        url = getattr(self.model, "base_url", "")
        return f"{name}({url})" if url else name

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------- the loop ----------------------------------------------------

    def should_stop(self) -> tuple[bool, str]:
        """(stop?, reason) — target reached on every domain, or out of iterations."""
        if len(self.history) >= self.max_iterations:
            return True, f"max_iterations reached ({self.max_iterations})"
        if self.history:
            last = self.history[-1]["domain_scores"]
            if last and all(score >= self.target for score in last.values()):
                return True, f"target reached: all domains >= {self.target:.2f}"
        return False, ""

    def run_iteration(self) -> IterationReport:
        """One pass: probe → plan → append to state → save. Raises LoopComplete
        if a stop condition was already met before this pass."""
        stop, reason = self.should_stop()
        if stop:
            raise LoopComplete(reason)
        n = len(self.history) + 1
        started_at = self._now()
        assessment = self.runner.run()
        plan = self.planner.plan(assessment)
        self.history.append({
            "n": n,
            "started_at": started_at,
            "domain_scores": assessment.domain_scores,
            "ranked": [list(pair) for pair in assessment.ranked],
            "weakest": assessment.weakest[0],
            "plan_mode": plan.mode,
            "plan": [{"domain": e.domain, "texts": list(e.target_texts)}
                     for e in plan.entries],
            "reasons": dict(plan.reason_per_domain),
            "trained": False,
        })
        self.save()
        return IterationReport(n_iteration=n, assessment=assessment, plan=plan,
                               started_at=started_at)

    def run(self) -> list[IterationReport]:
        """Iterate until a stop condition; returns the reports of this call."""
        reports = []
        while not self.should_stop()[0]:
            reports.append(self.run_iteration())
        return reports

    def mark_trained(self, n: int) -> None:
        """Called after the Academy finishes the targeted distill + retrain."""
        for it in self.history:
            if it["n"] == n:
                it["trained"] = True
        self.save()

    # ---------- trends -------------------------------------------------------

    def trend(self) -> dict[str, list[float]]:
        """domain → [score per iteration] for charting."""
        out: dict[str, list[float]] = {}
        for it in self.history:
            for domain, score in it["domain_scores"].items():
                out.setdefault(domain, []).append(score)
        return out

    def assessments_history(self) -> list[tuple[int, dict[str, float]]]:
        """[(iteration n, {domain: score}), …] — the report's input shape."""
        return [(it["n"], dict(it["domain_scores"])) for it in self.history]
