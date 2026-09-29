"""The full unattended cycle: probe → plan → per-domain distill → concatenate →
retrain → re-probe. This is the closed loop, driven against the real Academy API
with a mockable client for tests."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .academy import AcademyClient, AcademyError
from .loop import ImprovementLoop
from .planner import Plan


@dataclass
class CycleResult:
    iteration: int
    plan_mode: str
    distill_runs: list          # [(domain, stories, out_file)]
    concatenated: bool
    retrain_triggered: bool
    re_probed: bool
    new_scores: dict = field(default_factory=dict)


class LoopCloser:
    """Closes prometheus's plan into the Academy: per-domain distill runs,
    concatenation into the master textbook, retrain trigger, re-probe."""

    def __init__(self, loop: ImprovementLoop, academy: AcademyClient | None = None,
                 textbook_dir: str | None = None):
        self.loop = loop
        self.academy = academy
        self.textbook_dir = Path(textbook_dir) if textbook_dir else None

    def close_cycle(self, report, steps: int = 2500, size: str = "base") -> CycleResult:
        """Execute the plan of a completed iteration against the real Academy."""
        if not self.academy:
            raise AcademyError("no academy client configured")
        if not report.plan or not report.plan.entries:
            raise AcademyError("plan has no entries — nothing to distill")
        result = CycleResult(iteration=report.n_iteration, plan_mode=report.plan.mode,
                             distill_runs=[], concatenated=False, retrain_triggered=False,
                             re_probed=False)
        # per-domain distill runs, weakest domain first (the plan's order)
        for entry in report.plan.entries:
            out = f"distilled-{entry.domain}.txt"
            resp = self.academy.start_distill(entry.domain, len(entry.target_texts), out)
            if not resp.get("started"):
                raise AcademyError(f"distill for {entry.domain} refused: {resp.get('reason')}")
            result.distill_runs.append((entry.domain, len(entry.target_texts), out))
            # wait for this domain's run to complete before starting the next
            self.academy.wait_for_completion(on_poll=lambda s: None)
        # concatenate the per-domain textbooks into the master file
        if self.textbook_dir:
            master = self.textbook_dir / "distilled.txt"
            parts = []
            for _, _, out in result.distill_runs:
                p = self.textbook_dir / out
                if p.exists():
                    parts.append(p.read_text(encoding="utf-8", errors="ignore"))
            master.write_text("\n\n".join(parts), encoding="utf-8")
            result.concatenated = True
        # trigger the retrain
        self.academy.start_train(steps=steps, size=size)
        result.retrain_triggered = True
        self.loop.mark_trained(report.n_iteration)
        return result
