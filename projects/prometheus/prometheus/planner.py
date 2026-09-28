"""Planner: turn an Assessment into a targeted study plan.

The two weakest domains get 2× the texts of the strongest — the model studies
what it is worst at. When a teacher API is available the plan commissions a
targeted textbook; when not, the same seeds are handed over as a self-study
list only. Deterministic: same assessment in, same plan out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .curriculum import textbook_library
from .prober import Assessment


@dataclass
class TeacherAvailability:
    """Whether a teacher API can be commissioned to write the textbook."""
    available: bool = False
    note: str = "no teacher API configured"


@dataclass
class PlanEntry:
    """One domain's slice of the plan: its seeds, its score, its rank."""
    domain: str
    score: float
    rank: int
    target_texts: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """Ordered study plan — weakest domain first."""
    entries: list[PlanEntry]
    reason_per_domain: dict[str, str]
    mode: str = "study"          # "textbooks" (commissioned) | "study" (seed list)
    teacher_note: str = ""

    @property
    def total_texts(self) -> int:
        return sum(len(e.target_texts) for e in self.entries)

    def domains(self) -> list[str]:
        return [e.domain for e in self.entries]

    def texts_for(self, domain: str) -> list[str]:
        for e in self.entries:
            if e.domain == domain:
                return list(e.target_texts)
        return []


class Planner:
    """Assessment (+ teacher availability) → deterministic Plan."""

    def __init__(self, teachers: TeacherAvailability | None = None,
                 texts: dict[str, list[str]] | None = None,
                 base_texts: int = 2, target: float = 0.6):
        self.teachers = teachers or TeacherAvailability()
        self.texts = texts or textbook_library()
        self.base_texts = base_texts     # the strongest domain's quota
        self.target = target

    def plan(self, assessment: Assessment) -> Plan:
        ranked = assessment.ranked
        weakest_two = {domain for domain, _ in ranked[:2]}
        strongest_score = ranked[-1][1] if ranked else 0.0
        entries: list[PlanEntry] = []
        reasons: dict[str, str] = {}
        for rank, (domain, score) in enumerate(ranked, start=1):
            quota = self.base_texts * 2 if domain in weakest_two else self.base_texts
            seeds = list(self.texts.get(domain, []))[:quota]
            action = ("commission targeted texts" if self.teachers.available
                      else "self-study seed list")
            standing = "below" if score < self.target else "at/above"
            reasons[domain] = (f"rank {rank}/{len(ranked)} · score {score:.2f} "
                               f"({standing} target {self.target:.2f}) · vs strongest "
                               f"{strongest_score:.2f} → {action} ({quota})")
            entries.append(PlanEntry(domain=domain, score=score, rank=rank,
                                     target_texts=seeds))
        mode = "textbooks" if self.teachers.available else "study"
        return Plan(entries=entries, reason_per_domain=reasons, mode=mode,
                    teacher_note=self.teachers.note)


def make_plan(assessment: Assessment, base_texts: int = 2,
              teachers: TeacherAvailability | None = None) -> Plan:
    """Convenience wrapper: Planner(teachers, base_texts=base_texts).plan(assessment)."""
    return Planner(teachers=teachers, base_texts=base_texts).plan(assessment)
