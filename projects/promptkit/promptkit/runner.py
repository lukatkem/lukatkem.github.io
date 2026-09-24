"""Suite: prompt cases + assertions, run against any BaseLLM — offline with a mock."""
from __future__ import annotations

from dataclasses import dataclass, field

from .asserts import parse
from .errors import PromptError


@dataclass
class Case:
    name: str
    slots: dict                      # render kwargs
    assertions: list = field(default_factory=list)


@dataclass
class RunResult:
    case: str
    passed: bool
    reply: str
    failures: list


@dataclass
class SuiteReport:
    passed: int
    failed: int
    results: list

    @property
    def ok(self) -> bool:
        return self.failed == 0


class Suite:
    """Regression suite for ONE prompt across many cases."""

    def __init__(self, prompt, llm):
        self.prompt = prompt
        self.llm = llm
        self.cases: list[Case] = []

    def case(self, name: str, slots: dict, assertions: list) -> "Suite":
        self.cases.append(Case(name, slots, parse(assertions)))
        return self

    def run(self) -> SuiteReport:
        results = []
        for case in self.cases:
            rendered = self.prompt.render(**case.slots)
            try:
                reply = self.llm.complete([{"role": "user", "content": rendered}]).text
            except Exception as e:   # noqa: BLE001 — a crash IS a failed case
                results.append(RunResult(case.name, False, "", [f"llm error: {e}"]))
                continue
            failures = []
            for check in case.assertions:
                try:
                    check(reply)
                except AssertionError as e:
                    failures.append(str(e))
            results.append(RunResult(case.name, not failures, reply, failures))
        passed = sum(1 for r in results if r.passed)
        return SuiteReport(passed, len(results) - passed, results)
