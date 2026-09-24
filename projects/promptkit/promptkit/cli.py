"""python -m promptkit demo — a prompt regression suite, offline."""
from __future__ import annotations

from .asserts import contains, max_length, not_contains, regex
from .llm import MockLLM
from .prompt import Prompt
from .runner import Suite


def main() -> int:
    summary_prompt = Prompt(
        name="summarizer",
        template="Summarize the following support ticket in one sentence.\n\nTicket: {ticket}\nFocus on: {focus}.",
        version="1.2.0",
        slots={"ticket": "str", "focus": "str"},
    )

    # a mock "model" whose replies would FAIL the old prompt's assertions —
    # the suite is what catches a bad prompt edit before it ships
    llm = MockLLM([
        "Customer cannot log in; password reset did not arrive. Wants access restored today.",
        "Refund requested for the damaged order #4821. Emotionally upset.",
    ])

    suite = (Suite(summary_prompt, llm)
             .case("login issue",
                   {"ticket": "I cannot log in, the reset email never arrives.",
                    "focus": "the login problem"},
                   [contains("log in"), not_contains("refund"), max_length(200)])
             .case("refund issue",
                   {"ticket": "My order #4821 arrived damaged, I want a refund.",
                    "focus": "the refund"},
                   [contains("refund"), regex(r"#\d+"), max_length(200)]))

    report = suite.run()
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {r.case}")
        for f in r.failures:
            print(f"       {f}")
    print(f"\n{report.passed}/{report.passed + report.failed} cases passed "
          f"— prompt {summary_prompt.name} v{summary_prompt.version} is safe to ship")
    return 0 if report.ok else 1
