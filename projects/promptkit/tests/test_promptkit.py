"""promptkit tests — templates, rendering, assertions, suite runner."""
from __future__ import annotations

import pytest

from promptkit import (AssertionFailed, Case, Prompt, PromptError, Suite, parse,
                       contains, max_length, not_contains, regex)
from promptkit.llm import MockLLM


def make_prompt():
    return Prompt(name="p", template="Answer about {topic} for {audience}.",
                  slots={"topic": "str", "audience": "str"})


# ---------- template / slots ----------
def test_undeclared_slot_raises():
    with pytest.raises(PromptError):
        Prompt(name="p", template="hi {who}")


def test_unused_declared_slot_raises():
    with pytest.raises(PromptError):
        Prompt(name="p", template="static text", slots={"who": "str"})


def test_missing_kwarg_raises():
    p = make_prompt()
    with pytest.raises(PromptError):
        p.render(topic="robots")


def test_type_enforced():
    p = Prompt(name="p", template="n is {n}", slots={"n": "int"})
    with pytest.raises(PromptError):
        p.render(n="five")


def test_render_substitutes():
    assert make_prompt().render(topic="robots", audience="kids") == "Answer about robots for kids."


# ---------- assertions ----------
def test_contains_and_not_contains():
    contains("hello")("say HELLO now")
    with pytest.raises(AssertionFailed):
        not_contains("hello")("say HELLO now")


def test_max_length_and_regex():
    max_length(5)("abcd")
    with pytest.raises(AssertionFailed):
        max_length(3)("abcd")
    regex(r"\d+")("order 4821")
    with pytest.raises(AssertionFailed):
        regex(r"\d+")("no digits")


def test_non_callable_assertion_rejected():
    with pytest.raises(PromptError):
        parse(["not callable"])


# ---------- suite runner ----------
def test_suite_passes_with_mock():
    p = make_prompt()
    llm = MockLLM(["Answer about robots for kids."])
    report = (Suite(p, llm)
              .case("basic", {"topic": "robots", "audience": "kids"}, [contains("robots")])
              .run())
    assert report.ok and report.passed == 1


def test_suite_catches_bad_reply():
    p = make_prompt()
    llm = MockLLM(["totally unrelated answer"])
    report = (Suite(p, llm)
              .case("basic", {"topic": "robots", "audience": "kids"}, [contains("robots")])
              .run())
    assert not report.ok and report.failed == 1
    assert "contains" in report.results[0].failures[0]


def test_llm_crash_counts_as_failure_not_exception():
    class Exploding:
        def complete(self, messages):
            raise RuntimeError("boom")
    p = make_prompt()
    report = (Suite(p, Exploding())
              .case("basic", {"topic": "robots", "audience": "kids"}, [contains("robots")])
              .run())
    assert not report.ok and "boom" in report.results[0].failures[0]


def test_mock_exhaustion_raises_loudly():
    llm = MockLLM(["only one"])
    llm.complete([])
    with pytest.raises(RuntimeError):
        llm.complete([])


def test_case_chaining_returns_suite():
    p = make_prompt()
    llm = MockLLM(["x", "y"])
    s = Suite(p, llm)
    assert s.case("a", {"topic": "t", "audience": "a"}, []) is s
