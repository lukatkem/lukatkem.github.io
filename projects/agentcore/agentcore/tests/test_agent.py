"""The agent loop, driven end to end by the scripted MockLLM — no network."""
import json

import pytest

from agentcore.agent import Agent, AgentResult, Step, ToolResult
from agentcore.llm import LLMResponse, MockLLM, ToolCallRequest
from agentcore.tools import Registry, ToolError


def make_registry(tmp_path):
    (tmp_path / "note.txt").write_text("hello from the sandbox", encoding="utf-8")
    return Registry.with_builtins(tmp_path)


def wire(name, **arguments):
    body = json.dumps({"name": name, "arguments": arguments}, indent=2)
    return f"```tool\n{body}\n```"


# -- the happy path -----------------------------------------------------------


def test_tool_call_then_final_answer(tmp_path):
    llm = MockLLM(
        [
            "I'll compute that.\n" + wire("calculator", expression="2+2*3"),
            "2+2*3 is 8.",
        ]
    )
    agent = Agent(registry=make_registry(tmp_path), llm=llm)
    result = agent.run("What is 2+2*3?")

    assert isinstance(result, AgentResult)
    assert result.answer == "2+2*3 is 8."
    assert result.tool_calls_made == 1
    assert not result.stopped
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.index == 0
    assert "```tool" in step.reply  # the raw reply is preserved
    assert step.tool_calls[0].name == "calculator"
    assert step.tool_results[0].ok
    assert step.tool_results[0].result == 8

    # the model saw the system message, the user message, then the tool result
    first_turn = llm.turns[0]["messages"]
    assert first_turn[0]["role"] == "system"
    assert first_turn[1] == {"role": "user", "content": "What is 2+2*3?"}
    second_turn = llm.turns[1]["messages"]
    assert second_turn[-1]["role"] == "tool"
    assert json.loads(second_turn[-1]["content"]) == {"ok": True, "result": 8}
    # and the schemas were handed over
    assert {t["name"] for t in llm.turns[0]["tools"]} == {"calculator", "text_stats", "read_note"}


def test_transparent_step_log_shape(tmp_path):
    llm = MockLLM(
        [
            wire("text_stats", text="one two three") + "\nalso:\n" + wire("calculator", expression="6*7"),
            "Both done: 3 words and 42.",
        ]
    )
    result = Agent(registry=make_registry(tmp_path), llm=llm).run("stats and math, please")
    assert result.tool_calls_made == 2
    assert len(result.steps) == 1
    step = result.steps[0]
    assert isinstance(step, Step)
    assert [c.name for c in step.tool_calls] == ["text_stats", "calculator"]
    assert [r.name for r in step.tool_results] == ["text_stats", "calculator"]
    assert all(isinstance(r, ToolResult) for r in step.tool_results)
    assert step.tool_results[0].result == {"words": 3, "chars": 13, "lines": 1}
    assert step.tool_results[1].result == 42
    for record in step.tool_results:
        assert record.ok and record.error == ""


# -- the guards -----------------------------------------------------------------


def test_max_steps_guard_triggers():
    tool_reply = "Still working...\n" + wire("calculator", expression="1+1")
    llm = MockLLM([tool_reply, tool_reply])  # never gives a final answer
    result = Agent(registry=Registry.with_builtins(), llm=llm, max_steps=2).run("loop forever")
    assert result.stopped is True
    assert result.answer == ""
    assert "max_steps" in result.reason
    assert len(result.steps) == 2
    assert result.tool_calls_made == 2
    # the loop stopped instead of asking for a third turn (which would raise)


def test_malformed_tool_call_is_recovered(tmp_path):
    llm = MockLLM(
        [
            "Sure.\n```tool\n{\"name\": \"calculator\", \"arguments\": {\"expression\": 2+2}}\n```",  # broken JSON
            "Let me fix that.\n" + wire("calculator", expression="2+2"),
            "2+2 is 4.",
        ]
    )
    result = Agent(registry=make_registry(tmp_path), llm=llm, max_steps=5).run("What is 2+2?")
    assert result.answer == "2+2 is 4."
    assert result.tool_calls_made == 1
    assert "invalid JSON" in result.steps[0].error  # the failed turn is on the record
    # the model was shown its mistake and corrected itself
    correction = llm.turns[1]["messages"][-1]["content"]
    assert "could not be parsed" in correction


def test_unknown_tool_error_is_fed_back(tmp_path):
    llm = MockLLM(
        [
            wire("multiplier", x=2, y=3),  # no such tool
            "That tool does not exist — the answer is 6 anyway.",
        ]
    )
    result = Agent(registry=make_registry(tmp_path), llm=llm).run("multiply 2 by 3")
    assert result.answer == "That tool does not exist — the answer is 6 anyway."
    assert result.tool_calls_made == 1
    outcome = result.steps[0].tool_results[0]
    assert not outcome.ok
    assert "unknown tool" in outcome.error
    fed_back = json.loads(llm.turns[1]["messages"][-1]["content"])
    assert fed_back["ok"] is False and "unknown tool" in fed_back["error"]


def test_dangerous_tool_refused_without_confirm(tmp_path):
    llm = MockLLM(
        [
            "Opening it.\n" + wire("read_note", path="/etc/passwd"),
            "Refused — I can only read notes inside the sandbox.",
        ]
    )
    result = Agent(registry=make_registry(tmp_path), llm=llm).run("read /etc/passwd")
    outcome = result.steps[0].tool_results[0]
    assert not outcome.ok
    assert "refused" in outcome.error
    assert result.answer == "Refused — I can only read notes inside the sandbox."


def test_dangerous_tool_allowed_with_confirm_callback(tmp_path):
    asked = []
    llm = MockLLM(
        [
            wire("read_note", path="note.txt"),
            "The note says: hello from the sandbox",
        ]
    )
    agent = Agent(
        registry=make_registry(tmp_path),
        llm=llm,
        confirm_dangerous=lambda name: asked.append(name) or True,
    )
    result = agent.run("read note.txt")
    assert asked == ["read_note"]  # the callback was consulted, with the tool name
    assert result.steps[0].tool_results[0].ok
    assert result.steps[0].tool_results[0].result == "hello from the sandbox"
    assert result.answer == "The note says: hello from the sandbox"


def test_structured_tool_calls_skip_the_parser(tmp_path):
    llm = MockLLM(
        [
            LLMResponse(text="", tool_calls=[ToolCallRequest("calculator", {"expression": "6*7"})]),
            LLMResponse(text="42."),
        ]
    )
    result = Agent(registry=make_registry(tmp_path), llm=llm).run("six times seven")
    assert result.answer == "42."
    assert result.steps[0].tool_results[0].result == 42


# -- the mock itself --------------------------------------------------------------


def test_mock_llm_consumes_script_in_order_and_then_fails():
    llm = MockLLM(["first", "second"])
    assert llm.complete([], []).text == "first"
    assert llm.complete([], []).text == "second"
    with pytest.raises(RuntimeError, match="script exhausted"):
        llm.complete([], [])
