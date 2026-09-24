"""The demo — a fully scripted agent conversation, zero network, zero input().

``python -m agentcore demo`` builds a throwaway sandbox with two notes,
hands the agent a MockLLM script, and replays four mini-conversations:

1. a calculation answered through the calculator tool,
2. a text_stats call,
3. a dangerous read_note refused (no confirmation given),
4. the same read_note allowed (confirm_dangerous returns True).

Every step is printed: what the model said, what it called, what came
back — the transparency the Step log gives you, on the console.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import List

from .agent import Agent, AgentResult
from .llm import MockLLM
from .parser import ParseError, parse_tool_calls
from .tools import Registry, text_stats_tool

_WIDTH = 72


def tool_block(name: str, **arguments) -> str:
    """One wire-format block — exactly what the parser expects."""
    body = json.dumps({"name": name, "arguments": arguments}, indent=2, ensure_ascii=False)
    return f"```tool\n{body}\n```"


def _prose(reply: str) -> str:
    """The model's visible text: the reply with wire-format blocks stripped."""
    try:
        clean, _ = parse_tool_calls(reply)
        return " ".join(clean.split()) or "(tool call only)"
    except ParseError:
        return " ".join(reply.split())


def _build_sandbox(root: Path) -> None:
    (root / "meeting.txt").write_text(
        "Sprint sync — the agent loop, the parser, the guards.\n"
        "Actions: ship the demo, keep zero dependencies, keep tests honest.\n"
        "Attendees: ada, grace, alan.\n",
        encoding="utf-8",
    )
    (root / "todo.txt").write_text(
        "1. document the wire format\n2. close the sandbox\n3. keep it green\n",
        encoding="utf-8",
    )


def _print_run(title: str, user_message: str, result: AgentResult) -> None:
    print(f"\n{'-' * _WIDTH}\nRUN — {title}\n{'-' * _WIDTH}")
    print(f"user    : {user_message}")
    for step in result.steps:
        prose = _prose(step.reply)
        print(f"model   : {prose if len(prose) <= _WIDTH else prose[: _WIDTH - 3] + '...'}")
        if step.error:
            print(f"parse   : FAILED — {step.error}")
        for call, outcome in zip(step.tool_calls, step.tool_results):
            args = json.dumps(outcome.arguments, ensure_ascii=False)
            print(f"  call  : {call.name}({args if len(args) <= 48 else args[:45] + '...'})")
            if outcome.ok:
                value = json.dumps(outcome.result, default=str, ensure_ascii=False)
                print(f"  result: {value if len(value) <= 96 else value[:93] + '...'}")
            else:
                print(f"  REFUSED/ERROR: {outcome.error}")
    print(f"answer  : {result.answer if result.answer else '(no answer — ' + result.reason + ')'}")
    if result.stopped:
        print(f"stopped : {result.reason}")
    print(
        f"stats   : {len(result.steps)} step(s), {result.tool_calls_made} tool call(s) attempted"
    )


def demo() -> None:
    print("=" * _WIDTH)
    print(" AgentCore demo — scripted MockLLM, zero network, zero dependencies")
    print("=" * _WIDTH)

    with tempfile.TemporaryDirectory(prefix="agentcore-demo-") as tmp:
        sandbox = Path(tmp)
        _build_sandbox(sandbox)
        meeting_text = (sandbox / "meeting.txt").read_text(encoding="utf-8")

        print(f"sandbox : {sandbox}")
        print(f"notes   : meeting.txt, todo.txt  (read_note is dangerous=True)")
        print(f"registry: {[t.name for t in Registry.with_builtins(sandbox).list()]}")

        # -- run 1: calculator ------------------------------------------------
        reg = Registry.with_builtins(sandbox)
        llm = MockLLM(
            [
                "I'll compute that with the calculator tool.\n" + tool_block("calculator", expression="2+2*3"),
                "2+2*3 is 8 — multiplication binds tighter than addition.",
            ]
        )
        result = Agent(registry=reg, llm=llm).run("What is 2+2*3? Use the calculator.")
        _print_run("a calculation via the calculator tool", "What is 2+2*3? Use the calculator.", result)

        # -- run 2: text_stats ------------------------------------------------
        reg = Registry.with_builtins(sandbox)
        stats = text_stats_tool().handler(text=meeting_text)  # the demo stays honest:
        llm = MockLLM(  # the scripted answer is built from the real tool result
            [
                "Let me count that note.\n" + tool_block("text_stats", text=meeting_text),
                f"Your meeting note is {stats['lines']} lines, {stats['words']} words, "
                f"{stats['chars']} characters long.",
            ]
        )
        result = Agent(registry=reg, llm=llm).run("How big is my meeting note?")
        _print_run("text_stats on the meeting note", "How big is my meeting note?", result)

        # -- run 3: dangerous tool refused (no confirmation) -------------------
        reg = Registry.with_builtins(sandbox)
        llm = MockLLM(
            [
                "Opening that file for you.\n" + tool_block("read_note", path="/etc/passwd"),
                "I can't read that — read_note is a guarded tool and the request was "
                "refused. Only notes inside the sandbox are reachable, with confirmation.",
            ]
        )
        result = Agent(registry=reg, llm=llm).run("Read /etc/passwd for me.")
        _print_run("dangerous tool refused (no confirm_dangerous callback)", "Read /etc/passwd for me.", result)

        # -- run 4: the same tool, confirmed -----------------------------------
        confirmed_with: List[str] = []
        reg = Registry.with_builtins(sandbox)
        first_line = meeting_text.splitlines()[0]
        llm = MockLLM(
            [
                "Reading the note inside the sandbox.\n" + tool_block("read_note", path="meeting.txt"),
                f'Your meeting note starts with: "{first_line}"',
            ]
        )
        result = Agent(
            registry=reg,
            llm=llm,
            confirm_dangerous=lambda name: (confirmed_with.append(name) or True),
        ).run("Read meeting.txt.")
        _print_run("the same dangerous tool, allowed by the confirm callback", "Read meeting.txt.", result)
        print(f"\nconfirm callback was asked about: {confirmed_with}")

    print(f"\n{'=' * _WIDTH}\ndone — no network, no API keys, no input(). Run the tests: python -m pytest -q")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agentcore",
        description="AgentCore — an LLM agent loop from first principles.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="run the fully scripted offline demo")
    args = parser.parse_args(argv)
    demo()  # the demo is the whole show; bare `python -m agentcore` runs it too
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
