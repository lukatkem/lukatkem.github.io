"""python -m faultline demo — a real agentcore agent under injected faults."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "agentcore"))

from .runner import ChaosRunner


def _wire(tool, **arguments):
    body = json.dumps({"name": tool, "arguments": arguments}, indent=2)
    return f"```tool\n{body}\n```"


def _agent_factory(registry, scenario):
    """A real agentcore Agent with the scenario's own scripted LLM."""
    from agentcore.agent import Agent
    from agentcore.llm import MockLLM

    return Agent(registry=registry, llm=MockLLM(scenario["llm_script"]), max_steps=5)


def _registry_factory(profile):
    from agentcore.tools import Registry
    from .proxy import FaultyRegistry

    tmp = tempfile.mkdtemp()
    (Path(tmp) / "servers.txt").write_text("server count is 31 and status is ok", encoding="utf-8")
    registry = Registry.with_builtins(Path(tmp))
    return FaultyRegistry(registry, profile, json_tools=("read_note",))


def _scenarios():
    calc = ["I'll check that with a tool.\n" + _wire("calculator", expression="2+2*3"),
            "2+2*3 is 8."]
    note = ["Let me read the note.\n" + _wire("read_note", name="servers"),
            "The note says the server count is 31 and status is ok."]
    return [
        {"name": "calculator sum — healthy baseline",
         "message": "What is 2+2*3? Use the calculator.",
         "expected": "8", "faults": {}, "llm_script": calc},
        {"name": "calculator sum — timeout injected",
         "message": "What is 2+2*3? Use the calculator.",
         "expected": "8", "faults": {"calculator": [("timeout", 1.0)]}, "llm_script": calc},
        {"name": "calculator sum — empty response injected",
         "message": "What is the sum 2+2*3?",
         "expected": "8", "faults": {"calculator": [("empty", 1.0)]}, "llm_script": calc},
        {"name": "read_note — wrong-but-plausible data injected",
         "message": "Read the servers note and tell me the server count.",
         "expected": "31", "faults": {"read_note": [("wrong_but_plausible", 1.0)]},
         "llm_script": note},
        {"name": "read_note — truncated response injected",
         "message": "Read the note and summarize it.",
         "expected": "31", "faults": {"read_note": [("truncated", 1.0)]}, "llm_script": note},
        {"name": "read_note — connection error injected",
         "message": "Read the servers note.",
         "expected": "31", "faults": {"read_note": [("connection_error", 1.0)]},
         "llm_script": note + ["Sorry, I could not read the note because of an error."]},
    ]


def main() -> int:
    runner = ChaosRunner(_agent_factory, _registry_factory, _scenarios())
    report = runner.run()
    print("== faultline demo — chaos report ==")
    for c in report.cases:
        print(f"  {c.scenario:<46} baseline={c.baseline_verdict.value:<22} "
              f"faulted={c.faulted_verdict.value}")
    print("\n" + report.summary())
    print("silent failures are the headline: confident answers built on corrupted data")
    return 0
