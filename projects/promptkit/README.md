# promptkit — prompts as versioned, tested code

**A prompt is production logic; treat it like it.** Typed slots, strict rendering,
and assertion-based regression suites that run offline against a mock LLM — so
changing a prompt becomes a code change with a test gate, not a vibe.

## Quickstart

```bash
python -m pytest -q          # 13 tests
python -m promptkit demo     # a regression suite passing 2/2 cases
```

```python
from promptkit import Prompt, Suite, contains, max_length
from promptkit.llm import MockLLM

prompt = Prompt(name="summarizer", version="1.2.0",
                template="Summarize this ticket: {ticket}", slots={"ticket": "str"})
suite = (Suite(prompt, MockLLM(["Customer cannot log in…"]))
         .case("login", {"ticket": "can't log in"}, [contains("log in"), max_length(200)]))
assert suite.run().ok
```

## API

| piece | what it does |
|---|---|
| `Prompt(name, template, version, slots)` | typed slots, undeclared/unused slots raise |
| `Suite(prompt, llm).case(name, slots, assertions)` | chain cases; runs against any BaseLLM |
| assertions | `contains` / `not_contains` / `max_length` / `regex` / any callable |
| `MockLLM(replies)` | deterministic offline double; records calls, raises when exhausted |

## Honest scope

Assertions are exact-match style — no semantic similarity scoring. Pair the mock
with any real OpenAI-compatible client by subclassing `BaseLLM` (one method).
