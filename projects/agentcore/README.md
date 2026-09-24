# AgentCore — an LLM agent runtime from first principles

> An LLM agent loop from first principles — tools, parsing, guards — testable
> without a single network call.

Tool use is what turns a language model from an autocomplete into an agent,
and it is the most in-demand skill in applied LLM engineering. AgentCore
proves the mechanics can be written and tested from scratch — the tool
registry, the wire-format parser, the loop, and every guard — with a scripted
model standing in for the API: **zero dependencies, zero network, no API
keys** (the fixtures use obviously-fake strings only).

## The wire format

The model carries tool calls inside its plain-text reply as fenced blocks:

````
Here is my reasoning...

```tool
{"name": "calculator", "arguments": {"expression": "2+2*3"}}
```
````

Rules:

* the opening fence is exactly `` ```tool `` + a newline; the body is one
  JSON object with a string `name` and an object `arguments` (omittable for
  zero-parameter tools);
* a reply may contain any number of blocks, mixed freely with prose;
* everything outside the blocks is the model's visible text;
* malformed blocks raise `ParseError` with the reason — which the agent
  feeds back to the model so it can correct itself, instead of crashing.

`parse_tool_calls(text)` returns `(clean_text, [ParsedToolCall])` — the prose
with blocks stripped, plus every call in order.

## Quickstart

```bash
python -m pytest -q            # 47 tests (25 test functions, parametrized) — no network needed
python -m agentcore demo       # a fully scripted conversation, printed step by step
```

The demo runs four mini-conversations against a `MockLLM` script: a
calculation via the calculator tool, a `text_stats` call, a dangerous
`read_note` **refused** (no confirmation), and the same tool **allowed** when
a `confirm_dangerous` callback returns True.

## Architecture

| module | what it does |
|---|---|
| `agentcore/tools.py` | `Tool` dataclass + `Registry`: JSON-ish schemas, strict argument validation (types, missing, extra), the dangerous-tool gate; built-ins: `calculator` (AST walk, **never** `eval`), `text_stats`, `read_note` (sandboxed, `dangerous=True`) |
| `agentcore/llm.py` | `LLMResponse`, the one-method `BaseLLM` protocol, and `MockLLM` — a script of canned replies consumed in order, which is why everything runs offline |
| `agentcore/parser.py` | the wire format: fenced `` ```tool `` JSON blocks, robust to prose before/after and multiple blocks per reply |
| `agentcore/agent.py` | the loop: parse → validate → execute → feed results back → repeat; records every turn as a `Step` |
| `agentcore/cli.py` | `python -m agentcore demo` — the scripted showcase, no `input()`, no network |

### The loop

```
user message ──> LLM ──> parse ──┬── no tool calls ──> final answer
                                 └── tool calls ──> validate+execute each
                                        │              (results AND errors go
                                        ▼               back as tool messages)
                                     next turn
```

Guards, in the order they fire:

* **malformed block** → the model sees its own mistake and may retry;
* **unknown tool / bad arguments** → a `ToolError` goes back as a tool
  message — a validation failure is data, not a crash;
* **dangerous tool** → refused unless `confirm_dangerous(tool_name)` returns
  True (default: always refused);
* **sandbox** → `read_note` rejects absolute paths (POSIX and Windows),
  backslashes, `..` segments, and symlinks that resolve outside the sandbox;
* **max_steps** (default 8) → the loop stops with `answer=""`,
  `stopped=True`, and a reason — never an infinite tool-calling spiral;
* **exponent guard** → the calculator refuses `9**9**9`-style blowups and
  division by zero, negative bases with fractional exponents, names, calls,
  strings — anything that is not literal arithmetic.

Every `Step` in `AgentResult.steps` records the raw model reply, the parsed
calls, and each call's outcome (`ok`, `result`/`error`) — a run can be
replayed and audited line by line.

## Honest scope

* The wire format is deliberately simple. Real providers use structured
  tool-call fields, not fenced markdown — swapping that in is one `BaseLLM`
  subclass away: return `LLMResponse(text, tool_calls=[ToolCallRequest(...)])`
  and the agent executes the structured calls directly (tested; the parser is
  skipped).
* One tool call per turn is executed sequentially, no parallelism, no
  streaming, no retries/backoff, no token accounting.
* Every declared tool parameter is required; defaults are out of scope.
* The sandbox and the dangerous gate are real checks, written
  host-OS-independent — but they guard a demo file reader, not a production
  execution environment. Review before pointing tools at real files.
* The MockLLM raises loudly when the script runs dry — silent looping is a
  bug masquerading as a hang.

## Tests

```bash
python -m pytest -q
```

47 cases (25 test functions, parametrized): wire-format parsing (clean text,
multiple calls, every malformed shape), registry validation (types, unknown,
missing, extra, bad definitions), calculator correctness incl. division and
precedence, sandbox escapes (`../secrets.txt`, `/etc/passwd`, `C:\x`, UNC,
empty), dangerous refusal and confirmation, the full agent loop with
`MockLLM` scripts — tool call → answer, max_steps, malformed-call recovery,
unknown-tool feedback, structured tool calls — and the transparent step log.
