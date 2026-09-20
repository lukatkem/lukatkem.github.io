# mcpserver

A Model Context Protocol (MCP) server built from scratch: hand-written JSON-RPC 2.0 over stdio, Python standard library only. No official SDK, no third-party dependencies, no network access. Requires Python 3.8+.

## What is MCP?

The Model Context Protocol is an open standard for connecting LLM applications to external tools and context: a client (IDE, chat app, agent) launches a server process and communicates with it over JSON-RPC 2.0. After an `initialize` handshake that negotiates the protocol version and advertises capabilities, the client can list the server's tools and invoke them with structured arguments. Over the stdio transport, every message travels as one line of JSON — requests on the server's stdin, responses on its stdout — which is exactly what this project implements.

## Why implement the protocol from scratch?

- **No trust surface.** Installing an SDK pulls a dependency tree into a process that an LLM can invoke with arbitrary arguments. Here the entire protocol layer is one auditable file (`mcpserver/protocol.py`, ~250 lines).
- **The protocol is small.** MCP's core is a handful of JSON-RPC methods. Implementing them directly makes the mechanics explicit: version negotiation, capability advertisement, the `tools/list` → `tools/call` loop, and the split between protocol errors (JSON-RPC `error`) and tool errors (`isError: true` results).
- **Zero dependencies means zero breakage.** The server runs on any Python 3.8+ interpreter with nothing installed.

## Protocol surface

| Method | Kind | Behavior |
|---|---|---|
| `initialize` | request | Echoes the client's `protocolVersion` if supported (`2025-06-18`, `2024-11-05`), else replies `2025-06-18`; returns `capabilities: {tools: {}}` and `serverInfo: {name, version}`. |
| `notifications/initialized` | notification | Acknowledged silently — notifications never produce a reply. |
| `ping` | request | Replies `{}`. |
| `tools/list` | request | Returns the four tools with `name`, `description`, `inputSchema` (JSON Schema). |
| `tools/call` | request | `{name, arguments}` → `{content: [{type: "text", text: …}], isError}`. |
| anything else | request | JSON-RPC error `-32601` (method not found). |

**Error model.** Malformed JSON → `-32700` (parse error) and the loop keeps serving. Structurally invalid requests → `-32600`. Bad `tools/call` params such as an unknown tool → `-32602`. A tool whose *arguments* fail validation — e.g. a missing required argument — follows the MCP convention and returns a normal result with `isError: true` and a helpful message in `content`, so the calling LLM can self-correct; only protocol-level problems become JSON-RPC errors.

## Transcript

```
$ printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
    '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
    'not json at all' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"corpus_stats","arguments":{"text":"the quick brown fox jumps over the lazy dog"}}}' \
    | python -m mcpserver.server
{"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "mcpserver", "version": "1.0.0"}}}
{"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "parse error: Expecting value"}}
{"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "chars: 43\nwords: 9\nunique-words: 8\nlines: 1\nmd5: 77add1d5f41223d5582fca736a5cb335\ntop-words: the (2), quick (1), brown (1)"}], "isError": false}}
```

The notification produced no line, the malformed line produced a parse error, and the session survived both. Run it from the project root (or set `cwd`/`PYTHONPATH` for the child, as MCP clients do).

## Connecting an MCP client

Any MCP client that supports stdio servers launches the process itself and speaks the protocol above. A typical client configuration:

```json
{
  "mcpServers": {
    "mcpserver": {
      "command": "python",
      "args": ["-m", "mcpserver"],
      "cwd": "/absolute/path/to/mcpserver"
    }
  }
}
```

`python -m mcpserver` and `python -m mcpserver.server` are equivalent; `--version` prints the package version.

## Tool catalog

| Tool | Arguments | What it does |
|---|---|---|
| `dedupe_corpus` | `texts: [string]`, `threshold?: number = 0.8` | Near-duplicate detection: word 5-gram shingles → 64-permutation MinHash (deterministic blake2b seeding) → 16-band LSH → estimated Jaccard → union-find clusters. Reports unique vs. duplicate counts and each cluster with a representative. |
| `scrub_pii` | `text: string` | Replaces emails, phone-like digit sequences (9+ digits; dates like `2026-09-20` are preserved), bare digit runs of 10+ (cards/ids), API keys (`sk-…`, `sk-or-v1-…`, `nvapi-…`, `ghp_…`, `xox…`) and JWTs with `[REDACTED]`. Returns cleaned text plus a count per kind. |
| `quality_score` | `text: string` | 0..1 score with named reasons. Hard rejects (score 0): <200 chars, <25 words, >15% symbols, any line repeated 3+ times. Soft penalties (−0.15 each): SHOUTING caps, mean word length >9, lorem ipsum, link farms (>4 links). |
| `corpus_stats` | `text: string` | Character/word/unique-word/line counts, an MD5 fingerprint (non-cryptographic), and the 3 most common words. |

## Design notes

**dedupe_corpus.** Documents are lowercased and split into word 5-gram shingles (texts shorter than 5 words collapse to a single shingle so they stay comparable). Each shingle gets a 64-bit blake2b hash; the MinHash signature applies 64 affine permutations `(a·h + b) mod (2⁶¹−1)`, with the `a`/`b` coefficients derived deterministically from keyed blake2b — same input, same signature on every run. 16 bands × 4 rows of LSH generate candidate pairs without an all-pairs scan; each candidate pair is scored by the fraction of equal signature rows (the standard MinHash Jaccard estimate) and merged into a union-find forest when it clears the threshold.

**scrub_pii.** One master regex with named alternatives, ordered by priority: emails, API keys, JWTs, dates, then phone-like digit sequences. Dates are consumed first and passed through verbatim, so `2026-09-20` can never be redacted. Phone-pattern matches are classified in code: digit groups joined by separators count as a phone at 9+ digits, a bare run counts as a long digit run at 10+ digits, and anything shorter is left untouched.

**quality_score.** Hard rejects force score 0 and verdict `FAIL`; otherwise each soft penalty deducts 0.15 and the verdict is `PASS` at ≥ 0.85, `WARN` below. Output is a one-line summary (`score 0.85 · PASS · reasons: shouting-caps`) followed by one bullet per reason.

## Layout

```
mcpserver/
├── mcpserver/
│   ├── __init__.py     # package metadata
│   ├── protocol.py     # JSON-RPC 2.0 dispatch, MCP methods, stdio loop
│   ├── tools.py        # the four tools: algorithms + JSON schemas
│   ├── server.py       # python -m mcpserver.server
│   ├── cli.py          # argparse wrapper used by the entry point below
│   └── __main__.py     # enables python -m mcpserver
└── tests/
    └── test_mcp.py     # end-to-end subprocess tests + unit checks
```

## Tests

```
python -m pytest tests/ -q
```

The suite spawns the real server as a subprocess and speaks JSON-RPC to it over pipes: handshake and version negotiation, the exact tool catalog, every tool's core behavior, protocol errors vs. tool errors, notification silence, and recovery from malformed input. The child process is reaped (`terminate` + `wait`) in the fixture teardown.

## Limitations

- Implements the tools subset of MCP: no `resources/*`, `prompts/*`, `sampling/*`, or logging notifications.
- Requests are handled sequentially; JSON-RPC batch arrays are rejected as invalid requests.
- `tools/list` is static — no `notifications/tools/list_changed`.
- One client per process, which is the normal shape for a stdio transport.
