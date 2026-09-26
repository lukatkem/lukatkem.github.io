# Rulebook Copilot (margin)

A **retrieval-augmented copilot that answers only from your documents** — every
claim forced to carry a numbered citation you can audit. Built as a prop-trading
rulebook assistant, but the engine is document-agnostic: point it at any corpus.

Everything here is written from first principles — the BM25 scorer, the chunker,
the reciprocal-rank fusion, the streaming protocol — no retrieval framework, no
LangChain. FastAPI + a standard library core.

## How it answers

1. **Ingest** — Markdown-aware chunking (headings, lists, and tables are kept
   semantically intact) into an indexed store (`chunker.py`, `ingest.py`).
2. **Retrieve** — hybrid search: a hand-written **BM25** lexical scorer (`bm25.py`)
   fused with learned **embeddings** (`vectors.py`, Ollama `nomic-embed-text`)
   via **reciprocal-rank fusion** (`retrieve.py`).
3. **Answer** — the generator may only speak through `[n]` citations that point
   back at retrieved chunks; `evals.py` grades answers against a golden set.
4. **Stream** — NDJSON events (`streamfmt.py`): `meta` → `delta…` → `final`, with
   citations arriving **before** the prose so the UI can render sources early.

## API surface

| Endpoint | What it does |
|---|---|
| `POST /api/chat` | Ask the rulebook (auth + plan quotas enforced) |
| `POST /api/chat/stream` | Same, as a torn-packet-safe NDJSON stream |
| `POST /api/ingest` | Rebuild the index from the corpus directory |
| `GET /api/evals` | Latest golden-set eval report as JSON (auth required) |
| `GET /api/health` | Index size, embedder, LLM, billing status |
| `/` | Web UI: streaming chat with numbered, scored citations |

Auth and plans live in `auth.py` / `billing.py` (hashed credentials, quotas,
Stripe-ready tiers); `tracing.py` gives every request an ID and timings.

## Run it

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text && ollama pull qwen2.5:7b
make ingest          # build the index from corpus/
make serve           # http://localhost:8000
make eval            # retrieval + answer quality report
```

Docker: `docker build -t margin-copilot . && docker run -p 8000:8000 margin-copilot`
(fly.toml included for Fly.io).

## Evaluations

`evals/golden.jsonl` (repo root) holds the golden set — 70 prop-firm questions,
each with a gold document and the exact strings a correct answer must contain.
`make eval` runs `margin.evals` over the set and exits non-zero when recall@5
or answer accuracy fall below threshold (80% / 75% by default), which blocks
the PR in CI.

A run produces two report files:

- `evals/report.md` — human-readable scorecard: metrics, per-type breakdown,
  failing questions.
- `margin/evals/report.json` — structured snapshot (`example`, `generated_at`,
  `summary`, per-question `results`) written by `evals.save_report()` and
  served by `GET /api/evals` (auth required; 404 with "run make eval first"
  if absent). A committed `"example": true` snapshot ships with the app so
  the endpoint works out of the box; the next real run replaces it with
  `"example": false`.

## Tests — 26, all offline

```bash
python -m pytest tests tests_root -q     # when run from a checkout with tests_root
```

- `tests/test_stream.py` — NDJSON protocol: torn packets, event ordering
- `tests_root/` — auth & plans, billing quotas, BM25 scoring, chunking,
  retrieval fusion, eval harness (`pytest` from the project root)

No network needed — generators and embedders are faked in tests.

## Design notes

- **Citations are enforced, not hoped for** — the contract is checked, and the
  eval harness fails answers that speak without sources.
- **Streaming is a protocol, not a UI trick** — the NDJSON grammar is tested
  byte-by-byte, including chunks split mid-line.
- **Every stage is swappable** — BM25, vectors, fusion, and the generator are
  independent modules with narrow interfaces.
