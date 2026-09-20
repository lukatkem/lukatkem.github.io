# lukatkem.github.io — AI systems, built from scratch

The portfolio site **and** the code. The landing page at [lukatkem.github.io](https://lukatkem.github.io)
is generated from this repo (pure HTML/CSS, no build step), and every project
shown on it lives under [`projects/`](projects/).

| Project | What it is | Tests |
|---|---|---|
| [`projects/margin`](projects/margin) | RAG copilot — hybrid retrieval, forced citations, evals as a CI gate, SaaS layer, streaming | 23 |
| [`projects/hydro1`](projects/hydro1) | A GPT written and trained from zero + interpretability playground + one-click distillation from frontier teachers | 9 |
| [`projects/refinery`](projects/refinery) | Corpus data refinery — MinHash/LSH dedup, PII & secret scrubbing, quality gates. Stdlib only | 6 |
| [`projects/mcpserver`](projects/mcpserver) | Model Context Protocol server hand-written in JSON-RPC over stdio | 14 |
| [`projects/gateway`](projects/gateway) | Resilient multi-provider LLM gateway — circuit breakers, fallbacks, budgets, dashboard | 15 |
| [`projects/hnsw`](projects/hnsw) | HNSW approximate-nearest-neighbor index from scratch, benchmarked vs brute force | 13 |

**80 tests · 6 CI pipelines · 0 AI model downloads.**

## CI

`.github/workflows/tests.yml` runs each project's suite on every push
(`paths:` filters keep jobs focused). Refinery additionally smoke-runs the
full CLI against the dirty demo corpus.

## Secrets

No secrets live in this repo — see [SECURITY.md](SECURITY.md). Short version:
keys stay in git-ignored `.env` files outside version control; gateway and
copilot API keys are stored only as hashes; CI files contain no credentials.

## The classic site

The previous portfolio lives on at [`/v1/`](v1/).
