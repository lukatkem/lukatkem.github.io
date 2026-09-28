# lukatkem.github.io — 14 AI systems, built from scratch

**Every core algorithm of the modern AI stack — implemented by hand, tested, and shipped.**
No transformers libraries for the model, no vector DB for retrieval, no framework for the
agent loop, no SDK for the protocol. 276 tests. Live demo: [lukatkem.github.io/hydro1-play](https://lukatkem.github.io/hydro1-play)

## The systems

| project | what it proves | tests |
|---|---|---|
| [margin](projects/margin/) — Rulebook Copilot | RAG with forced citations, hybrid retrieval, streaming, evals-as-CI | 26 |
| [hydro1](projects/hydro1/) — a GPT from zero | transformer trained from scratch + live interpretability + self-improvement loop ([prometheus](https://github.com/lukatkem/prometheus)) | 9 |
| [refinery](projects/refinery/) — corpus refinery | MinHash + LSH dedup, PII scrubbing, quality gates | 6 |
| [mcpserver](projects/mcpserver/) — MCP from scratch | JSON-RPC 2.0 over stdio, no SDK | 14 |
| [gateway](projects/gateway/) — resilient LLM gateway | fallback chains, circuit breakers, budgets | 15 |
| [hnsw](projects/hnsw/) — vector index | HNSW graphs from first principles | 13 |
| [agentcore](https://github.com/lukatkem/agentcore) — agent runtime | tool loop, wire format, guards — offline-testable | 47 |
| [ragshield](https://github.com/lukatkem/ragshield) — injection defense | 6 heuristic families, sanitizer, output filter | 19 |
| [tokenforge](https://github.com/lukatkem/tokenforge) — BPE tokenizer | trained from scratch, deterministic, lossless | 15 |
| [weightsmith](https://github.com/lukatkem/weightsmith) — quantization | per-tensor/row/channel int8 + calibration | 40 |
| [gradia](https://github.com/lukatkem/gradia) — autograd | reverse-mode AD + MLP in pure Python | 28 |
| [ctxpack](https://github.com/lukatkem/ctxpack) — context packer | greedy / density / exact knapsack | 19 |
| [promptkit](projects/promptkit/) — prompt testing | prompts as versioned, tested code | 13 |
| [evalforge](projects/evalforge/) — eval generator | docs → golden sets, offline | 12 |

**Standalone repos:** [agentcore](https://github.com/lukatkem/agentcore) ·
[ragshield](https://github.com/lukatkem/ragshield) · [tokenforge](https://github.com/lukatkem/tokenforge) ·
[weightsmith](https://github.com/lukatkem/weightsmith) · [gradia](https://github.com/lukatkem/gradia) ·
[ctxpack](https://github.com/lukatkem/ctxpack) · [prometheus](https://github.com/lukatkem/prometheus)

**The flagship:** [prometheus](https://github.com/lukatkem/prometheus) — a closed loop where the
model measures its own weaknesses, commissions a targeted textbook, retrains, and re-measures.
Iteration-1 report: [evalboard/prometheus_report.html](evalboard/prometheus_report.html) —
computers 0.02 weakest → targeted distill commissioned.

**Eval data:** [real 70-question golden set](projects/margin/evals/report.json) —
recall@5 92.9% · MRR 0.755 · accuracy 78.6%. Dashboard: [evalboard](evalboard/dashboard.html).

License: MIT. Built and maintained by Luka Tkemaladze — [profile](https://github.com/lukatkem).
