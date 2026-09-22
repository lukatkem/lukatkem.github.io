# Hydro-1 — a language model trained from scratch, X-rayed live

Named for this workspace and the smallest atom. A decoder-only GPT — attention,
MLPs, residual stream, sampling — written line by line in `hydro1/model.py`
(~180 lines, no transformer libraries), then **trained from zero on this Mac**.
Nothing downloaded: no pretrained weights, no tokenizer model, no APIs. The
checkpoints in `checkpoints/` were computed into existence by `train.py` on the
machine's own GPU (Apple MPS).

The companion playground (`web/`) makes the model's thinking visible:

- **Next-token brain** — hover any character, see the model's full probability
  distribution for what comes next; red glow marks where it was surprised
- **Attention X-ray** — scrub through every layer × every head and watch which
  characters the model consults when predicting (the causal mask is visible as
  pure lower-triangle)
- **Training DNA** — the actual loss curves from the run that produced these
  weights, plus sample outputs at checkpoints along the way

## Run it

```bash
python -m hydro1.train --smoke   # 3-layer, 1-min run — proves the loop learns
python -m hydro1.train           # the real run: ~10M params on 50MB of TinyStories
python -m hydro1.server          # playground at http://localhost:8001
pytest tests/ -q                 # shapes, causal-mask proof, overfit test
```

## Arena & distillation — the student meets frontier teachers

Hydro-1's next tier is **knowledge distillation**, the way modern small models
are actually built: frontier teachers run in the cloud (nothing downloads, no
local VRAM), they *write and grade* new training text, and only graded examples
reach the student corpus. It is one click in the **Academy** tab of the web UI:

1. **Teacher keys** — paste a key; the provider is auto-detected by prefix
   (`sk-or-…` → OpenRouter, `nvapi-…` → NVIDIA NIM, `AQ.`/`AIza…` → Google
   AI Studio, `gsk_…` → Groq, anything else → your `TEACHER_BASE_URL`). Keys
   are stored only in `hydro1/.env` (gitignored), applied live without a
   restart, and each teacher is probed and shown green/red.
2. **Distill, multi-teacher** — every frontier + free teacher takes turns
   *co-authoring* the textbook, text by text; if one provider hiccups the text
   rotates to the next with polite backoff. Graders on the cheap tiers then
   filter it — the final `VERDICT: PASS/FAIL` marker is parsed wherever a
   reasoning model leaves it. **Domain packs** mix the textbook:
   cybersecurity, Python, how computers work, and stories — weighted
   (cyber heaviest), every pack written defensively and explanation-first.
3. **Retrain** — trains on the textbook only (own tokenizer + checkpoints, so
   the base student is never touched). The playground hot-reloads the
   stronger student the moment it's done; *Remove distilled student* swaps
   back to the base run for A/B comparison.

The teacher roster is built from whichever provider keys exist:

| Provider | Models |
|---|---|
| Google AI Studio | `gemini-3.6-flash` (frontier) · `gemini-3.6-flash-lite` (fast grader) |
| Groq | `openai/gpt-oss-120b` · `qwen/qwen3.8-27b` (frontier) · `openai/gpt-oss-20b` (fast) |
| OpenRouter | `z-ai/glm-5.3` (frontier) · `z-ai/glm-5.3-flash` (fast grader) · `z-ai/glm-5.2:free` |
| NVIDIA NIM | `deepseek-ai/deepseek-v4.1-flash` · `moonshotai/kimi-k3` · `z-ai/glm-5.3` (direct) |

Override the roster with `TEACHER_MODELS` (JSON). Every entry carries its own
`base_url` + `key_env`, so mixing providers per-teacher just works.

CLI equivalent:

```bash
cp .env.example .env   # add teacher keys (OpenAI-compatible providers)
python -m hydro1.distill --stories 400 --grade --domain all   # co-authored multi-domain textbook
python -m hydro1.train --corpus corpus/distilled.txt --steps 8000 --out-suffix distilled
python -m hydro1.server                                       # playground auto-fields the distilled student
```

The **Arena** tab runs one prompt through Hydro-1 and every configured teacher
side by side with latencies. Without a key it falls back to a local teacher so
the arena works immediately; with a key you watch the student sit next to
frontier models — then close the gap after retraining.

## Scaling the student — how size actually grows

A transformer's parameter count is roughly `12 × layers × d²`. Three sizes
ship in the trainer and the Academy UI:

| size | layers × d | non-embedding params |
|---|---|---|
| `small` | 6 × 384 | ~10.7M |
| `base` | 8 × 512 | ~25M |
| `max` | 10 × 640 | ~49M |

The catch is **scaling laws** (the Chinchilla result): a model only gets
smarter when *data grows with it* — the sweet spot is ~20 training tokens per
parameter. A 25M-parameter model "wants" ~500M tokens; a 192KB textbook is
~200K. That's why the Academy's first lever is always **more teacher-written,
graded text** and the size ladder comes second — parameters and data are
scaled together, never alone.

## What a parameter is, honestly

One parameter = one adjustable number inside the network. Training is: guess →
measure how wrong → nudge every number slightly in the direction that reduces
the wrongness → repeat. "Knowledge" is the specific final values of those
millions of numbers. Frontier models differ from Hydro-1 in *scale only*:
GPT-3 was 175B parameters on 300B tokens; today's 600B+ giants train on ~15T
tokens across thousands of GPUs for months — same architecture, same AdamW +
warmup + cosine schedule, same causal mask, just sharded across a datacenter
(pipeline/tensor/data parallelism) instead of one Mac GPU. The MinHash-style
data refinery, the distillation loop, and the training math here are the same
machinery at 1/10,000th the scale — which is exactly why they fit in a repo
you can read.

## The specs of the shipped run

| | |
|---|---|
| Architecture | decoder-only GPT, pre-LN, weight-tied embeddings |
| Parameters | ~10.7M non-embedding |
| Context | 256 characters |
| Tokenizer | char-level (~100 vocab) — built from the corpus, nothing external |
| Data | TinyStories (50MB text subset) — the training *data*, not a model |
| Hardware | one Apple M4, MPS backend, batch 64, ~6000 steps |

## Why char-level

No tokenizer to download (the vocab is literally the corpus's characters), the
playground can honestly show per-character attention, and a 10M model trained
for 20 minutes producing readable English is a stronger statement than a
borrowed BPE pipeline. The tradeoffs (shorter effective context, more tokens
per word) are discussed honestly in the writeup.

## What this proves

Anyone can call an API. This artifact proves the opposite direction: gradient
flow, learning-rate warmup + cosine decay, the causal mask, weight tying,
cross-entropy over a from-scratch vocabulary — all understood at the level
where they can be *implemented and debugged*, not just imported. The tests
encode that claim: one of them fails if attention ever leaks into the future,
another fails if the training loop can't overfit.
