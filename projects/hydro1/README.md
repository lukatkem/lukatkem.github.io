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
are actually built: frontier teachers run in the cloud (GLM 5.3 / GLM 5.3
Flash via an OpenAI-compatible endpoint — nothing downloads, no local VRAM),
they *write and grade* new training text, and only graded examples reach the
student corpus.

The whole flow is one-click in the **Academy** tab of the web UI:

1. **Teacher key** — paste an API key; the provider is auto-detected by
   prefix (`sk-or-…` → OpenRouter, `nvapi-…` → NVIDIA NIM, anything else →
   your `TEACHER_BASE_URL`). Keys are stored only in `hydro1/.env`
   (gitignored), applied live without a restart, and each teacher is probed
   and shown green/red.
2. **Distill** — the teacher writes short stories, then grades its own work;
   only stories that pass become the student's textbook. Live progress bar +
   log in the UI. Reasoning teachers (GLM 5.3, DeepSeek V4) deliberate
   before answering — the client budgets tokens for that and parses the
   final `VERDICT:` marker wherever the model puts it.
3. **Retrain** — trains on the textbook only (own tokenizer + checkpoints, so
   the base student is never touched). The playground hot-reloads the
   stronger student the moment it's done; *Remove distilled student* swaps
   back to the base run for A/B comparison.

The teacher roster is built from whichever provider keys exist:

| Provider | Models |
|---|---|
| OpenRouter | `z-ai/glm-5.3` (frontier) · `z-ai/glm-5.3-flash` (fast grader) · `z-ai/glm-5.2:free` |
| NVIDIA NIM | `deepseek-ai/deepseek-v4-flash-0731` |

Override the roster with `TEACHER_MODELS` (JSON). Every entry carries its own
`base_url` + `key_env`, so mixing providers per-teacher just works.

CLI equivalent:

```bash
cp .env.example .env   # add TEACHER_API_KEY (Z.AI or any OpenAI-compatible provider)
python -m hydro1.distill --stories 200 --grade     # teachers write + grade (~200 stories)
python -m hydro1.train --corpus corpus/distilled.txt --steps 8000 --out-suffix distilled
python -m hydro1.server                            # playground auto-fields the distilled student
```

The **Arena** tab runs one prompt through Hydro-1 and every configured teacher
side by side with latencies. Without a key it falls back to a local teacher so
the arena works immediately; with a key you watch the 10M student sit next to
a frontier model — then close the gap after retraining.

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
