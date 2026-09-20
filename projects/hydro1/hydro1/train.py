#!/usr/bin/env python3
"""Train Hydro-1 from scratch — the whole run, no pretrained anything.

Device autodetect: MPS (Apple Silicon) → CUDA → CPU. Writes checkpoints +
loss-history JSON (the playground's "training DNA" panel reads this) and
periodic samples so the run tells its own story.

    python -m hydro1.train                 # full run
    python -m hydro1.train --smoke         # tiny config, ~1 min, proves learning
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import BatchSampler, encode_corpus, load_corpus
from .model import GPT
from .tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "tinystories-50mb.txt"
OUT = ROOT / "checkpoints"


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def evaluate_loss(model: GPT, sampler: BatchSampler, dev: str, iters: int = 20) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = sampler.batch(dev)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model: GPT, tokenizer: CharTokenizer, prompt: str, n: int = 220, temperature: float = 0.8) -> str:
    dev = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=dev)
    out = model.generate(idx, max_new_tokens=n, temperature=temperature, top_k=40)
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to prove learning")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-bytes", type=int, default=None)
    ap.add_argument("--corpus", type=str, default=None, help="corpus file (default TinyStories subset)")
    ap.add_argument("--out-suffix", type=str, default="", help="checkpoint filename suffix")
    args = ap.parse_args()

    smoke = args.smoke
    steps = args.steps or (300 if smoke else 6000)
    max_bytes = args.max_bytes or (2_000_000 if smoke else 60_000_000)

    torch.manual_seed(1337)
    dev = device()
    print(f"device: {dev}")

    corpus_path = Path(args.corpus) if args.corpus else CORPUS
    text = load_corpus(corpus_path, max_bytes=max_bytes)
    tokenizer = CharTokenizer.from_text(text)
    # a distilled run has its own vocab — never overwrite the base run's files
    suffix = f"-{args.out_suffix}" if args.out_suffix else ""
    tok_file = OUT / (f"tokenizer{suffix}.json" if suffix else "tokenizer.json")
    tokenizer.save(tok_file)  # save BEFORE training so servers always match
    data = encode_corpus(text, tokenizer)
    print(f"corpus: {len(text):,} chars · vocab {tokenizer.vocab_size} · {len(data):,} tokens")

    cfg = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=192 if smoke else 384,
        n_layers=3 if smoke else 6,
        n_heads=3 if smoke else 6,
        block_size=128 if smoke else 256,
        dropout=0.0,
    )
    train_data, val_data = BatchSampler.split(data, min_val=cfg["block_size"] + 2)
    model = GPT(**cfg).to(dev)
    print(f"model: {model.num_params():,} non-embedding params · blocks={cfg['n_layers']} d={cfg['d_model']}")

    batch_size = 32 if smoke else 64
    train_sampler = BatchSampler(train_data, cfg["block_size"], batch_size)
    val_sampler = BatchSampler(val_data, cfg["block_size"], batch_size)

    lr = 1e-3 if smoke else 6e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = 30 if smoke else 200
    sched = lambda step: step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))  # noqa: E731

    OUT.mkdir(exist_ok=True)
    history = {"config": {**cfg, "steps": steps, "batch_size": batch_size, "lr": lr, "device": dev}, "loss": [], "val_loss": [], "samples": []}
    t0 = time.time()
    model.train()
    best_val = float("inf")

    for step in range(1, steps + 1):
        lr_s = lr * sched(step)
        for g in opt.param_groups:
            g["lr"] = lr_s
        x, y = train_sampler.batch(dev)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step == 1:
            history["loss"].append([step, round(loss.item(), 4)])
        if step % (50 if smoke else 500) == 0 or step == steps:
            val = evaluate_loss(model, val_sampler, dev, iters=10)
            history["val_loss"].append([step, round(val, 4)])
            if val < best_val:
                best_val = val
                torch.save({"model": model.state_dict(), "config": cfg}, OUT / f"best{suffix}.pt")
            if not smoke:
                s = sample(model, tokenizer, "Once upon a time", n=200)
                history["samples"].append({"step": step, "val_loss": round(val, 4), "text": s})
                print(f"step {step:5d} · loss {loss.item():.4f} · val {val:.4f} · {time.time()-t0:.0f}s")
                print("  sample:", s[:140].replace("\n", " "))
            else:
                print(f"step {step:5d} · loss {loss.item():.4f} · val {val:.4f}")

        if step % (100 if smoke else 1000) == 0:
            torch.save({"model": model.state_dict(), "config": cfg}, OUT / f"latest{suffix}.pt")
            (OUT / f"history{suffix}.json").write_text(json.dumps(history, indent=1))

    torch.save({"model": model.state_dict(), "config": cfg}, OUT / f"latest{suffix}.pt")
    tokenizer.save(tok_file)
    (OUT / f"history{suffix}.json").write_text(json.dumps(history, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min · best val {best_val:.4f} → {OUT}")


if __name__ == "__main__":
    main()
