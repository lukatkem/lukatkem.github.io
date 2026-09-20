"""Inference utilities shared by CLI + playground server.

`inspect_prompt` is the X-ray: runs the model and returns every layer's
attention maps, per-position top-k next-token predictions, and the loss per
character — the raw material for the playground's visualizations.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import GPT
from .tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "checkpoints"


def load_model(path: Path | None = None, device: str = "cpu") -> tuple[GPT, CharTokenizer]:
    """Prefer the distilled student when present — the arena should always
    field the strongest locally-trained weights. Each student keeps its own
    tokenizer (a distilled corpus has a different vocab than the base run)."""
    tok_path = CHECKPOINTS / "tokenizer.json"
    if path is None:
        distilled = CHECKPOINTS / "latest-distilled.pt"
        if distilled.exists():
            path = distilled
            tok_path = CHECKPOINTS / "tokenizer-distilled.json"
        else:
            path = CHECKPOINTS / "latest.pt"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = GPT(**blob["config"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    if not tok_path.exists():
        tok_path = CHECKPOINTS / "tokenizer.json"
    tokenizer = CharTokenizer.load(tok_path)
    return model, tokenizer


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    dev = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=dev)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


@torch.no_grad()
def inspect_prompt(
    model: GPT,
    tokenizer: CharTokenizer,
    text: str,
    top_k: int = 8,
) -> dict:
    """Return attention maps + per-char next-token predictions for the text."""
    dev = next(model.parameters()).device
    ids = tokenizer.encode(text)[-model.block_size :]
    idx = torch.tensor([ids], dtype=torch.long, device=dev)
    logits, _ = model(idx)
    probs = torch.softmax(logits[0], dim=-1)  # (T, vocab)

    # per-position top-k next-token candidates
    predictions = []
    for t in range(len(ids)):
        top = torch.topk(probs[t], min(top_k, probs.size(-1)))
        predictions.append(
            [
                {"char": tokenizer.itos[i.item()], "p": round(p.item(), 4)}
                for p, i in zip(top.values, top.indices)
            ]
        )

    # per-character loss (surprisal) — where the model is confused
    tgt = torch.tensor(ids[1:] + [tokenizer.unk], dtype=torch.long, device=dev)
    per_char_loss = torch.nn.functional.cross_entropy(
        logits[0], tgt, reduction="none"
    ).tolist()

    attentions = [
        block.attn.attn_map[0].tolist()  # block.attn is the module; .attn_map the tensor
        for block in model.blocks
    ]
    return {
        "chars": [tokenizer.itos[i] for i in ids],
        "predictions": predictions,
        "per_char_loss": [round(v, 3) for v in per_char_loss],
        "layers": len(attentions),
        "heads": len(attentions[0]) if attentions else 0,
        "attention": attentions,
    }


def history() -> dict:
    empty = {"loss": [], "val_loss": [], "samples": []}
    # match the checkpoint load_model() fields: distilled student → its history
    if (CHECKPOINTS / "latest-distilled.pt").exists():
        p = CHECKPOINTS / "history-distilled.json"
        if p.exists():
            return json.loads(p.read_text())
    path = CHECKPOINTS / "history.json"
    if not path.exists():
        return empty
    return json.loads(path.read_text())
