"""Data pipeline: corpus → byte tensor → random batches.

The whole corpus is encoded once into a flat uint16 tensor; batches are random
contiguous windows. Simple, fast, and honest — no shuffling overhead a
from-scratch run doesn't need.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .tokenizer import CharTokenizer


def load_corpus(path: Path, max_bytes: int | None = None) -> str:
    data = Path(path).read_text(encoding="utf-8", errors="ignore")
    if max_bytes:
        data = data[:max_bytes]
    return data


def encode_corpus(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    return torch.from_numpy(ids.astype(np.int64))


class BatchSampler:
    def __init__(self, data: torch.Tensor, block_size: int, batch_size: int, seed: int = 1337):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.data) - self.block_size - 1
        if hi <= 0:
            raise ValueError(
                f"corpus too small: {len(self.data)} tokens cannot fill one "
                f"{self.block_size}-token window + target"
            )
        ix = self.rng.integers(0, hi, size=self.batch_size)
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)

    @staticmethod
    def split(data: torch.Tensor, val_frac: float = 0.01, min_val: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Val keeps ≥ min_val tokens (a full window + target) when the corpus
        allows, so tiny distilled corpora don't collapse the val split."""
        n = int(len(data) * (1 - val_frac))
        if min_val > 0:
            n = min(n, len(data) - min_val)
        n = max(n, 1)
        return data[:n], data[n:]
