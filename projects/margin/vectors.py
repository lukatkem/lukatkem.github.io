"""Dense vectors: Ollama embeddings (nomic-embed-text) with a TF-IDF fallback.

The fallback keeps retrieval working with zero model downloads — CI runs on
it. `kind` records which embedder built a vector set; vectors and queries are
always compared within the same kind.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import httpx
import numpy as np

from .bm25 import tokenize
from .config import EMBED_MODEL, OLLAMA_BASE, OLLAMA_TIMEOUT


def _ollama_embed(texts: list[str]) -> list[list[float]] | None:
    try:
        r = httpx.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        embs = r.json().get("embeddings")
        if embs and len(embs) == len(texts):
            return embs
    except Exception:
        return None
    return None


class TfidfEncoder:
    """Deterministic fallback encoder: l2-normalized sublinear TF-IDF."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: list[str]) -> None:
        df: Counter = Counter()
        for t in texts:
            df.update(set(tokenize(t)))
        n = max(len(texts), 1)
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        self.idf = np.array(
            [math.log((n + 1) / (df[term] + 1)) + 1.0 for term in sorted(df)],
            dtype=np.float32,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        assert self.idf is not None, "fit() before encode()"
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, t in enumerate(texts):
            counts = Counter(tokenize(t))
            for term, c in counts.items():
                j = self.vocab.get(term)
                if j is not None:
                    mat[row, j] = (1.0 + math.log(c)) * self.idf[j]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def vec_to_blob(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_topk(query: np.ndarray, matrix: np.ndarray, ids: list[int], top_k: int) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    q = query / (np.linalg.norm(query) or 1.0)
    m = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
    scores = m @ q
    order = np.argsort(-scores)[:top_k]
    return [(ids[i], float(scores[i])) for i in order]
