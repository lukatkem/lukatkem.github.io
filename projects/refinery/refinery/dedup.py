"""Stage 5 — deduplication: exact + near, both from scratch.

Near-duplicate detection without any library:
  * shingles   — k consecutive words per document
  * MinHash    — 64 hash permutations compress a shingle set into a
                 signature; signature agreement estimates Jaccard
  * banded LSH — 16 bands × 4 rows: documents sharing a band bucket are
                 candidate duplicates (prob. of a 0.8-duplicate missing
                 every band ≈ (1 - 0.8⁴)¹⁶ ≈ 0.4%)
  * union-find — merge candidates into clusters; keep one representative

This is the same architectural shape production corpus tools use — just
readable, and honest about the math.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, k: int = 5) -> set[bytes]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {" ".join(words).encode()} if words else set()
    return {" ".join(words[i : i + k]).encode() for i in range(len(words) - k + 1)}


def _hash64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


@dataclass(frozen=True)
class MinHasher:
    num_perm: int = 64
    seed: int = 1337

    def __post_init__(self):
        if self.num_perm % 4 != 0:
            raise ValueError("num_perm must be a multiple of the band count divisor (4)")

    def _params(self) -> list[tuple[int, int]]:
        # (a, b) universal-hash pairs derived deterministically from the seed
        out = []
        for i in range(self.num_perm):
            h = _hash64(f"{self.seed}:{i}".encode())
            out.append((h | 1, (h >> 32) | 1))  # odd multipliers
        return out

    def signature(self, shingle_set: set[bytes]) -> tuple[int, ...]:
        if not shingle_set:
            return tuple([0] * self.num_perm)
        hashed = [_hash64(s) for s in shingle_set]
        sig = []
        for a, b in self._params():
            sig.append(min((a * x + b) & 0xFFFFFFFFFFFFFFFF for x in hashed))
        return tuple(sig)


def jaccard_estimate(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    agree = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return agree / len(sig_a)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def clusters(
    texts: list[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    bands: int = 16,
) -> list[list[int]]:
    """Return groups of duplicate indices (each group ≥ 2). First member of a
    group is the survivor; everything else is a near/exact duplicate.

    texts must already be normalized — garbage in, garbage buckets out."""
    hasher = MinHasher(num_perm=num_perm)
    sigs = [hasher.signature(shingles(t)) for t in texts]

    uf = _UnionFind(len(texts))
    rows = num_perm // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for idx, sig in enumerate(sigs):
        for b in range(bands):
            key = (b, sig[b * rows : (b + 1) * rows])
            buckets.setdefault(key, []).append(idx)
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if uf.find(a) != uf.find(b) and jaccard_estimate(sigs[a], sigs[b]) >= threshold:
                    uf.union(a, b)

    groups: dict[int, list[int]] = {}
    for idx in range(len(texts)):
        groups.setdefault(uf.find(idx), []).append(idx)
    return [g for g in groups.values() if len(g) >= 2]


def exact_duplicate_groups(texts: list[str]) -> list[list[int]]:
    seen: dict[str, int] = {}
    groups: dict[int, list[int]] = {}
    for idx, t in enumerate(texts):
        h = hashlib.md5(t.encode()).hexdigest()
        if h in seen:
            groups.setdefault(seen[h], []).append(idx)
        else:
            seen[h] = idx
    return [[k] + v for k, v in groups.items()]
