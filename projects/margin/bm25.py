"""BM25 lexical ranking — pure Python, no dependencies.

k1=1.5, b=0.75 (standard). Tokenization: lowercase alphanumeric runs plus a
light suffix stripper so "payouts" matches "payout".
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9$%][a-z0-9._%]*")
_KEEP_S = ("ss", "us", "is")  # "class", "status" — not plurals
_PLURAL_KEEP = ("sses", "xes", "ches", "shes", "zes")


def _strip_suffix(t: str) -> str:
    if t.endswith(_PLURAL_KEEP):
        return t[:-2]  # boxes→box, buses→bus
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"  # policies→policy
    if t.endswith("ing") and len(t) - 3 >= 3:
        return t[:-3]
    if t.endswith("ed") and len(t) - 2 >= 3:
        return t[:-2]
    if t.endswith("s") and len(t) - 1 >= 3 and not t.endswith(_KEEP_S):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text.lower())]
    return [_strip_suffix(t) for t in toks]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def add(self, doc_id: int, text: str) -> None:
        toks = tokenize(text)
        assert doc_id == len(self.docs), "doc ids must be sequential"
        self.docs.append(toks)
        c = Counter(toks)
        self.tf.append(c)
        self.doc_len.append(len(toks))
        for term in c:
            self.df[term] += 1
        self.avgdl = sum(self.doc_len) / len(self.doc_len)

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((n - df + 0.5) / df + 1.0)

    def score(self, query: str, doc_id: int) -> float:
        q = tokenize(query)
        s = 0.0
        tf, dl = self.tf[doc_id], self.doc_len[doc_id]
        norm = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
        for term in q:
            f = tf.get(term, 0)
            if f == 0:
                continue
            s += self._idf(term) * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
