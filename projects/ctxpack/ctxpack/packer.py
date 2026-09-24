"""Three packers, one contract: never exceed the usable budget, and every
dropped document carries a named reason.

- pack_greedy    : highest score first, stop at the first doc that doesn't fit.
- pack_density   : highest score-per-character first, skip-and-continue —
                   favors small dense documents over long sprawling ones.
- pack_exact     : 0/1 knapsack by dynamic programming, provably optimal
                   within a 64-character bucket rounding of the budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .budget import Budget
from .doc import Doc


@dataclass
class Packing:
    included: list[Doc] = field(default_factory=list)
    dropped: list[tuple[Doc, str]] = field(default_factory=list)
    used_chars: int = 0
    usable_chars: int = 0
    total_score: float = 0.0

    @property
    def utilization(self) -> float:
        if self.usable_chars <= 0:
            return 0.0
        return round(min(self.used_chars / self.usable_chars, 1.0), 4)


def _framed(doc: Doc, budget: Budget) -> int:
    return len(doc.text) + budget.per_doc_overhead


def _record(packing: Packing, docs: list[Doc], budget: Budget) -> Packing:
    packing.used_chars = sum(_framed(d, budget) for d in packing.included)
    packing.usable_chars = budget.usable()
    packing.total_score = round(sum(d.score for d in packing.included), 6)
    return packing


def pack_greedy(docs: list[Doc], budget: Budget) -> Packing:
    """Classic greedy: highest score first, stop at the first non-fitting doc."""
    packing = Packing()
    usable = budget.usable()
    used = 0
    for doc in sorted(docs, key=lambda d: (-d.score, d.id)):
        cost = _framed(doc, budget)
        if used + cost <= usable:
            packing.included.append(doc)
            used += cost
        elif used == 0 and cost > usable:
            packing.dropped.append((doc, "too_large_alone"))
        else:
            packing.dropped.append((doc, "budget_exhausted"))
    return _record(packing, docs, budget)


def pack_density(docs: list[Doc], budget: Budget) -> Packing:
    """Value-per-character first, skip-and-continue past items that don't fit."""
    packing = Packing()
    usable = budget.usable()
    used = 0
    for doc in sorted(docs, key=lambda d: (-(d.score / max(len(d.text), 1)), d.id)):
        cost = _framed(doc, budget)
        if cost > usable:
            packing.dropped.append((doc, "too_large_alone"))
            continue
        if used + cost <= usable:
            packing.included.append(doc)
            used += cost
        else:
            packing.dropped.append((doc, "too_large_skipped"))
    return _record(packing, docs, budget)


def pack_exact(docs: list[Doc], budget: Budget, granularity: int = 1) -> Packing:
    """0/1 knapsack by dynamic programming — provably optimal total score.
    granularity=1 is exact per-character DP (cheap at realistic budgets: the
    table is docs × usable_chars). Larger values trade optimality for memory
    on very large windows; rounding is conservative (capacity floors)."""
    packing = Packing()
    usable = budget.usable()
    if granularity <= 0:
        granularity = 1
    items = []
    for doc in sorted(docs, key=lambda d: d.id):
        cost = _framed(doc, budget)
        if cost <= usable:
            buckets = max(1, -(-cost // granularity))  # ceil division
            items.append((doc, cost, buckets, doc.score))
    capacity = usable // granularity   # floor: conservative, never exceeds budget
    n = len(items)
    # dp[b] = (best_score, chosen_mask); reconstruct via masks for determinism
    dp = [0.0] * (capacity + 1)
    choice: list[list[bool]] = [[False] * (capacity + 1) for _ in range(n)]
    for i, (_, _, buckets, score) in enumerate(items):
        for b in range(capacity, buckets - 1, -1):
            cand = dp[b - buckets] + score
            if cand > dp[b]:
                dp[b] = cand
                choice[i][b] = True
    b = max(range(capacity + 1), key=lambda x: dp[x])
    chosen: list[Doc] = []
    chosen_set = set()
    for i in range(n - 1, -1, -1):
        if choice[i][b]:
            doc, cost, buckets, _ = items[i]
            chosen.append(doc)
            chosen_set.add(doc.id)
            b -= buckets
    used = 0
    for doc in sorted(docs, key=lambda d: d.id):
        if doc in chosen:
            packing.included.append(doc)
            used += _framed(doc, budget)
        else:
            reason = "too_large_alone" if _framed(doc, budget) > usable else "not_selected"
            packing.dropped.append((doc, reason))
    packing.used_chars = used
    packing.total_score = round(sum(d.score for d in packing.included), 6)
    return packing
