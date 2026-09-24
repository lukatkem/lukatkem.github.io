"""Quality metrics — pure math comparing a tensor to its dequantized twin.

Every function accepts the same nested-list shapes quantize_tensor accepts
(1D or 2D, matching each other) and raises MetricsError on empty, ragged,
non-finite, or shape-mismatched inputs. argmax ties break to the lowest
index on both sides, so agreement is deterministic.
"""
from __future__ import annotations

import math
from typing import List, Tuple, Union

from .errors import MetricsError

Tensor = Union[List[float], List[List[float]]]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricsError(f"{name}: not a number: {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise MetricsError(f"{name}: not finite: {value!r}")
    return v


def _as_rows(values: object, name: str) -> List[List[float]]:
    """Normalize to rows (a 1D tensor becomes a single row), validating on the way."""
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise MetricsError(f"{name} must be a non-empty list")
    if isinstance(values[0], (list, tuple)):
        rows: List[List[float]] = []
        width = None
        for r, row in enumerate(values):
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                raise MetricsError(f"{name}: row {r} is empty or not a list")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise MetricsError(f"{name}: ragged row {r} has {len(row)} values, expected {width}")
            rows.append([_number(v, name) for v in row])
        return rows
    return [[_number(v, name) for v in values]]


def _paired_rows(a: Tensor, b: Tensor) -> Tuple[List[List[float]], List[List[float]]]:
    rows_a, rows_b = _as_rows(a, "a"), _as_rows(b, "b")
    if len(rows_a) != len(rows_b) or len(rows_a[0]) != len(rows_b[0]):
        raise MetricsError(f"shape mismatch: a is {len(rows_a)}x{len(rows_a[0])}, "
                           f"b is {len(rows_b)}x{len(rows_b[0])}")
    return rows_a, rows_b


def max_abs_error(a: Tensor, b: Tensor) -> float:
    """Largest absolute difference between matching elements."""
    rows_a, rows_b = _paired_rows(a, b)
    return max(abs(x - y) for ra, rb in zip(rows_a, rows_b) for x, y in zip(ra, rb))


def mean_abs_error(a: Tensor, b: Tensor) -> float:
    """Mean absolute difference over all elements."""
    rows_a, rows_b = _paired_rows(a, b)
    total = sum(abs(x - y) for ra, rb in zip(rows_a, rows_b) for x, y in zip(ra, rb))
    return total / (len(rows_a) * len(rows_a[0]))


def snr_db(a: Tensor, b: Tensor) -> float:
    """Signal-to-noise ratio in dB: a is the signal, a - b is the noise.

    10·log10(Σa² / Σ(a−b)²). Identical inputs give +inf; an all-zero signal
    with any noise gives -inf.
    """
    rows_a, rows_b = _paired_rows(a, b)
    signal = 0.0
    noise = 0.0
    for ra, rb in zip(rows_a, rows_b):
        for x, y in zip(ra, rb):
            signal += x * x
            noise += (x - y) ** 2
    if noise == 0.0:
        return math.inf
    if signal == 0.0:
        return -math.inf
    return 10.0 * math.log10(signal / noise)


def _argmax(row: List[float]) -> int:
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def argmax_agreement(a: Tensor, b: Tensor) -> float:
    """Fraction of rows whose largest element lands on the same index in a and b.

    A 1D pair counts as a single row, so agreement is 1.0 or 0.0 there. This
    is the metric that decides whether a quantized model still makes the
    same prediction the original would.
    """
    rows_a, rows_b = _paired_rows(a, b)
    hits = sum(1 for ra, rb in zip(rows_a, rows_b) if _argmax(ra) == _argmax(rb))
    return hits / len(rows_a)
