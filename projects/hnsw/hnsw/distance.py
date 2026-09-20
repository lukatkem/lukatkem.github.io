"""Distance metrics for the HNSW index.

Every metric is expressed as a *distance*: smaller means more similar.  All
functions share the same ``(a, b) -> float`` signature and validate that both
operands are non-empty vectors of equal dimension.

- :func:`euclidean` returns the **squared** L2 distance.  The square root is
  monotonic, so rankings are unchanged, and skipping it on every graph
  traversal saves a transcendental call per comparison.  ``HNSW.search``
  applies ``math.sqrt`` once, to the final results, so users see true L2.
- :func:`cosine` returns ``1 - cosine_similarity`` in ``[0, 2]``.  The index
  L2-normalizes vectors once at insert/query time (see :func:`normalize`) so
  its hot path degenerates to a single dot product via
  :func:`cosine_normalized`; this general function recomputes norms and is
  meant for brute-force comparisons and external use.
- :func:`inner_product` returns the **negated** dot product so that
  maximum-inner-product (MIPS) search fits the smaller-is-closer convention.

All three metrics are symmetric, which the graph construction relies on.
"""

from __future__ import annotations

import math
from operator import mul
from typing import Callable, Dict, Sequence, Tuple

try:  # CPython >= 3.12: C-speed dot product, used where available.
    from math import sumprod as _dot
except ImportError:  # pragma: no cover - fallback for older interpreters
    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(map(mul, a, b))

__all__ = [
    "Vector",
    "DistanceFn",
    "euclidean",
    "cosine",
    "cosine_normalized",
    "inner_product",
    "normalize",
    "get_metric",
    "METRICS",
]

Vector = Tuple[float, ...]
DistanceFn = Callable[[Vector, Vector], float]


def _check_pair(a: Sequence[float], b: Sequence[float]) -> None:
    """Raise ``ValueError`` unless ``a`` and ``b`` are non-empty and equal-length."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("vectors must be non-empty")
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Squared Euclidean distance (L2**2) between ``a`` and ``b``.

    Ranking under the squared distance is identical to true L2; callers that
    need human-scale distances take the square root once at the end, which is
    exactly what ``HNSW.search`` does.
    """
    _check_pair(a, b)
    diff = [x - y for x, y in zip(a, b)]
    return _dot(diff, diff)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance ``1 - cos(a, b)``, in ``[0, 2]``.

    Zero vectors have no direction; by convention they are orthogonal to
    everything, yielding a distance of 1.0.
    """
    _check_pair(a, b)
    dot = _dot(a, b)
    norm_a = math.sqrt(_dot(a, a))
    norm_b = math.sqrt(_dot(b, b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def cosine_normalized(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance for vectors that already have unit L2 norm: ``1 - <a, b>``.

    This is the hot-path comparison used by ``HNSW`` when ``metric="cosine"``.
    It assumes both operands went through :func:`normalize` exactly once (the
    index does this when a point is inserted and when a query arrives) instead
    of renormalizing on every comparison.
    """
    _check_pair(a, b)
    return 1.0 - _dot(a, b)


def inner_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Negated inner product ``-<a, b>``.

    Negation turns maximum-inner-product search into a distance: the point
    with the *largest* dot product against the query gets the *smallest*
    distance.  Note that inner product is not a metric (it violates the
    triangle inequality), so graph indexes generally reach lower recall on it
    than on L2 or cosine; that is a property of MIPS itself, not of this
    implementation.
    """
    _check_pair(a, b)
    return -_dot(a, b)


def normalize(v: Sequence[float]) -> Vector:
    """Return ``v`` rescaled to unit L2 norm, as a tuple of floats.

    The zero vector cannot be normalized and is returned unchanged.
    """
    norm = math.sqrt(_dot(v, v))
    if norm == 0.0:
        return tuple(v)
    return tuple(x / norm for x in v)


METRICS: Dict[str, DistanceFn] = {
    "euclidean": euclidean,
    "cosine": cosine,
    "inner_product": inner_product,
}


def get_metric(name: str) -> DistanceFn:
    """Resolve a metric name to its function.

    Raises ``ValueError`` for unknown names.
    """
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(
            f"unknown metric {name!r}; expected one of {sorted(METRICS)}"
        ) from None
