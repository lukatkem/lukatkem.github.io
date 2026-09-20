"""Tests for the HNSW index and its distance metrics.

Run from the project root:

    python -m pytest tests/ -q

Everything is seeded, so every assertion is deterministic: the statistical
recall thresholds below either always pass or always fail on a given CPython
version, never flake.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from hnsw import HNSW
from hnsw.distance import cosine, euclidean, get_metric, inner_product, normalize

METRIC_FNS = {
    "euclidean": euclidean,
    "cosine": cosine,
    "inner_product": inner_product,
}


def brute_force_topk(
    metric: str, points: Dict[Any, Sequence[float]], query: Sequence[float], k: int
) -> List[Any]:
    """Exact reference: exhaustive scan under the same distance convention."""
    dist = METRIC_FNS[metric]
    ranked = sorted((dist(query, vec), pid) for pid, vec in points.items())
    return [pid for _, pid in ranked[:k]]


def recall_at_k(predicted: Sequence[Any], expected: Sequence[Any], k: int) -> float:
    return len(set(predicted[:k]).intersection(expected[:k])) / k


# --------------------------------------------------------------------- #
# Exactness on a tiny index                                             #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_exactness_on_tiny_index(metric: str) -> None:
    """With ef_search >= N the beam covers the whole graph: recall must be 1.0."""
    rng = random.Random(7)
    points = {i: tuple(rng.uniform(-1.0, 1.0) for _ in range(8)) for i in range(50)}
    index = HNSW(dim=8, M=8, ef_construction=32, metric=metric, seed=7)
    index.add_batch(points)
    for _ in range(5):
        query = tuple(rng.uniform(-1.0, 1.0) for _ in range(8))
        truth = brute_force_topk(metric, points, query, 3)
        got = [pid for pid, _ in index.search(query, k=3, ef_search=50)]
        assert recall_at_k(got, truth, 3) == 1.0


# --------------------------------------------------------------------- #
# Statistical recall at realistic size                                  #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_statistical_recall_at_ef_64(metric: str) -> None:
    """Mean recall@10 >= 0.9 on 1000 random 32-d points, 30 queries, ef=64."""
    rng = random.Random(42)
    n, dim = 1000, 32
    points = {i: [rng.gauss(0.0, 1.0) for _ in range(dim)] for i in range(n)}
    queries = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(30)]

    index = HNSW(dim=dim, M=16, ef_construction=100, metric=metric, seed=1337)
    index.add_batch(points)

    total = 0.0
    for query in queries:
        truth = brute_force_topk(metric, points, query, 10)
        got = [pid for pid, _ in index.search(query, k=10, ef_search=64)]
        total += recall_at_k(got, truth, 10)
    assert total / len(queries) >= 0.9


# --------------------------------------------------------------------- #
# Metric behavior                                                       #
# --------------------------------------------------------------------- #


def test_metric_conventions() -> None:
    """Each metric returns sane values under known geometry."""
    a = (1.0, 2.0, 2.0)  # |a| = 3
    b = (2.0, -2.0, 1.0)  # |b| = 3, a.b = 0 → orthogonal
    opposite = (-2.0, -4.0, -4.0)

    assert cosine(a, b) == pytest.approx(1.0)  # orthogonal → 1
    assert cosine(a, a) == pytest.approx(0.0)  # identical → 0
    assert cosine(a, opposite) == pytest.approx(2.0)  # antipodal → 2
    assert cosine(a, (0.0, 0.0, 0.0)) == pytest.approx(1.0)  # zero-vector convention

    assert euclidean((0.0, 0.0), (3.0, 4.0)) == pytest.approx(25.0)  # squared!
    assert euclidean(a, b) == pytest.approx(18.0)  # (-1, 4, 1)

    assert inner_product(a, b) == pytest.approx(0.0)
    assert inner_product((1.0,), (2.0,)) == pytest.approx(-2.0)  # negated dot

    assert normalize((3.0, 4.0)) == pytest.approx((0.6, 0.8))
    assert normalize((0.0, 0.0)) == (0.0, 0.0)


def test_inner_product_ranking_differs_from_euclidean() -> None:
    """Crafted case: MIPS prefers the big-norm point, L2 prefers the close one."""
    points: Dict[str, Tuple[float, float]] = {
        "big": (3.0, 0.0),  # dot 3.0 with the query, but L2^2 = 5.0 away
        "near": (1.4, 1.4),  # dot 2.8 with the query, but L2^2 = 0.32 away
    }
    query = (1.0, 1.0)
    # The crafted geometry is what we think it is:
    assert inner_product(query, points["big"]) < inner_product(query, points["near"])
    assert euclidean(query, points["big"]) > euclidean(query, points["near"])

    ip = HNSW(dim=2, M=4, ef_construction=8, metric="inner_product", seed=1)
    ip.add_batch(points)
    l2 = HNSW(dim=2, M=4, ef_construction=8, metric="euclidean", seed=1)
    l2.add_batch(points)

    assert ip.search(query, k=2, ef_search=8)[0][0] == "big"
    assert l2.search(query, k=2, ef_search=8)[0][0] == "near"


def test_input_validation() -> None:
    """Dimension mismatches, bad metric names, duplicate ids all fail loudly."""
    with pytest.raises(ValueError):
        euclidean((1.0, 2.0), (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        cosine((1.0,), ())
    with pytest.raises(ValueError):
        inner_product((1.0,), (2.0, 3.0))
    with pytest.raises(ValueError):
        get_metric("manhattan")

    with pytest.raises(ValueError):
        HNSW(dim=3, metric="manhattan")

    index = HNSW(dim=3)
    with pytest.raises(ValueError):
        index.add(1, (1.0, 2.0))  # wrong dimension
    index.add(1, (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        index.add(1, (9.0, 9.0, 9.0))  # duplicate id
    with pytest.raises(ValueError):
        index.add(2, (1.0, 2.0, float("nan")))  # non-finite
    with pytest.raises(ValueError):
        index.search((1.0, 2.0))  # wrong query dimension


# --------------------------------------------------------------------- #
# Search result shape                                                   #
# --------------------------------------------------------------------- #


def test_search_returns_k_sorted_unique() -> None:
    """Results are k unique ids sorted ascending by distance."""
    rng = random.Random(5)
    points = {i: [rng.gauss(0.0, 1.0) for _ in range(12)] for i in range(100)}
    index = HNSW(dim=12, M=8, ef_construction=64, metric="cosine", seed=5)
    index.add_batch(points)
    query = [rng.gauss(0.0, 1.0) for _ in range(12)]

    results = index.search(query, k=7, ef_search=64)
    assert len(results) == 7
    distances = [d for _, d in results]
    assert distances == sorted(distances)
    assert all(d >= 0.0 for d in distances)
    ids = [pid for pid, _ in results]
    assert len(set(ids)) == 7
    # Returned distances match a direct computation under the same metric.
    for pid, d in results:
        assert d == pytest.approx(cosine(query, points[pid]))

    # k larger than the index returns everything available.
    everything = index.search(query, k=500, ef_search=64)
    assert len(everything) == 100

    # An empty index searches cleanly.
    assert HNSW(dim=12).search(query, k=3) == []


# --------------------------------------------------------------------- #
# Persistence                                                           #
# --------------------------------------------------------------------- #


def test_save_load_roundtrip(tmp_path) -> None:
    """Search results are identical after save/load, and files are deterministic."""
    rng = random.Random(3)
    points = {f"vec-{i}": [rng.gauss(0.0, 1.0) for _ in range(16)] for i in range(120)}
    queries = [[rng.gauss(0.0, 1.0) for _ in range(16)] for _ in range(5)]

    index = HNSW(dim=16, M=8, ef_construction=64, metric="cosine", seed=5)
    index.add_batch(points)
    expected = [index.search(q, k=5, ef_search=32) for q in queries]

    path = str(tmp_path / "index.json")
    index.save(path)
    loaded = HNSW.load(path)
    assert [loaded.search(q, k=5, ef_search=32) for q in queries] == expected
    assert len(loaded) == len(index) == 120
    assert loaded.metric == "cosine" and loaded.dim == 16

    # Same state saved twice → byte-identical file.
    index.save(str(tmp_path / "index-again.json"))
    assert (tmp_path / "index-again.json").read_bytes() == (tmp_path / "index.json").read_bytes()

    # A file that is not an HNSW index is rejected.
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "nope"}', encoding="utf-8")
    with pytest.raises(ValueError):
        HNSW.load(str(bad))


def test_save_load_preserves_int_ids_and_tombstones(tmp_path) -> None:
    """Integer ids survive the JSON roundtrip, as do removed points."""
    rng = random.Random(9)
    points = {i: [rng.gauss(0.0, 1.0) for _ in range(8)] for i in range(60)}
    index = HNSW(dim=8, M=6, ef_construction=48, metric="euclidean", seed=9)
    index.add_batch(points)
    index.remove(3)
    index.remove(41)

    path = str(tmp_path / "idx.json")
    index.save(path)
    loaded = HNSW.load(path)

    query = points[0]
    for idx in (index, loaded):
        got = [pid for pid, _ in idx.search(query, k=50, ef_search=60)]
        assert 3 not in got and 41 not in got
    assert len(loaded) == 58
    assert loaded.num_deleted == 2


# --------------------------------------------------------------------- #
# Deletion                                                              #
# --------------------------------------------------------------------- #


def test_remove_tombstones() -> None:
    """Removed points never come back, but the graph keeps answering."""
    rng = random.Random(21)
    points = {i: [rng.gauss(0.0, 1.0) for _ in range(8)] for i in range(80)}
    index = HNSW(dim=8, M=8, ef_construction=64, metric="euclidean", seed=21)
    index.add_batch(points)

    victim = 7
    assert index.search(points[victim], k=5, ef_search=64)[0][0] == victim
    assert len(index) == 80

    index.remove(victim)
    results = index.search(points[victim], k=10, ef_search=64)
    assert all(pid != victim for pid, _ in results)
    assert len(results) == 10
    assert len(index) == 79
    assert index.num_deleted == 1

    index.remove(victim)  # idempotent on already-removed ids
    with pytest.raises(KeyError):
        index.remove("nope")

    # A deleted entry point is still usable for navigation.
    small = HNSW(dim=2, metric="euclidean")
    small.add("a", (0.0, 0.0))
    small.add("b", (1.0, 0.0))
    small.remove("a")
    assert small.search((0.0, 0.0), k=1, ef_search=4) == [("b", 1.0)]


# --------------------------------------------------------------------- #
# Determinism                                                           #
# --------------------------------------------------------------------- #


def test_determinism_same_seed_same_structure() -> None:
    """Same seed + same insertion order → same graph, same answers."""
    rng = random.Random(11)
    points = {i: [rng.gauss(0.0, 1.0) for _ in range(16)] for i in range(150)}
    queries = [[rng.gauss(0.0, 1.0) for _ in range(16)] for _ in range(5)]

    def build() -> HNSW:
        index = HNSW(dim=16, M=8, ef_construction=64, metric="cosine", seed=1337)
        index.add_batch(points)
        return index

    first, second = build(), build()
    assert first._entry == second._entry
    assert first._max_level == second._max_level
    assert first._levels == second._levels
    assert first._links == second._links
    for q in queries:
        assert first.search(q, k=5, ef_search=32) == second.search(q, k=5, ef_search=32)


def test_layer_assignment_distribution() -> None:
    """Levels follow the geometric distribution: mostly 0, rarely high."""
    index = HNSW(dim=2, M=16, seed=1337)
    for i in range(500):  # bypass add() to sample levels cheaply
        assert index._random_level() >= 0
    # With mL = 1/ln(16) ~ 0.36, P(level >= 1) = 1/16, P(level >= 3) = 1/4096.
    # 500 draws should contain at least one level >= 1 but not explode.
    index2 = HNSW(dim=2, M=16, seed=1337)
    levels = [index2._random_level() for _ in range(2000)]
    assert sum(1 for lv in levels if lv >= 1) >= 50
    assert max(levels) <= 5
    assert math.isclose(index2.mL, 1.0 / math.log(16))
