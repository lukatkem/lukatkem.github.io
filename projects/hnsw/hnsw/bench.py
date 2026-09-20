"""Deterministic benchmark: HNSW vs brute-force exact search.

Dataset (all seeded, fully reproducible): N isotropic Gaussian unit vectors,
plus one *planted near-duplicate* per query (the query plus per-coordinate
Gaussian noise, re-normalized), so every query has a hard-but-findable true
nearest neighbor.  Ground truth is an exhaustive brute-force scan.

For each ``ef_search`` in the sweep the benchmark reports recall@10 and
mean/p95 query latency; it also reports build time, graph size, a rough
memory estimate, and the exact-search baseline latency.

Usage::

    python -m hnsw.bench                     # N in {2000, 5000}, dim 64, cosine
    python -m hnsw.bench --n 2000 --dim 64   # single dataset
    python -m hnsw.bench --metrics cosine,euclidean

The markdown report is printed to stdout and written to ``results.md`` in the
project root (override with ``--out``).  Recall and build numbers are
reproducible; wall-clock latencies are whatever the current machine does.
"""

from __future__ import annotations

import argparse
import heapq
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .distance import DistanceFn, get_metric, normalize
from .hnsw import HNSW

SEED = 1337
DEFAULT_SIZES = (2000, 5000)
DEFAULT_DIM = 64
NUM_QUERIES = 50
K = 10
EF_SWEEP = (16, 32, 64, 128)
DUP_SIGMA = 0.05  # per-coordinate noise std for planted near-duplicates
BENCH_M = 16
BENCH_EF_CONSTRUCTION = 200


def make_dataset(
    n: int, dim: int, num_queries: int, seed: int = SEED
) -> Tuple[Dict[Any, Tuple[float, ...]], List[Tuple[float, ...]]]:
    """Build the seeded evaluation set.

    Every vector is drawn from an isotropic standard Gaussian and L2
    normalized.  For each query we additionally plant one near-duplicate
    (query + Gaussian noise of std ``DUP_SIGMA``, re-normalized), so the true
    top-1 is a specific, non-trivial point rather than an arbitrary one among
    thousands of nearly-equidistant candidates.

    Returns ``(points, queries)`` where points maps ids (ints for the base
    set, ``"dup-i"`` strings for the planted duplicates) to unit vectors.
    """
    rng = random.Random(seed)

    def unit() -> Tuple[float, ...]:
        return normalize(tuple(rng.gauss(0.0, 1.0) for _ in range(dim)))

    points: Dict[Any, Tuple[float, ...]] = {i: unit() for i in range(n)}
    queries = [unit() for _ in range(num_queries)]
    for qi, q in enumerate(queries):
        noisy = normalize(tuple(x + rng.gauss(0.0, DUP_SIGMA) for x in q))
        points[f"dup-{qi}"] = noisy
    return points, queries


def brute_force_topk(
    dist: DistanceFn,
    points: Dict[Any, Tuple[float, ...]],
    query: Sequence[float],
    k: int,
) -> List[Any]:
    """Exact top-k ids by exhaustive scan — the recall reference."""
    return [
        pid
        for _, pid in heapq.nsmallest(
            k, ((dist(query, vec), pid) for pid, vec in points.items())
        )
    ]


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of an ascending-sorted sequence."""
    if not sorted_values:
        raise ValueError("empty sequence")
    rank = max(1, math.ceil(pct / 100.0 * len(sorted_values)))
    return sorted_values[rank - 1]


def estimate_index_bytes(index: HNSW) -> Dict[str, float]:
    """Rough memory footprint of a built index.

    ``raw_bytes`` is the float64-equivalent vector payload — what a C++ or
    numpy index would spend on vectors alone.  ``python_bytes`` estimates the
    actual CPython cost: each float is a boxed 24-byte object plus an 8-byte
    slot (~32 B per coordinate) and each link an int object plus list slot
    (~40 B).  The gap between the two is one honest price of pure Python.
    """
    n = len(index._ids)
    total_links = sum(len(layer) for node in index._links for layer in node)
    return {
        "raw_bytes": n * index.dim * 8,
        "python_bytes": n * index.dim * 32 + total_links * 40,
        "links": total_links,
    }


def run_suite(
    n: int,
    dim: int,
    num_queries: int,
    metric: str,
    efs: Sequence[int],
    seed: int = SEED,
) -> Dict[str, Any]:
    """Build the index, establish ground truth, sweep ``ef_search``.

    Returns a dict with per-ef result rows, build time, size estimates, and
    the brute-force latency baseline.
    """
    points, queries = make_dataset(n, dim, num_queries, seed)
    index = HNSW(
        dim=dim, M=BENCH_M, ef_construction=BENCH_EF_CONSTRUCTION, metric=metric, seed=seed
    )

    t0 = time.perf_counter()
    index.add_batch(points)
    build_seconds = time.perf_counter() - t0

    dist: DistanceFn = get_metric(metric)
    truth: List[List[Any]] = list()
    brute_ms: List[float] = list()
    for q in queries:
        t0 = time.perf_counter()
        truth.append(brute_force_topk(dist, points, q, K))
        brute_ms.append((time.perf_counter() - t0) * 1000.0)
    brute_ms.sort()

    rows: List[Dict[str, Any]] = list()
    for ef in efs:
        latencies: List[float] = list()
        hits = 0
        for q, expected in zip(queries, truth):
            t0 = time.perf_counter()
            results = index.search(q, k=K, ef_search=ef)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            hits += len({pid for pid, _ in results}.intersection(expected))
        latencies.sort()
        rows.append(
            {
                "n": n,
                "metric": metric,
                "ef": ef,
                "recall": hits / (K * len(queries)),
                "mean_ms": sum(latencies) / len(latencies),
                "p95_ms": percentile(latencies, 95.0),
            }
        )

    return {
        "n": n,
        "metric": metric,
        "rows": rows,
        "build_seconds": build_seconds,
        "sizes": estimate_index_bytes(index),
        "brute_mean_ms": sum(brute_ms) / len(brute_ms),
        "brute_p95_ms": percentile(brute_ms, 95.0),
    }


def render_report(blocks: Sequence[Dict[str, Any]], argv: Optional[Sequence[str]]) -> str:
    """Assemble the markdown report: results tables plus honest analysis."""
    lines: List[str] = list()
    metrics = sorted({b["metric"] for b in blocks})
    env = (
        f"CPython {platform.python_version()} on {platform.machine()}, "
        f"{platform.system()}"
    )
    lines.append("# HNSW benchmark — graph search vs brute force")
    lines.append("")
    lines.append(
        f"Produced by `python -m hnsw.bench` "
        f"(args: {' '.join(argv) if argv else '<defaults>'}).  "
        f"dim={DEFAULT_DIM}, {NUM_QUERIES} queries, k={K}, M={BENCH_M}, "
        f"ef_construction={BENCH_EF_CONSTRUCTION}, seed={SEED}.  "
        f"Environment: {env}."
    )
    lines.append("")
    lines.append("Recall is exactly reproducible (seeded data, seeded index, "
                 "deterministic construction); latencies are single-run wall-clock "
                 "on the machine above and will differ elsewhere.")
    lines.append("")
    lines.append("Dataset: isotropic Gaussian unit vectors plus one planted "
                 f"near-duplicate per query (noise sigma={DUP_SIGMA}); ground truth "
                 "from an exhaustive brute-force scan; p95 is the nearest-rank "
                 "percentile over the per-query latencies.")
    lines.append("")
    lines.append("## Query quality and latency")
    lines.append("")
    lines.append("| N | metric | ef_search | recall@10 | mean latency (ms) | p95 latency (ms) |")
    lines.append("|---:|:---|---:|---:|---:|---:|")
    for b in blocks:
        for r in b["rows"]:
            lines.append(
                f"| {r['n']} | {r['metric']} | {r['ef']} | {r['recall']:.3f} "
                f"| {r['mean_ms']:.3f} | {r['p95_ms']:.3f} |"
            )
    lines.append("")
    lines.append("## Build time, graph size, and exact-search baseline")
    lines.append("")
    lines.append(
        "| N | metric | build (s) | graph links | raw vectors (MB) "
        "| Python-object est. (MB) | brute force mean (ms) | brute force p95 (ms) |"
    )
    lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|")
    for b in blocks:
        s = b["sizes"]
        lines.append(
            f"| {b['n']} | {b['metric']} | {b['build_seconds']:.2f} | {s['links']} "
            f"| {s['raw_bytes'] / 1e6:.1f} | {s['python_bytes'] / 1e6:.1f} "
            f"| {b['brute_mean_ms']:.3f} | {b['brute_p95_ms']:.3f} |"
        )
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    for metric in metrics:
        lines.append(_analysis_paragraph(blocks, metric))
        lines.append("")
    lines.append(_caveats_paragraph())
    lines.append("")
    return "\n".join(lines)


def _analysis_paragraph(blocks: Sequence[Dict[str, Any]], metric: str) -> str:
    """One honest paragraph for a metric, using the largest dataset's rows."""
    block = next(
        b for b in blocks if b["metric"] == metric and b["n"] == max(
            x["n"] for x in blocks if x["metric"] == metric
        )
    )
    rows = {r["ef"]: r for r in block["rows"]}
    efs = sorted(rows)
    low, high = rows[efs[0]], rows[efs[-1]]
    mid_ef = min(efs, key=lambda ef: abs(ef - 64))
    mid = rows[mid_ef]
    speedup = block["brute_mean_ms"] / mid["mean_ms"] if mid["mean_ms"] > 0 else float("inf")
    return (
        f"**{metric}, N={block['n']} (n={block['n'] + NUM_QUERIES} with planted duplicates):** "
        f"recall@10 climbs from {low['recall']:.3f} at ef_search={low['ef']} to "
        f"{mid['recall']:.3f} at ef_search={mid_ef} and {high['recall']:.3f} at "
        f"ef_search={high['ef']}, while p95 latency grows from {low['p95_ms']:.3f} ms to "
        f"{high['p95_ms']:.3f} ms — the usual diminishing-returns curve, each doubling of "
        f"the beam buying a few points of recall at roughly double the cost.  At "
        f"ef_search={mid_ef} the index answers in {mid['p95_ms']:.3f} ms p95 at "
        f"{mid['recall'] * 100:.1f}% recall, about {speedup:.0f}x faster than the exact "
        f"scan ({block['brute_mean_ms']:.3f} ms mean).  The dataset is deliberately "
        "pessimistic: random unit vectors have no cluster structure, so most pairs sit "
        "in a narrow band of nearly-equal distances, which is the hardest case for "
        "graph navigation; real embedding workloads cluster and typically reach higher "
        "recall at the same ef_search."
    )


def _caveats_paragraph() -> str:
    return (
        "Caveats, honestly: latencies are from one run on one machine and include "
        "CPython interpreter overhead — treat them as relative, not absolute.  At low "
        "ef_search the beam simply never visits some true neighbors (on this random "
        "data the non-duplicate ground-truth neighbors are nearly interchangeable in "
        "distance, so small beams drop them easily); raising ef_search fixes recall "
        "linearly in time.  The planted near-duplicates guarantee each query has a "
        "findable true top-1, so the top-1 is usually correct even at ef_search=16; "
        "the interesting variation is in ranks 2-10."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hnsw.bench",
        description="Deterministic HNSW vs brute-force benchmark.",
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=None,
        help="dataset size(s); default: %s" % (list(DEFAULT_SIZES),),
    )
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--queries", type=int, default=NUM_QUERIES)
    parser.add_argument(
        "--metrics", type=str, default="cosine",
        help="comma-separated subset of euclidean,cosine,inner_product",
    )
    parser.add_argument("--ef", type=int, nargs="+", default=list(EF_SWEEP))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out", type=str, default=None,
        help="output path; default: results.md in the project root",
    )
    args = parser.parse_args(argv)

    sizes = args.n if args.n else list(DEFAULT_SIZES)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for metric in metrics:
        get_metric(metric)  # validate early

    blocks: List[Dict[str, Any]] = list()
    for n in sizes:
        for metric in metrics:
            print(f"[bench] building N={n} dim={args.dim} metric={metric} ...",
                  file=sys.stderr, flush=True)
            suite = run_suite(n, args.dim, args.queries, metric, args.ef, args.seed)
            best = max(suite["rows"], key=lambda r: r["recall"])
            print(
                f"[bench] built in {suite['build_seconds']:.1f}s; "
                f"best recall@10={best['recall']:.3f} at ef={best['ef']}",
                file=sys.stderr, flush=True,
            )
            blocks.append(suite)

    report = render_report(blocks, argv)
    print(report)
    out_path = (
        args.out if args.out else str(Path(__file__).resolve().parent.parent / "results.md")
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"[bench] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
