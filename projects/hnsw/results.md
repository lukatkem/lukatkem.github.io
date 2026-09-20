# HNSW benchmark — graph search vs brute force

Produced by `python -m hnsw.bench` (args: <defaults>).  dim=64, 50 queries, k=10, M=16, ef_construction=200, seed=1337.  Environment: CPython 3.14.7 on arm64, Darwin.

Recall is exactly reproducible (seeded data, seeded index, deterministic construction); latencies are single-run wall-clock on the machine above and will differ elsewhere.

Dataset: isotropic Gaussian unit vectors plus one planted near-duplicate per query (noise sigma=0.05); ground truth from an exhaustive brute-force scan; p95 is the nearest-rank percentile over the per-query latencies.

## Query quality and latency

| N | metric | ef_search | recall@10 | mean latency (ms) | p95 latency (ms) |
|---:|:---|---:|---:|---:|---:|
| 2000 | cosine | 16 | 0.930 | 0.213 | 0.355 |
| 2000 | cosine | 32 | 0.968 | 0.328 | 0.411 |
| 2000 | cosine | 64 | 0.996 | 0.535 | 0.669 |
| 2000 | cosine | 128 | 1.000 | 0.798 | 0.889 |
| 5000 | cosine | 16 | 0.874 | 0.357 | 0.590 |
| 5000 | cosine | 32 | 0.932 | 0.514 | 0.792 |
| 5000 | cosine | 64 | 0.970 | 1.007 | 1.676 |
| 5000 | cosine | 128 | 0.992 | 1.555 | 2.209 |

## Build time, graph size, and exact-search baseline

| N | metric | build (s) | graph links | raw vectors (MB) | Python-object est. (MB) | brute force mean (ms) | brute force p95 (ms) |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 2000 | cosine | 2.93 | 55074 | 1.0 | 6.4 | 2.026 | 2.086 |
| 5000 | cosine | 11.08 | 136782 | 2.6 | 15.8 | 5.096 | 5.286 |

## Analysis

**cosine, N=5000 (n=5050 with planted duplicates):** recall@10 climbs from 0.874 at ef_search=16 to 0.970 at ef_search=64 and 0.992 at ef_search=128, while p95 latency grows from 0.590 ms to 2.209 ms — the usual diminishing-returns curve, each doubling of the beam buying a few points of recall at roughly double the cost.  At ef_search=64 the index answers in 1.676 ms p95 at 97.0% recall, about 5x faster than the exact scan (5.096 ms mean).  The dataset is deliberately pessimistic: random unit vectors have no cluster structure, so most pairs sit in a narrow band of nearly-equal distances, which is the hardest case for graph navigation; real embedding workloads cluster and typically reach higher recall at the same ef_search.

Caveats, honestly: latencies are from one run on one machine and include CPython interpreter overhead — treat them as relative, not absolute.  At low ef_search the beam simply never visits some true neighbors (on this random data the non-duplicate ground-truth neighbors are nearly interchangeable in distance, so small beams drop them easily); raising ef_search fixes recall linearly in time.  The planted near-duplicates guarantee each query has a findable true top-1, so the top-1 is usually correct even at ef_search=16; the interesting variation is in ranks 2-10.
