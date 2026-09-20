# hnsw — approximate nearest neighbors in pure Python

A from-scratch implementation of **HNSW (Hierarchical Navigable Small World
graphs)** — the algorithm behind `hnswlib`, Faiss's graph index, and most
modern vector databases — using only the Python standard library. No numpy,
no faiss, no C extensions. Deterministic construction, JSON persistence, a
built-in brute-force benchmark, and a test suite.

Implements the paper: Malkov & Yashunin, *Efficient and robust approximate
nearest neighbor search using Hierarchical Navigable Small World graphs*
(IEEE TPAMI 2018, arXiv:1603.09320) — Algorithms 1, 2, 4, and 5.

## Why approximate nearest neighbors

Exact k-NN is a brute-force scan: for every query you compute the distance to
all N vectors. That is `O(N · d)` per query, every time. At 10 million
vectors of dimension 768, one query costs ~7.7 billion multiply-adds — tens
of seconds on a CPU, for one query. No index on top of exact search changes
that arithmetic; only giving up *guaranteed* exactness does.

Approximate nearest-neighbor (ANN) search trades a controlled amount of
recall — you find, say, 97% of the true top-10 — for orders of magnitude less
work per query. For most applications (recommendation, dedup, RAG retrieval),
returning 97 of the true top 100 a thousand times faster is strictly better
than returning all 100 slowly.

## How HNSW works

HNSW builds a "small world" proximity graph and layers it like a skip list.
Every point lives on layer 0. A geometric subset is promoted to layer 1, a
smaller subset to layer 2, and so on — the expected number of layers is
`log_M(N)`. Links on upper layers are long-range; layer 0 is dense.

```
level 2:    E ─────────────────── J                ← entry point lives here;
            │                     │                  ~N/M² nodes, long links
level 1:    E ────── J ───────────┼───── A ───
            │        │            │      │
level 0:    E──A──J──B──C──D──F───J2──A2─G──H──K  ← every node, ~2M links each
```

A search starts at the single **entry point** on the top layer and walks
greedily toward the query: hop to any neighbor that is closer, and when no
neighbor improves, drop one layer down. The upper layers act as an express
lane, getting you into the right neighborhood in `O(log N)` hops. On layer 0
the walk stops being purely greedy and becomes a **beam search** of width
`ef` (the paper's SEARCH-LAYER): keep the `ef` best candidates seen so far,
expand the nearest unexpanded one, and stop when the closest frontier node is
farther than the worst of the kept set. The final `k` results are the best of
the beam.

Two details make the graph navigable rather than a hub-and-spoke mess:

- **Heuristic pruning** (Algorithm 4). When picking a node's M links, a
  candidate is kept only if it is closer to the node than to every
  already-selected neighbor — i.e. no existing link already "covers" that
  direction. This forces links to spread geometrically around each node,
  which is what makes greedy descent actually converge.
- **Asymmetric degree caps.** Nodes keep at most `M` links on upper layers
  but `2·M` on layer 0 (`M_max0`), because layer 0 is where all the real
  walking happens. Overfull lists are re-pruned with the same heuristic.

Node levels are drawn as `floor(-ln(U) · mL)` with `mL = 1/ln(M)`, exactly as
in the paper.

## Complexity, honestly

| | brute force | HNSW |
|---|---|---|
| Query | `O(N · d)` distance computations | `O(ef · M · log N)`-ish visits |
| Build | free | `O(N · efC · M · log N)`-ish, paid once |
| Memory | the vectors | vectors + `~N · M · 2` links |

The "-ish" is doing real work in that table: HNSW's query cost is
logarithmic in practice but its constants are `ef` and `M`, which you tune
against recall. That is the actual deal — you buy recall with `ef`.

Measured on this implementation (cosine, dim 64, M=16, efC=200, 50 queries,
seeded data — full tables in [results.md](results.md)):

| N | ef_search | recall@10 | p95 latency | brute-force mean |
|---:|---:|---:|---:|---:|
| 2000 | 64 | 0.996 | 0.67 ms | 2.03 ms |
| 2000 | 128 | 1.000 | 0.89 ms | 2.03 ms |
| 5000 | 64 | 0.970 | 1.68 ms | 5.10 ms |
| 5000 | 128 | 0.992 | 2.21 ms | 5.10 ms |

Recall rises with `ef` (0.93 → 1.00 at N=2000 sweeping ef 16 → 128) while
latency scales roughly linearly with `ef`. Note the asymptotics only show at
scale: at N=5000 the graph is only ~3-4x faster than the brute-force scan,
because both costs are dominated by Python interpreter overhead per distance.
The gap widens linearly with N — the brute-force row doubles when N doubles,
the HNSW row does not — which is exactly the point of the structure.

## Usage

```python
from hnsw import HNSW

index = HNSW(dim=64, M=16, ef_construction=200, metric="cosine", seed=1337)
index.add_batch({f"doc{i}": vec for i, vec in enumerate(vectors)})

hits = index.search(query, k=10, ef_search=64)   # [(id, distance), ...] ascending

index.save("index.json")
index2 = HNSW.load("index.json")                 # bit-identical search behavior
```

Benchmark CLI:

```bash
python -m hnsw.bench                  # N in {2000, 5000}, dim 64, cosine
python -m hnsw.bench --n 2000 --dim 64 --metrics cosine,euclidean
```

Tests (all deterministic — fixed seeds, no statistical flaking):

```bash
python -m pytest tests/ -q
```

### Distance conventions

All metrics are returned as distances (smaller = more similar):

- `"euclidean"` — true L2. The graph tracks **squared** L2 internally (the
  sqrt is monotone, so rankings are identical) and applies `math.sqrt` once,
  to the final results.
- `"cosine"` — `1 - cosine_similarity`, in [0, 2]. Vectors are L2-normalized
  **once at insert/query time**, so each comparison is a single dot product;
  nothing is renormalized per comparison.
- `"inner_product"` — **negated** dot product (MIPS convention: largest dot →
  smallest distance). Inner product is not a metric, so graph indexes
  generally reach lower recall on it; that is a property of MIPS, not a bug.

## Design notes and trade-offs

- **Single-threaded, no locks.** One index per thread, or guard it yourself.
  Pure-Python GIL economics make internal threading pointless anyway.
- **Deterministic.** Level draws come from a seeded `random.Random`, all
  containers are insertion-ordered, and heap ties break on stable integer
  positions. Same seed + same insertion order ⇒ byte-identical save files and
  bit-identical search results.
- **Deletions are tombstones.** `remove(id)` is O(1): the point is marked
  deleted and filtered from results, but its storage and links stay and
  searches still route *through* it, so connectivity and recall are
  unaffected while latency creeps up as the deleted fraction grows. Memory is
  only reclaimed by rebuilding. Hard deletes need neighborhood re-wiring or a
  full rebuild; this implementation takes the cheap option and says so.
- **JSON persistence.** Compact and inspectable, but O(text) in size and
  parse time; a production build would use a binary format.
- **Storage layout.** Nodes live in flat parallel lists indexed by an integer
  position, with an id↔position map; heaps and link lists carry positions,
  not user ids, so hot-path tuples stay cheap and comparable.

## Limitations of the pure-Python build

- **Throughput.** Each distance computation costs ~1-3 µs of interpreted
  Python; a C++ HNSW with SIMD does the same work 10-50x faster per core, and
  faiss adds multi-threading and batch queries on top. If your latency budget
  is microseconds or your QPS is high, pure Python is not the answer.
- **Memory.** Every float is a boxed 24-byte object plus an 8-byte list slot:
  ~32 B per coordinate vs 4 B in faiss (float32). Each of the ~55k links in
  the N=2000 index costs another ~40 B. Measured estimate for N=2000, dim 64:
  **6.4 MB vs ~1.0 MB** of raw float64 vectors — a 6x premium that grows with
  dimensionality (1M × 768-d vectors ≈ 25 GB vs ~3 GB).
- **No vectorized batch search, no quantization, no GPU.** No IVF/PQ
  compression means memory grows linearly with N at full float precision.

**When to reach for faiss instead:** more than ~100k-1M vectors, a hard
latency/QPS SLO, RAM pressure (use IVF or PQ compression), batch workloads,
or GPUs. Below that — prototypes, tests, teaching, datasets in the low
hundreds of thousands — this implementation is correct, deterministic, and
readable, which at that scale usually matters more than raw speed.

## Project layout

```
hnsw/
├── hnsw/
│   ├── __init__.py     # public API
│   ├── distance.py     # euclidean / cosine / inner_product + validation
│   ├── hnsw.py         # the HNSW graph: add, search, remove, save/load
│   └── bench.py        # deterministic benchmark vs brute force
├── tests/test_hnsw.py  # 13 tests: exactness, recall, roundtrip, determinism
├── results.md          # benchmark output (real, reproducible numbers)
└── README.md
```
