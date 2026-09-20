"""hnsw: a from-scratch, pure-standard-library HNSW approximate-NN index.

No numpy, no faiss — only the Python standard library.  Implements the
Hierarchical Navigable Small World graph of Malkov & Yashunin (2016) with
deterministic construction, JSON persistence, and three distance metrics.

Example::

    from hnsw import HNSW

    index = HNSW(dim=64, metric="cosine")
    index.add_batch({f"doc{i}": vec for i, vec in enumerate(vectors)})
    hits = index.search(query, k=10, ef_search=64)   # [(id, distance), ...]

Single-threaded by design; see :mod:`hnsw.hnsw` for details.
"""

from .distance import (
    METRICS,
    cosine,
    cosine_normalized,
    euclidean,
    get_metric,
    inner_product,
    normalize,
)
from .hnsw import HNSW

__version__ = "1.0.0"

__all__ = [
    "HNSW",
    "METRICS",
    "cosine",
    "cosine_normalized",
    "euclidean",
    "get_metric",
    "inner_product",
    "normalize",
    "__version__",
]
