"""Hierarchical Navigable Small World (HNSW) approximate nearest-neighbor index.

A from-scratch, pure-standard-library implementation of the HNSW graph from
Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs" (IEEE TPAMI 2018, arXiv
1603.09320).  The paper's algorithms map to this module as follows:

- Algorithm 1 (INSERT)                     -> :meth:`HNSW.add`
- Algorithm 2 (SEARCH-LAYER)               -> :meth:`HNSW._search_layer`
- Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) -> :meth:`HNSW._select_neighbors_heuristic`
- Algorithm 5 (K-NN-SEARCH)                -> :meth:`HNSW.search`

Notes and deliberate deviations
-------------------------------
* **Single-threaded.**  Instances are not thread-safe; there is no locking of
  any kind.  Guard access externally or build one index per thread.
* **Deterministic.**  Level draws come from a seeded :class:`random.Random`
  and every container is insertion-ordered, so identical call sequences on
  identically-constructed indexes produce identical graphs and identical
  search results.
* **Deletions are tombstones.**  :meth:`HNSW.remove` marks a point deleted and
  search results filter it out, but the node and its links stay in the graph
  (traversal may still route through it) and memory is never reclaimed.  See
  the method docstring for the trade-off.
* **Distances.**  The euclidean metric tracks L2 squared internally and takes
  the square root only on final search results; the cosine metric stores
  L2-normalized vectors so each comparison is a single dot product; the inner
  product metric negates the dot so smaller means more similar.
"""

from __future__ import annotations

import heapq
import json
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .distance import DistanceFn, cosine_normalized, get_metric, normalize

__all__ = ["HNSW"]

# (distance, node position) pairs; "position" is the node's internal index.
_Candidate = Tuple[float, int]

_SAVE_FORMAT = "hnsw-json"
_SAVE_VERSION = 1


class HNSW:
    """Multi-layer proximity graph for approximate nearest-neighbor search.

    Parameters
    ----------
    dim:
        Vector dimensionality; every added vector and every query must match.
    M:
        Number of neighbors selected per node per layer at insert time, and
        the maximum number of links per node on layers > 0 (layer 0 allows
        ``2 * M``).  Typical values are 8-48: higher M means higher recall,
        more memory, and slower builds.
    ef_construction:
        Width of the candidate beam used while inserting (the paper's
        ``efConstruction``).  Higher builds a better graph, more slowly.
    mL:
        Level-generation factor.  Defaults to the paper's ``1 / ln(M)``,
        which yields roughly ``log_M(N)`` layers.
    metric:
        ``"euclidean"``, ``"cosine"``, or ``"inner_product"`` (see
        :mod:`hnsw.distance`).
    seed:
        Seed for the random level generator; fixes the graph topology.

    Single-threaded by design; see the module docstring.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        mL: Optional[float] = None,
        metric: str = "cosine",
        seed: int = 1337,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if M < 2:
            raise ValueError(f"M must be >= 2, got {M}")
        if ef_construction < 1:
            raise ValueError(f"ef_construction must be >= 1, got {ef_construction}")
        if mL is not None and mL <= 0.0:
            raise ValueError(f"mL must be > 0, got {mL}")

        self.dim = int(dim)
        self.M = int(M)
        self.M0 = 2 * self.M  # M_max0 in the paper: link cap on layer 0
        self.ef_construction = int(ef_construction)
        self.mL = 1.0 / math.log(self.M) if mL is None else float(mL)
        self.metric = metric
        self.seed = seed

        # get_metric validates the name even on the cosine fast path.
        self._dist: DistanceFn = (
            cosine_normalized if metric == "cosine" else get_metric(metric)
        )
        self._normalize_input = metric == "cosine"

        self._rng = random.Random(seed)

        # Flat storage indexed by an internal integer position ``pos``.
        self._ids: List[Any] = list()                    # pos -> external id
        self._vectors: List[Tuple[float, ...]] = list()  # pos -> prepared vector
        self._levels: List[int] = list()                 # pos -> top level of the node
        self._links: List[List[List[int]]] = list()      # pos -> layer -> neighbor pos
        self._pos: Dict[Any, int] = dict()               # external id -> pos
        self._deleted: Set[int] = set()                  # tombstoned positions
        self._entry: Optional[int] = None                # entry-point position
        self._max_level: int = -1

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Number of live (non-deleted) points in the index."""
        return len(self._ids) - len(self._deleted)

    def __contains__(self, id: Any) -> bool:
        """True if ``id`` is present (deleted points still count as present)."""
        return id in self._pos

    @property
    def num_deleted(self) -> int:
        """Number of tombstoned points still occupying graph slots."""
        return len(self._deleted)

    @property
    def max_level(self) -> int:
        """Top layer of the graph (``-1`` for an empty index)."""
        return self._max_level

    def __repr__(self) -> str:
        return (
            f"HNSW(dim={self.dim}, M={self.M}, metric={self.metric!r}, "
            f"points={len(self)}, deleted={len(self._deleted)}, "
            f"max_level={self._max_level})"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def add(self, id: Any, vector: Sequence[float]) -> None:
        """Insert one point (the paper's Algorithm 1, INSERT).

        The point draws a random top level (geometric distribution with
        factor ``mL``), is greedily descended from the entry point through
        the layers above its level, then connected on every layer it
        participates in using ``ef_construction``-wide searches and the
        diversity heuristic.  Overfull neighbor lists are re-pruned with the
        same heuristic.

        Raises
        ------
        ValueError
            If ``id`` is already present, the dimension is wrong, or the
            vector contains a non-finite value.
        """
        if id in self._pos:
            raise ValueError(f"id {id!r} already present in the index")
        vec = self._prepare(vector)
        level = self._random_level()

        pos = len(self._ids)
        self._ids.append(id)
        self._vectors.append(vec)
        self._levels.append(level)
        self._links.append([list() for _ in range(level + 1)])
        self._pos[id] = pos

        if self._entry is None:  # first point simply becomes the entry point
            self._entry = pos
            self._max_level = level
            return

        # Phase 1: greedy descent through layers above the node's level
        # (SEARCH-LAYER with ef=1).  Deleted nodes are fine to stand on here;
        # this is pure navigation.
        ep = self._entry
        for layer in range(self._max_level, level, -1):
            ep = self._search_layer(vec, [ep], 1, layer, include_deleted=True)[0][1]

        # Phase 2: connect on each layer the node participates in.
        eps = [ep]
        for layer in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(vec, eps, self.ef_construction, layer)
            neighbors = self._select_neighbors_heuristic(found, self.M)
            m_max = self.M0 if layer == 0 else self.M
            for nb in neighbors:
                self._links[pos][layer].append(nb)
                self._links[nb][layer].append(pos)
                lst = self._links[nb][layer]
                if len(lst) > m_max:
                    self._links[nb][layer] = self._prune_links(nb, layer, m_max)
            eps = [p for _, p in found]  # paper: ep <- W

        if level > self._max_level:
            self._max_level = level
            self._entry = pos

    def add_batch(self, vectors: Mapping[Any, Sequence[float]]) -> None:
        """Insert many points, in the mapping's own iteration order.

        Plain dicts preserve insertion order, so ``add_batch`` over a dict is
        deterministic and equivalent to calling :meth:`add` per item.
        """
        for pid, vec in vectors.items():
            self.add(pid, vec)

    def search(
        self,
        query: Sequence[float],
        k: int = 10,
        ef_search: int = 64,
    ) -> List[Tuple[Any, float]]:
        """Approximate k-nearest-neighbor search (the paper's Algorithm 5).

        Greedy descent from the entry point down to layer 1 (ef=1 per layer),
        then a beam search of width ``ef_search`` on layer 0.  Deleted points
        are traversed for connectivity but never returned.

        Returns up to ``k`` ``(id, distance)`` pairs sorted by ascending
        distance.  Distance semantics follow ``self.metric``:

        * ``"euclidean"`` -- true L2; the internally tracked L2 squared gets
          its square root applied here, once, at the end;
        * ``"cosine"`` -- ``1 - cosine_similarity``, in ``[0, 2]``;
        * ``"inner_product"`` -- negated dot product (smaller = more similar).

        ``ef_search`` is widened to at least ``k``.  Results are approximate:
        increasing ``ef_search`` increases recall at linear latency cost.
        """
        if self._entry is None:
            return list()
        k = max(1, int(k))
        ef = max(int(ef_search), k)
        vec = self._prepare(query)

        ep = self._entry
        for layer in range(self._max_level, 0, -1):
            ep = self._search_layer(vec, [ep], 1, layer, include_deleted=True)[0][1]

        found = self._search_layer(vec, [ep], ef, 0)
        results = [(self._ids[p], d) for d, p in found[:k]]
        if self.metric == "euclidean":
            results = [(pid, math.sqrt(d)) for pid, d in results]
        return results

    def remove(self, id: Any) -> None:
        """Tombstone a point in O(1) — the honest, cheap kind of delete.

        The node's storage and links are **not** reclaimed and the graph is
        not repaired: searches still route *through* the tombstone to keep
        the graph connected, so recall is unaffected, but per-query latency
        creeps up as the deleted fraction grows and the memory is only freed
        by rebuilding the index (re-insert every live point).  A true delete
        would require neighborhood re-wiring or a full rebuild; this
        implementation chooses the tombstone and says so.

        Raises ``KeyError`` for unknown ids; removing an already-removed id
        is a no-op.
        """
        pos = self._pos[id]  # KeyError with the offending id for unknown ids
        self._deleted.add(pos)

    def save(self, path: str) -> None:
        """Serialize the full index to a JSON file.

        Stores parameters, vectors, per-layer adjacency (as external ids),
        the entry point, and tombstones.  Both the in-memory layout and the
        JSON writing are insertion-ordered, so saving the same index state
        twice produces byte-identical files, and :meth:`load` restores an
        index with bit-identical search behavior.

        JSON is compact enough for a portfolio project; a production build
        would use a binary format to cut parse time and file size.
        """
        id_of_pos = self._ids
        data: Dict[str, Any] = {
            "format": _SAVE_FORMAT,
            "version": _SAVE_VERSION,
            "params": {
                "dim": self.dim,
                "M": self.M,
                "ef_construction": self.ef_construction,
                "mL": self.mL,
                "metric": self.metric,
                "seed": self.seed,
            },
            "entry_point": id_of_pos[self._entry] if self._entry is not None else None,
            "max_level": self._max_level,
            "ids": list(id_of_pos),
            "vectors": [list(vec) for vec in self._vectors],
            "levels": list(self._levels),
            "links": [
                [[id_of_pos[nb] for nb in layer] for layer in node_links]
                for node_links in self._links
            ],
            "deleted": [id_of_pos[p] for p in sorted(self._deleted)],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str) -> "HNSW":
        """Restore an index written by :meth:`save` (exact roundtrip).

        Raises ``ValueError`` if the file is not a serialized HNSW index or
        has an unsupported format version.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or data.get("format") != _SAVE_FORMAT:
            raise ValueError(f"not a {_SAVE_FORMAT} file: {path!r}")
        if data.get("version") != _SAVE_VERSION:
            raise ValueError(f"unsupported format version: {data.get('version')!r}")

        params = data["params"]
        index = cls(
            dim=params["dim"],
            M=params["M"],
            ef_construction=params["ef_construction"],
            mL=params["mL"],
            metric=params["metric"],
            seed=params["seed"],
        )
        ids = data["ids"]
        vectors = data["vectors"]
        levels = data["levels"]
        if not (len(ids) == len(vectors) == len(levels)):
            raise ValueError("corrupt file: ids/vectors/levels length mismatch")

        for pid, vec, level in zip(ids, vectors, levels):
            pos = len(index._ids)
            index._ids.append(pid)
            index._vectors.append(tuple(float(x) for x in vec))
            index._levels.append(int(level))
            index._pos[pid] = pos
        pos_of_id = index._pos
        index._links = [
            [[pos_of_id[nb] for nb in layer] for layer in node_links]
            for node_links in data["links"]
        ]
        index._deleted = {pos_of_id[pid] for pid in data.get("deleted", [])}
        entry = data["entry_point"]
        index._entry = pos_of_id[entry] if entry is not None else None
        index._max_level = int(data["max_level"])
        return index

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _random_level(self) -> int:
        """Draw a top level from a geometric distribution: ``floor(-ln(U) * mL)``.

        ``1 - random()`` maps the half-open ``[0, 1)`` onto ``(0, 1]`` so the
        logarithm never sees zero.
        """
        u = 1.0 - self._rng.random()
        return int(-math.log(u) * self.mL)

    def _prepare(self, vector: Sequence[float]) -> Tuple[float, ...]:
        """Validate a vector and put it into comparison form.

        Converts to a tuple of floats (rejecting non-finite values), enforces
        the dimension, and L2-normalizes exactly once when the metric is
        cosine, so the hot path never renormalizes per comparison.
        """
        if len(vector) != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got length {len(vector)}")
        vec = tuple(float(x) for x in vector)
        for x in vec:
            if not math.isfinite(x):
                raise ValueError(f"vector contains non-finite value: {x!r}")
        if self._normalize_input:
            vec = normalize(vec)
        return vec

    def _search_layer(
        self,
        query: Tuple[float, ...],
        entry_points: Sequence[int],
        ef: int,
        layer: int,
        include_deleted: bool = False,
    ) -> List[_Candidate]:
        """Beam search on a single layer (the paper's Algorithm 2, SEARCH-LAYER).

        Keeps a max-heap ``worst`` of at most ``ef`` best-so-far results and a
        min-heap ``candidates`` frontier.  Stops when the closest frontier
        node is farther than the worst kept result and the result set is full:
        every node closer than that has by then been expanded.

        Tombstoned nodes are always *traversed* (their links are followed) so
        deletions never disconnect the graph; ``include_deleted`` only decides
        whether they may occupy result slots.  The index sets it to True for
        navigation-only searches (greedy descent), where standing on a
        deleted node is fine, and leaves it False for result-producing
        searches so users never see removed ids.

        Returns ``[(distance, pos), ...]`` sorted ascending by distance.
        """
        dist = self._dist
        vectors = self._vectors
        links = self._links
        deleted = self._deleted

        visited = set(entry_points)
        candidates: List[_Candidate] = list()  # min-heap of (distance, pos)
        worst: List[_Candidate] = list()       # max-heap via (-distance, pos)
        for p in entry_points:
            d = dist(query, vectors[p])
            heapq.heappush(candidates, (d, p))
            if include_deleted or p not in deleted:
                heapq.heappush(worst, (-d, p))
        while len(worst) > ef:  # entry set may exceed ef; keep the best ef
            heapq.heappop(worst)

        while candidates:
            d_c, c = heapq.heappop(candidates)
            if len(worst) >= ef and d_c > -worst[0][0]:
                break
            for nb in links[c][layer]:
                if nb in visited:
                    continue
                visited.add(nb)
                d = dist(query, vectors[nb])
                if len(worst) < ef or d < -worst[0][0]:
                    heapq.heappush(candidates, (d, nb))
                    if include_deleted or nb not in deleted:
                        heapq.heappush(worst, (-d, nb))
                        if len(worst) > ef:
                            heapq.heappop(worst)

        results = [(-neg_d, p) for neg_d, p in worst]
        results.sort()
        return results

    def _select_neighbors_heuristic(
        self,
        candidates: Sequence[_Candidate],
        m: int,
        keep_pruned: bool = True,
    ) -> List[int]:
        """Diversity-aware neighbor selection (the paper's Algorithm 4).

        Walks candidates from closest to farthest and keeps a candidate only
        if it is closer to the query than to *every* already-selected
        neighbor — i.e. it is not already "covered" by a selected node lying
        in the same direction.  This spreads links around the query instead
        of clustering them all on one near hub, which is what makes the
        graph navigable.

        ``keep_pruned=True`` backfills leftover slots with the closest
        discarded candidates (the paper's ``keepPrunedConnections``), keeping
        node degrees healthy in sparse regions.

        ``candidates`` are ``(distance_to_query, pos)`` pairs; the list is
        re-sorted here so selection is deterministic regardless of caller.
        """
        ordered = sorted(candidates)
        selected: List[_Candidate] = list()
        discarded: List[_Candidate] = list()
        vectors = self._vectors
        dist = self._dist
        for d_eq, e in ordered:
            if len(selected) >= m:
                break
            covered = False
            for _, s in selected:
                if dist(vectors[e], vectors[s]) < d_eq:
                    covered = True
                    break
            if covered:
                discarded.append((d_eq, e))
            else:
                selected.append((d_eq, e))
        if keep_pruned:
            for d_eq, e in discarded:
                if len(selected) >= m:
                    break
                selected.append((d_eq, e))
        return [p for _, p in selected]

    def _prune_links(self, node: int, layer: int, m_max: int) -> List[int]:
        """Re-select a node's links with the heuristic after an overflow.

        Candidates are the node's current neighbors (the freshly added one
        included), ranked by distance *to the node*; the diversity heuristic
        then keeps the best ``m_max``.  This matches hnswlib's shrink step.
        """
        vec = self._vectors[node]
        dist = self._dist
        cands = [(dist(vec, self._vectors[nb]), nb) for nb in self._links[node][layer]]
        return self._select_neighbors_heuristic(cands, m_max)
