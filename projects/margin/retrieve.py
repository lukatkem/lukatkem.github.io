"""Hybrid retrieval: BM25 ∥ dense cosine → Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .config import BM25_CANDIDATES, RRF_K, VEC_CANDIDATES
from .store import connect
from .vectors import TfidfEncoder, blob_to_vec, cosine_topk


@dataclass
class Hit:
    chunk_id: int
    doc: str
    title: str
    heading: str
    text: str
    bm25_rank: int | None
    vec_rank: int | None
    score: float


class Retriever:
    """In-memory index over the SQLite chunk table. Call .reload() after ingest."""

    def __init__(self) -> None:
        self.conn = connect()
        self.bm25: BM25 | None = None
        self.enc: TfidfEncoder | None = None
        self.kind = "none"
        self.reload()

    def reload(self) -> None:
        rows = self.conn.execute(
            """SELECT c.id, c.doc, d.title, c.heading, c.text
               FROM chunks c JOIN docs d ON d.path = c.doc ORDER BY c.id"""
        ).fetchall()
        self.chunk_rows = rows
        self.ids = [r["id"] for r in rows]
        # BM25 needs sequential 0-based doc ids; map internal index → chunk id
        self.bm25 = BM25()
        self._bm25_to_chunk: list[int] = []
        for seq, r in enumerate(rows):
            self.bm25.add(seq, f"{d(r['title'])} {r['heading']} {r['text']}")
            self._bm25_to_chunk.append(r["id"])
        self.mat = None
        kind_row = self.conn.execute("SELECT kind FROM vectors LIMIT 1").fetchone()
        self.kind = kind_row["kind"] if kind_row else "none"
        if rows:
            vecs, vids = [], []
            for r in self.conn.execute(
                "SELECT chunk_id, vec FROM vectors WHERE kind = ? ORDER BY chunk_id",
                (self.kind,),
            ):
                vecs.append(blob_to_vec(r["vec"]))
                vids.append(r["chunk_id"])
            if vecs:
                self.mat = np_stack(vecs)
                self.vec_ids = vids
                if self.kind == "tfidf":
                    self.enc = TfidfEncoder()
                    self.enc.fit([r["text"] for r in rows])
            else:
                self.mat = None

    def _dense(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.mat is None:
            return []
        if self.kind == "ollama":
            from .vectors import _ollama_embed

            emb = _ollama_embed([query])
            if not emb:
                return []
            q = emb[0][0] if isinstance(emb[0][0], list) else emb[0]
            import numpy as np

            qv = np.asarray(q, dtype=np.float32)
        else:
            if self.enc is None:
                return []
            qv = self.enc.encode([query])[0]
        id_map = {cid: i for i, cid in enumerate(self.vec_ids)}
        sub_idx = [id_map[c] for c in self.ids if c in id_map]
        sub_mat = self.mat[sub_idx]
        return cosine_topk(qv, sub_mat, self.ids, k)

    def search(self, query: str, top_k: int = 6, per_doc_limit: int = 2) -> list[Hit]:
        if not self.ids:
            return []
        lex = self.bm25.search(query, BM25_CANDIDATES)
        lex = [(self._bm25_to_chunk[i], s) for i, s in lex]
        den = self._dense(query, VEC_CANDIDATES)

        rrf: dict[int, float] = {}
        ranks: dict[int, tuple[int | None, int | None]] = {}
        for rank, (cid, _) in enumerate(lex):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranks[cid] = (rank, ranks.get(cid, (None, None))[1])
        for rank, (cid, _) in enumerate(den):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            ranks[cid] = (ranks.get(cid, (None, None))[0], rank)

        fused = sorted(rrf.items(), key=lambda x: -x[1])
        # diversity: at most per_doc_limit chunks per source doc
        picked: list[tuple[int, float]] = []
        doc_count: dict[str, int] = {}
        doc_of = {r["id"]: r["doc"] for r in self.chunk_rows}
        for cid, score in fused:
            doc = doc_of[cid]
            if doc_count.get(doc, 0) >= per_doc_limit:
                continue
            doc_count[doc] = doc_count.get(doc, 0) + 1
            picked.append((cid, score))
            if len(picked) >= top_k:
                break
        by_id = {r["id"]: r for r in self.chunk_rows}
        hits = []
        for cid, score in picked:
            r = by_id[cid]
            bm, ve = ranks.get(cid, (None, None))
            hits.append(
                Hit(
                    chunk_id=cid,
                    doc=r["doc"],
                    title=r["title"],
                    heading=r["heading"],
                    text=r["text"],
                    bm25_rank=bm,
                    vec_rank=ve,
                    score=round(score, 6),
                )
            )
        return hits


def d(title: str | None) -> str:
    return title or ""


def np_stack(vecs):
    import numpy as np

    return np.stack(vecs)
