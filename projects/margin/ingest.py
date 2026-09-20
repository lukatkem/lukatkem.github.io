"""Corpus ingestion: docs → chunks → vectors (Ollama if present, TF-IDF else)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from .chunker import chunk_markdown
from .config import CHUNK_OVERLAP, CHUNK_TARGET, CORPUS_DIR
from .store import connect, now
from .vectors import TfidfEncoder, _ollama_embed, vec_to_blob

MARKDOWN_EXTS = {".md", ".markdown", ".txt"}


def discover_docs(corpus_dir: Path) -> list[Path]:
    return sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in MARKDOWN_EXTS)


def ingest(corpus_dir: Path | None = None, verbose: bool = True) -> dict:
    corpus_dir = corpus_dir or CORPUS_DIR
    conn = connect()
    docs = discover_docs(corpus_dir)
    if not docs:
        raise SystemExit(f"no markdown docs found under {corpus_dir}")

    # rebuild store (corpus is source of truth; user uploads live in separate docs and survive)
    conn.execute("DELETE FROM vectors")
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM docs")

    all_chunks = []
    for path in docs:
        rel = str(path.relative_to(corpus_dir))
        md = path.read_text(encoding="utf-8")
        h = hashlib.sha256(md.encode()).hexdigest()[:16]
        conn.execute(
            "INSERT INTO docs(path,title,hash,ingested_at) VALUES(?,?,?,?)",
            (rel, md.splitlines()[0].lstrip("# ") if md else rel, h, now()),
        )
        chunks = chunk_markdown(rel, md, CHUNK_TARGET, CHUNK_OVERLAP)
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO chunks(doc,heading,ord,text,n_tokens) VALUES(?,?,?,?,?)",
                (c.doc, c.heading, c.ord, c.text, len(c.text.split())),
            )
            all_chunks.append((cur.lastrowid, f"{c.label}\n{c.text}"))
    conn.commit()

    # embeddings — try Ollama first, fall back to TF-IDF
    texts = [t for _, t in all_chunks]
    ids = [i for i, _ in all_chunks]
    kind = "tfidf"
    embs = _ollama_embed(texts)
    if embs:
        kind = "ollama"
        mat = np.array(embs, dtype=np.float32)
    else:
        if verbose:
            print("ollama embeddings unavailable → TF-IDF fallback", file=sys.stderr)
        enc = TfidfEncoder()
        enc.fit(texts)
        mat = enc.encode(texts)
        enc_path = corpus_dir.parent / "data" / "tfidf.json"
        enc_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        enc_path.write_text(json.dumps(enc.vocab))
        # persist encoder for query-time use
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(k,v) VALUES('tfidf_vocab',?)",
            (json.dumps(enc.vocab),),
        )
        conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('vector_kind',?)", (kind,))
    for i, vec in zip(ids, mat):
        conn.execute(
            "INSERT INTO vectors(chunk_id,kind,dim,vec) VALUES(?,?,?,?)",
            (i, kind, len(vec), vec_to_blob(vec)),
        )
    conn.commit()
    stats = {
        "docs": len(docs),
        "chunks": len(all_chunks),
        "embedder": kind,
        "dim": int(mat.shape[1]),
    }
    if verbose:
        print(f"ingested: {stats}")
    return stats


if __name__ == "__main__":
    ingest()
