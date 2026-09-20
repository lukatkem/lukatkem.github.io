"""Refinery — raw text in, training-grade corpus out.

Data curation is where modern AI is actually won: models are only as good
as the corpus they train on, and real corpora arrive full of duplicates,
junk, leaked secrets and personal data. Refinery is a from-scratch data
refinery — every algorithm (MinHash, LSH, union-find, heuristic scoring)
is implemented in this package with zero dependencies.

Pipeline, in order:
    normalize  — unicode NFKC, whitespace, boilerplate
    pii scrub  — emails, phones, ids, API keys, JWTs  (secrets in corpora are real)
    language   — stopword + letter-ratio filter for the target language
    quality    — Gopher/C4-style heuristics with a per-doc reason list
    exact dedup— md5 of normalized text
    near dedup — MinHash signatures + banded LSH + union-find clusters

    python -m refinery.cli --input DIR --out clean.txt --report report.md
"""
from .pipeline import Doc, Refinery, RefineryResult

__all__ = ["Doc", "Refinery", "RefineryResult"]
__version__ = "1.0.0"
