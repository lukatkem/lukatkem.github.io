"""Markdown-aware chunking.

Splits docs on ATX headings (keeping the heading breadcrumb), then packs
sections into ~CHUNK_TARGET-char chunks with sentence-boundary overlap.
Deterministic and dependency-free so it is unit-testable in CI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

@dataclass
class Chunk:
    doc: str        # relative doc path, e.g. "rules/03-daily-drawdown.md"
    title: str      # first h1 of the doc
    heading: str    # breadcrumb of nearest headings, e.g. "Payouts > Split schedule"
    ord: int        # chunk ordinal within the doc
    text: str

    @property
    def label(self) -> str:
        return f"{self.title} — {self.heading}" if self.heading else self.title

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$])")
_H_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def _split_sections(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(breadcrumb, section_text)])."""
    title = ""
    lines = md.splitlines()
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []  # (level, heading text)

    for line in lines:
        m = _H_RE.match(line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = text
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            sections.append((breadcrumb(stack), []))
        else:
            if not sections:
                sections.append(("", []))
            sections[-1][1].append(line)

    out = []
    for bc, body in sections:
        text = "\n".join(body).strip()
        if text:
            out.append((bc, text))
    return title, out


def breadcrumb(stack: list[tuple[int, str]]) -> str:
    return " > ".join(t for _, t in stack)


def _split_long(text: str, target: int) -> list[str]:
    """Split an oversized section on sentence boundaries, then hard-wrap."""
    if len(text) <= target:
        return [text]
    sentences = _SENT_RE.split(text)
    pieces, cur = [], ""
    for s in sentences:
        if len(s) > target:  # pathological: hard-wrap
            for i in range(0, len(s), target):
                pieces.append(s[i : i + target])
            continue
        if cur and len(cur) + len(s) + 1 > target:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_markdown(doc: str, md: str, target: int = 900, overlap: int = 150) -> list[Chunk]:
    _, sections = _split_sections(md)
    chunks: list[Chunk] = []
    ord_ = 0
    for bc, body in sections:
        pieces = _split_long(body, target)
        for i, piece in enumerate(pieces):
            text = piece
            # carry a sentence of context from the previous piece
            if i > 0 and overlap > 0 and len(pieces) > 1:
                tail = _SENT_RE.split(pieces[i - 1])[-1]
                text = f"{tail[-overlap:]} {piece}".strip() if tail else piece
            chunks.append(Chunk(doc=doc, title="", heading=bc, ord=ord_, text=text))
            ord_ += 1
    # fill titles from doc h1 (empty when the doc has no h1)
    title = next((c for c in chunks), None)
    h1 = ""
    m = re.search(r"^#\s+(.+)$", md, re.M)
    if m:
        h1 = m.group(1).strip()
    for c in chunks:
        c.title = h1 or doc
    return chunks
