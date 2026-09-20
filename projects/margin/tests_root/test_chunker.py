"""Chunker tests: heading breadcrumbs, packing, overlap, edge cases."""
from margin.chunker import chunk_markdown, _split_long, breadcrumb, _split_sections

MD = """# Payouts

Intro text here.

## Schedule

Bi-weekly windows open on the 14th and 28th.

### Splits

80% start, 90% later. This section has a longer body so that packing logic
kicks in when combined with more sentences. Here is one. And another.
"""


def test_sections_and_breadcrumb():
    title, sections = _split_sections(MD)
    assert title == "Payouts"
    bcs = [bc for bc, _ in sections]
    assert "" in bcs                      # intro before first heading
    assert "Schedule" in bcs
    assert "Schedule > Splits" in bcs


def test_chunks_carry_breadcrumb():
    chunks = chunk_markdown("rules/11-payouts.md", MD)
    splits = [c for c in chunks if "Schedule" in c.heading and "Splits" not in c.heading]
    assert splits and all(c.title == "Payouts" for c in chunks)
    assert all(c.doc == "rules/11-payouts.md" for c in chunks)


def test_no_h1_doc_still_chunks():
    chunks = chunk_markdown("x.md", "## Only H2\n\nBody text.")
    assert chunks and chunks[0].title == "x.md"


def test_long_text_packs_to_target():
    text = ("The daily loss limit resets at midnight UTC. " * 60)
    pieces = _split_long(text, target=300)
    assert all(len(p) <= 400 for p in pieces)
    assert len(pieces) >= 3


def test_empty_doc_is_safe():
    assert chunk_markdown("e.md", "") == []
