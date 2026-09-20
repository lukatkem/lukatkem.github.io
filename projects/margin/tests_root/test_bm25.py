"""BM25 ranking + tokenizer behavior."""
from margin.bm25 import BM25, tokenize


def test_tokenize_stems_plurals():
    assert "payout" in tokenize("Payouts are bi-weekly")
    assert "rule" in tokenize("the RULES")


def test_relevant_doc_ranks_first():
    bm = BM25()
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Payout windows open on the 14th and 28th of each month.",
        "Max daily loss is 5% of starting balance measured on equity.",
    ]
    for i, d in enumerate(docs):
        bm.add(i, d)
    results = bm.search("when are payout windows", top_k=3)
    assert results[0][0] == 1
    results = bm.search("daily loss limit equity", top_k=3)
    assert results[0][0] == 2


def test_no_match_returns_empty():
    bm = BM25()
    bm.add(0, "something entirely different here")
    assert bm.search("zzzqqq", top_k=3) == []
