"""Refinery tests — every stage gets dirty synthetic input, expected verdicts out."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refinery import dedup, language, normalize, pii, quality  # noqa: E402
from refinery.pipeline import Doc, Refinery  # noqa: E402

CLEAN = (
    "Once upon a time there was a little girl named Lily. She loved to play "
    "in the garden behind her house. Every morning she watered the flowers "
    "and said hello to the bees. One day she found a tiny bird with a hurt "
    "wing. She took it inside and made it a small bed. In a week the bird "
    "flew away happy and strong. Lily waved goodbye and smiled."
)


def test_normalize_is_idempotent_and_tidy():
    dirty = "  Hello   world\u00a0!\r\n\r\n\r\n\r\nSecond   line\t\there  "
    once = normalize.normalize(dirty)
    assert normalize.normalize(once) == once, "not idempotent"
    assert "\r" not in once and "  " not in once and "\n\n\n" not in once


def test_pii_scrubs_emails_phones_secrets_but_keeps_dates():
    text = ("Contact sam@example.com or +1 (555) 123-4567. "
            "Card 4532015112830366. Key sk-or-v1-abcdef1234567890abcdef12. "
            "Token eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c "
            "Saved on 2026-09-20 and cost 42 dollars.")
    clean, counts = pii.scrub(text)
    assert "@" not in clean and "[REDACTED]" in clean
    assert counts["email"] == 1 and counts["phone"] == 1
    assert counts["digit-id"] == 1, "16-digit card must be redacted"
    assert counts["secret"] == 2, "api key + jwt must be redacted"
    assert "2026-09-20" in clean, "dates must survive"
    assert "42" in clean, "small numbers must survive"


def test_language_filter_separates_english_from_noise():
    assert language.is_language(CLEAN)
    assert not language.is_language("გამარჯობა ბატონო ეს არის ქართული ტექსტი და აქ ბევრი სიტყვაა საერთოდ")
    assert not language.is_language("!!! ??? ### $$$ %%% ^^^ &&&")


def test_quality_rejects_junk_with_reasons():
    ok = quality.quality(CLEAN)
    assert ok.passed and ok.score >= 0.6, ok.reasons

    repeater = "This exact long line is repeated over and over again here.\n" * 12
    q = quality.quality(repeater)
    assert not q.passed and any("repeated-lines" in r for r in q.reasons)

    symbols = "!!! $$$ ### " * 60
    q2 = quality.quality(symbols)
    assert not q2.passed and any("symbol-soup" in r for r in q2.reasons)


def test_minhash_finds_near_duplicates_ignoring_unique_doc():
    base = ("the quick brown fox jumps over the lazy dog and runs through the "
            "green forest while the sun sets slowly behind the tall quiet hills")
    near = base.replace("quick", "fast")  # one-word edit ≈ 0.68 expected jaccard
    unique = "completely different content about submarines and deep ocean trenches far away"
    texts = [base, near, f"{base} with a tiny tail added at the end", unique, unique]
    groups = dedup.clusters(texts, threshold=0.55)
    # base + its one-edit copy + its tailed copy form one cluster;
    # the two identical "unique" docs form another; nothing crosses over
    assert sorted(groups) == [[0, 1, 2], [3, 4]], groups
    est = dedup.jaccard_estimate(
        dedup.MinHasher().signature(dedup.shingles(base)),
        dedup.MinHasher().signature(dedup.shingles(near)),
    )
    assert est > 0.55, f"one-edit near-dup estimated at only {est:.2f}"


def test_pipeline_funnel_and_clean_output(tmp_path):
    docs = [
        Doc("clean", "a", CLEAN),
        Doc("dup", "a", CLEAN),  # exact duplicate
        Doc("pii", "a", CLEAN.replace("Lily", "Lily <leak@corp.io>", 1)),
        Doc("junk", "a", "!!! $$$ " * 100),
        Doc("foreign", "a", "გამარჯობა ეს არის ქართული ტექსტი ბევრი სიტყვებით საერთოდ და კიდევ"),
    ]
    result = Refinery().run(docs)
    assert result.funnel["after-near-dedup"] == 1, "only one survivor expected"
    kept = result.kept
    assert len(kept) == 1 and kept[0].doc.id == "clean"
    assert "[REDACTED]" not in kept[0].doc.text, "leaked email must be scrubbed from the survivor"
    assert {"dup", "junk", "foreign", "pii"} == {v.doc.id for v in result.dropped}

    from refinery.report import report as md_report

    md = md_report(result, ["test"])
    assert "Funnel" in md and "Why documents were dropped" in md
