"""The pipeline — stages, funnel, per-document verdicts."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import dedup, language, normalize, pii, quality


@dataclass
class Doc:
    id: str
    source: str
    text: str


@dataclass
class Verdict:
    doc: Doc
    kept: bool
    stage: str  # stage that decided the final fate
    score: float
    reasons: list[str] = field(default_factory=list)
    pii: dict[str, int] = field(default_factory=dict)


@dataclass
class RefineryResult:
    verdicts: list[Verdict]
    funnel: dict[str, int]  # docs surviving after each stage
    elapsed_s: float

    @property
    def kept(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.kept]

    @property
    def dropped(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.kept]


class Refinery:
    """Normalize → scrub PII → language → quality → exact dedup → near dedup.

    Order matters: cheap stateless checks run first so the expensive
    fuzzy pass only ever sees documents worth comparing."""

    def __init__(
        self,
        lang: str = "en",
        min_language_score: float = 0.45,
        min_quality: float = 0.6,
        dedup_threshold: float = 0.8,
        scrub_pii: bool = True,
    ):
        self.lang = lang
        self.min_language_score = min_language_score
        self.min_quality = min_quality
        self.dedup_threshold = dedup_threshold
        self.scrub_pii = scrub_pii

    def run(self, docs: list[Doc]) -> RefineryResult:
        t0 = time.perf_counter()
        verdicts: list[Verdict] = []
        funnel: dict[str, int] = {}

        survivors: list[Verdict] = []
        for d in docs:
            text = normalize.normalize(d.text)

            pii_counts: dict[str, int] = {}
            reasons: list[str] = []
            if self.scrub_pii:
                text, pii_counts = pii.scrub(text)

            if not text:
                verdicts.append(Verdict(d, False, "normalize", 0.0, ["empty-after-normalize"], pii_counts))
                continue
            if not language.is_language(text, self.lang, self.min_language_score):
                verdicts.append(Verdict(d, False, "language", language.language_score(text),
                                        [f"not-{self.lang} (score {language.language_score(text):.2f})"], pii_counts))
                continue
            q = quality.quality(text)
            if not q.passed:
                verdicts.append(Verdict(d, False, "quality", q.score, q.reasons, pii_counts))
                continue
            survivors.append(Verdict(Doc(d.id, d.source, text), True, "dedup", q.score, [], pii_counts))
            funnel["language+quality"] = funnel.get("language+quality", 0) + 1

        funnel["pii+language+quality"] = len(survivors)

        # exact duplicates
        texts = [v.doc.text for v in survivors]
        exact_groups = dedup.exact_duplicate_groups(texts)
        exact_dropped: set[int] = {i for g in exact_groups for i in g[1:]}
        stage1 = [v for i, v in enumerate(survivors) if i not in exact_dropped]
        funnel["after-exact-dedup"] = len(stage1)

        # near duplicates (only survivors — cheap fuzzy pass on the shortlist)
        remaining = [v for i, v in enumerate(survivors) if i not in exact_dropped]
        rem_texts = [v.doc.text for v in remaining]
        near_groups = dedup.clusters(rem_texts, threshold=self.dedup_threshold)
        near_dropped: set[int] = {i for g in near_groups for i in g[1:]}
        final = [v for i, v in enumerate(remaining) if i not in near_dropped]
        funnel["after-near-dedup"] = len(final)

        for i, v in enumerate(remaining):
            if i in near_dropped:
                v.kept, v.stage = False, "near-dedup"
                v.reasons = ["near-duplicate"]
            verdicts.append(v)
        for i, v in enumerate(survivors):
            if i in exact_dropped:
                verdicts.append(Verdict(v.doc, False, "exact-dedup", v.score, ["exact-duplicate"], v.pii))

        return RefineryResult(verdicts, funnel, time.perf_counter() - t0)
