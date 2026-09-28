"""Probing + scoring: ask the model every probe, score keyword coverage.

keyword_hit_rate = hits / len(keywords) per probe, matched leniently as a
case-insensitive substring (a 25M-char-level GPT paraphrases loosely — the
curriculum measures topic coverage, not prose). Per-domain score is the mean
hit rate over that domain's 12 probes. ranked is ascending: weakest first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .curriculum import DOMAINS, Probe
from .model_backends import Base, ModelUnavailable   # noqa: F401 (re-exported)


def keyword_hit_rate(reply: str, keywords: tuple[str, ...]) -> float:
    """hits / len(keywords) — case-insensitive, lenient substring matching."""
    if not keywords:
        return 0.0
    hay = reply.lower()
    hits = sum(1 for kw in keywords if kw.lower() in hay)
    return hits / len(keywords)


@dataclass
class Assessment:
    """One measurement of the student: scores per domain, weakest first."""

    domain_scores: dict[str, float]          # domain -> mean keyword hit rate
    ranked: list[tuple[str, float]]          # ascending — weakest domain first
    replies: dict[str, str] = field(default_factory=dict)   # prompt -> reply
    n_probes: int = 0

    @property
    def weakest(self) -> tuple[str, float]:
        return self.ranked[0] if self.ranked else ("none", 0.0)

    def weakest_domains(self, n: int = 2) -> list[str]:
        return [domain for domain, _ in self.ranked[:n]]


class ProbeRunner:
    """Runs the curriculum against a model and scores each domain."""

    def __init__(self, model: Base):
        self.model = model

    def probe_reply(self, probe: Probe) -> str:
        return self.model.complete(probe.prompt)

    def score_reply(self, probe: Probe, reply: str) -> float:
        rate = keyword_hit_rate(reply, probe.keywords)
        return round(rate, 4)

    def run(self, domains: list[str] | None = None) -> Assessment:
        """Probe every prompt (or just the named domains) → Assessment."""
        names = list(domains) if domains else list(DOMAINS)
        scores: dict[str, float] = {}
        replies: dict[str, str] = {}
        n = 0
        for domain in names:
            rates = []
            for probe in DOMAINS[domain]:
                try:
                    reply = self.probe_reply(probe)
                except ModelUnavailable:
                    raise
                except Exception:                     # noqa: BLE001 — a failed probe scores 0
                    reply = ""
                replies[probe.prompt] = reply
                rates.append(self.score_reply(probe, reply))
                n += 1
            scores[domain] = round(sum(rates) / len(rates), 4) if rates else 0.0
        ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
        return Assessment(domain_scores=scores, ranked=ranked, replies=replies, n_probes=n)
