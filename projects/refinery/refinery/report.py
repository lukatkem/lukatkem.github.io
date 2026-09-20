"""Markdown report — the funnel, the reasons, the receipts."""
from __future__ import annotations

from collections import Counter

from .pipeline import RefineryResult


def report(result: RefineryResult, sources: list[str]) -> str:
    lines: list[str] = []
    n0 = len(result.verdicts)
    kept = len(result.kept)
    lines.append("# Refinery report")
    lines.append("")
    lines.append(f"**Inputs:** {n0} documents from {len(sources)} source(s)  ")
    lines.append(f"**Kept:** {kept} ({kept / max(n0, 1):.0%}) · cleaned in {result.elapsed_s:.2f}s")
    lines.append("")

    lines.append("## Funnel")
    lines.append("")
    lines.append("| stage | surviving |")
    lines.append("|---|---|")
    prev = n0
    for stage, n in result.funnel.items():
        pct = f"{n / max(prev, 1):.0%} of previous" if prev else ""
        lines.append(f"| {stage} | {n} |{'' if not pct else f' ({pct})'}|")
        prev = n
    lines.append("")

    drops = result.dropped
    by_stage = Counter(v.stage for v in drops)
    lines.append("## Why documents were dropped")
    lines.append("")
    lines.append("| stage | dropped |")
    lines.append("|---|---|")
    for stage, n in by_stage.most_common():
        lines.append(f"| {stage} | {n} |")
    lines.append("")

    pii_total: Counter = Counter()
    for v in result.verdicts:
        for kind, n in v.pii.items():
            pii_total[kind] += n
    lines.append("## Scrubbed personal data / secrets")
    lines.append("")
    if pii_total:
        for kind, n in pii_total.most_common():
            lines.append(f"- **{kind}**: {n} occurrence(s) replaced with `[REDACTED]`")
    else:
        lines.append("- none found")
    lines.append("")

    lines.append("## Example drops (up to 8)")
    lines.append("")
    for v in drops[:8]:
        snippet = v.doc.text[:90].replace("\n", " ")
        lines.append(f"- `{v.stage}` · doc {v.doc.id}: “{snippet}…” — {'; '.join(v.reasons)}")
    lines.append("")
    return "\n".join(lines)
