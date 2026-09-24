"""Render a Packing into a prompt string + a human-readable audit."""
from __future__ import annotations

from .packer import Packing


def render(packing: Packing, header: str = "Use these documents to answer.") -> str:
    parts = [header, ""]
    for doc in packing.included:
        parts.append(f"[doc {doc.id} | score {doc.score:.2f}]")
        parts.append(doc.text)
        parts.append("")
    return "\n".join(parts).rstrip()


def render_report(packing: Packing, budget_chars: int) -> str:
    lines = [f"packed {len(packing.included)} docs · {packing.used_chars}/{budget_chars} chars "
             f"({packing.utilization:.0%} utilization) · total score {packing.total_score}"]
    if packing.dropped:
        lines.append("dropped:")
        for doc, reason in packing.dropped:
            lines.append(f"  - {doc.id} ({reason})")
    return "\n".join(lines)
