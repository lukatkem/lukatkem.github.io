"""Iteration report → self-contained dark HTML with an SVG domain bar chart."""
from __future__ import annotations

from .loop import ImprovementLoop

_C = {"bg": "#0c0c0c", "fg": "#d7e2ea", "dim": "#646973", "bar": "#5aa9ff", "line": "#2c4a6e"}


def render_html(loop: ImprovementLoop) -> str:
    trend = loop.trend()
    domains = sorted(trend.keys())
    max_iters = max((len(v) for v in trend.values()), default=1)
    max_score = 1.0

    bars = []
    for d in domains:
        for i, score in enumerate(trend[d]):
            w = max(2, int(score * 100))
            x = 90 + i * 150
            y = 40 + domains.index(d) * 46
            bars.append(f'<rect x="{x}" y="{y}" width="{w}" height="22" fill="{_C["bar"]}"/>')
            bars.append(f'<text x="{x + w + 8}" y="{y + 16}" fill="{_C["fg"]}" '
                        f'font-size="13">{score:.2f}</text>')
            bars.append(f'<text x="14" y="{y + 16}" fill="{_C["dim"]}" font-size="13">{d}</text>')
    iter_labels = "".join(
        f'<text x="{90 + i * 150}" y="{30 + 0}" fill="{_C["dim"]}" font-size="12">iter {i + 1}</text>'
        for i in range(max_iters))

    iters_html = ""
    for it in reversed(loop.history):
        weakest = min(it["domain_scores"].items(), key=lambda kv: kv[1]) if it["domain_scores"] else ("—", 0)
        plan = ", ".join(f"{d}×{t}" for d, t in it.get("plan", [])) or "—"
        trained = "yes" if it.get("trained") else "pending"
        iters_html += (f'<tr><td>{it["n"]}</td><td>{weakest[0]} ({weakest[1]:.2f})</td>'
                       f'<td>{plan}</td><td>{trained}</td></tr>')

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>prometheus — self-improvement report</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{background:{_C['bg']};color:{_C['fg']};
font-family:-apple-system,'Segoe UI',Roboto,Arial,sans-serif;padding:40px}}
h1{{font-weight:700;letter-spacing:.02em;margin-bottom:4px}}p.dim{{color:{_C['dim']};margin-bottom:26px}}
svg{{background:#101418;border:1px solid #22303e;border-radius:12px}}
table{{border-collapse:collapse;margin-top:26px;width:100%;max-width:720px}}
td,th{{border-bottom:1px solid #22303e;padding:9px 12px;font-size:14px;text-align:left}}
th{{color:{_C['dim']};font-size:11px;letter-spacing:.2em;text-transform:uppercase}}</style></head>
<body><h1>prometheus</h1>
<p class="dim">the model that writes its own curriculum — measured {len(loop.history)} iteration(s) ·
target {loop.target:.0%} · state {loop.state_path.name}</p>
<svg width="{90 + max_iters * 150}" height="{60 + len(domains) * 46}" role="img"
 aria-label="domain scores per iteration">{iter_labels}{''.join(bars)}</svg>
<table><tr><th>iteration</th><th>weakest domain</th><th>targeted plan</th><th>retrained</th></tr>
{iters_html}</table></body></html>"""
