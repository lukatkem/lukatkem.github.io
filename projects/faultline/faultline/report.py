"""Chaos report → self-contained dark HTML with an SVG donut + verdict matrix."""
from __future__ import annotations

from .runner import ChaosReport, RecoveryVerdict

_C = {"bg": "#0c0c0c", "fg": "#d7e2ea", "dim": "#646973", "ok": "#2ea44f",
      "silent": "#e5534b", "deg": "#d29922", "line": "#22303e"}


def render_html(report: ChaosReport, caption: str = "") -> str:
    # donut: recovery / silent / degraded
    total = len(report.cases) or 1
    segs = []
    offset = 0.0
    colors = {"recovery": _C["ok"], "silent": _C["silent"], "degraded": _C["deg"]}
    for label, frac in (("recovery", report.recovery_rate),
                        ("silent", report.silent_failure_rate),
                        ("degraded", report.degradation_rate)):
        length = frac * 100
        segs.append(f'<circle cx="70" cy="70" r="52" fill="none" stroke="{colors[label]}" '
                    f'stroke-width="16" stroke-dasharray="{length:.1f} {100 - length:.1f}" '
                    f'stroke-dashoffset="{-offset:.1f}" pathLength="100"/>')
        offset += length
    donut = (f'<svg width="140" height="140" viewBox="0 0 140 140" role="img" '
             f'aria-label="verdict donut">{"".join(segs)}'
             f'<text x="70" y="66" text-anchor="middle" fill="{_C["fg"]}" font-size="15" '
             f'font-weight="700">{report.recovery_rate:.0%}</text>'
             f'<text x="70" y="84" text-anchor="middle" fill="{_C["dim"]}" font-size="10">'
             f'recovery</text></svg>')

    rows = []
    for c in report.cases:
        color = _C["silent"] if c.faulted_verdict is RecoveryVerdict.SILENT_FAILURE else _C["fg"]
        rows.append(
            f'<tr><td>{c.scenario}</td><td>{c.baseline_verdict.value}</td>'
            f'<td style="color:{color}">{c.faulted_verdict.value}</td>'
            f'<td class="faults">{", ".join(c.faults_injected) or "—"}</td>'
            f'<td class="snippet">{c.agent_answer[:90]}</td></tr>')

    silent_rows = "".join(
        f'<tr><td>{c.scenario}</td><td class="faults">{", ".join(c.faults_injected)}</td>'
        f'<td class="snippet">{c.agent_answer[:90]}</td></tr>'
        for c in report.cases if c.faulted_verdict is RecoveryVerdict.SILENT_FAILURE)
    silent_section = (f'<h2>⚠ silent failures — confident answers built on corrupted data</h2>'
                      f'<table>{silent_rows}</table>') if silent_rows else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>FaultLine — agent chaos report</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{background:{_C['bg']};color:{_C['fg']};
font-family:-apple-system,'Segoe UI',Roboto,Arial,sans-serif;padding:40px;line-height:1.5}}
h1{{font-weight:700;margin-bottom:4px}}p.dim{{color:{_C['dim']};margin-bottom:24px}}
.donut{{float:left;margin-right:30px}}.clear{{clear:both}}
h2{{font-size:15px;margin:26px 0 10px;color:{_C['fg']}}}
table{{border-collapse:collapse;width:100%;max-width:980px;margin-bottom:20px}}
td,th{{border-bottom:1px solid {_C['line']};padding:8px 10px;font-size:13px;text-align:left}}
th{{color:{_C['dim']};font-size:11px;letter-spacing:.18em;text-transform:uppercase}}
.faults{{color:{_C['deg']};font-size:12px}}.snippet{{max-width:380px;font-size:12px;color:{_C['dim']}}}
</style></head><body>
<h1>FaultLine</h1><p class="dim">{caption or 'agent chaos report'} · {len(report.cases)} scenarios ·
{report.summary()}</p>
<div class="donut">{donut}
<p style="margin-top:8px;font-size:11px;color:{_C['dim']}">green recovery · red silent ·
yellow degraded</p></div>
<div class="clear"></div>
<h2>verdict matrix — baseline vs faulted</h2>
<table><tr><th>scenario</th><th>baseline</th><th>faulted</th><th>injected faults</th>
<th>agent answer</th></tr>{''.join(rows)}</table>
{silent_section}
</body></html>"""
