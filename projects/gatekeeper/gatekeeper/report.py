def render_html(log: list[dict], caption: str = "") -> str:
    total = len(log) or 1
    ok = sum(1 for e in log if e.get("status") == "ok")
    errors = total - ok
    ok_frac = ok / total; err_frac = errors / total
    donut = (f'<circle cx="70" cy="70" r="52" fill="none" stroke="#2ea44f" stroke-width="16" '
             f'stroke-dasharray="{ok_frac*100:.1f} {100-ok_frac*100:.1f}" pathLength="100"/>'
             f'<circle cx="70" cy="70" r="52" fill="none" stroke="#e5534b" stroke-width="16" '
             f'stroke-dasharray="{err_frac*100:.1f} {100-err_frac*100:.1f}" '
             f'stroke-dashoffset="-{ok_frac*100:.1f}" pathLength="100"/>')
    rows = "".join(
        f'<tr><td>{i+1}</td><td>{e.get("provider","?")}</td>'
        f'<td style="color:{"#2ea44f" if e.get("status")=="ok" else "#e5534b"}">'
        f'{e.get("status") or e.get("attestation_error") or e.get("error","?")}</td></tr>'
        for i, e in enumerate(log))
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
            f'*{{margin:0;box-sizing:border-box}}body{{background:#0c0c0c;color:#d7e2ea;padding:40px;'
            f'font-family:system-ui,sans-serif}}table{{border-collapse:collapse;width:100%;margin-top:20px}}'
            f'td,th{{border-bottom:1px solid #22303e;padding:8px 12px;font-size:13px;text-align:left}}'
            f'h1{{font-weight:700}}svg{{margin:20px 0}}</style></head><body>'
            f'<h1>Gatekeeper — security report</h1><p style="color:#646973">{caption}</p>'
            f'<svg width="140" height="140">{donut}'
            f'<text x="70" y="66" text-anchor="middle" fill="#d7e2ea" font-size="15" font-weight="700">{ok}</text>'
            f'<text x="70" y="84" text-anchor="middle" fill="#646973" font-size="10">attested</text></svg>'
            f'<table><tr><th>#</th><th>Provider</th><th>Result</th></tr>{rows}</table></body></html>')
