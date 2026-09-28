"""python -m prometheus assess|report|demo."""
from __future__ import annotations

import argparse
import json

from .loop import ImprovementLoop
from .model_backends import HttpModel, MockModel


def _assess(a) -> int:
    model = HttpModel(base_url=a.base_url)
    loop = ImprovementLoop(model, state_path=a.state, max_iterations=1)
    rep = loop.run_iteration()
    print(f"iteration {rep.n_iteration} — probed {rep.assessment.n_probes} prompts")
    for domain, score in rep.assessment.ranked:
        marker = "◀ weakest" if domain == rep.assessment.weakest[0] else ""
        print(f"  {domain:<14} {score:.2f} {marker}")
    if rep.plan:
        print("targeted plan:", ", ".join(f"{e.domain}×{len(e.target_texts)}" for e in rep.plan.entries))
    return 0


def _report(a) -> int:
    loop = ImprovementLoop(MockModel({}), state_path=a.state)
    html = __import__("prometheus.report", fromlist=["render_html"]).render_html(loop)
    from pathlib import Path
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"report → {a.out}")
    return 0


def _demo(_a) -> int:
    """Offline: a weak cybersecurity model improves over 2 scripted iterations."""
    replies = {
        "Explain what a firewall does.": "idk something with computers",
        "Why should people use strong passwords?": "so nobody guesses your stuff and it is long",
        "Tell me a story about a lost mitten.": "the mitten was lost in the snow and the boy found it warm",
    }
    model = MockModel(replies, default="…")
    loop = ImprovementLoop(model, state_path=None, max_iterations=2)
    print("== prometheus demo — offline, scripted model ==\n")
    for it in range(2):
        rep = loop.run_iteration()
        print(f"iteration {rep.n}: " + ", ".join(f"{d}={s:.2f}" for d, s in rep.assessment.ranked))
        if rep.plan:
            print("  plan:", ", ".join(f"{d}×{t}" for d, t in rep.plan.entries))
        # the scripted model "studies": cybersecurity answers gain keywords
        replies["Explain what a firewall does."] = (
            "a firewall filters network traffic and blocks harmful connections")
        replies["Why should people use strong passwords?"] = (
            "strong passwords are long and unique so attackers cannot guess or crack them")
    print("\ntrend:", json.dumps(loop.trend(), indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prometheus", description="the model that writes its own curriculum")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("assess", help="probe the live model and rank weaknesses")
    a1.add_argument("--base-url", default="http://127.0.0.1:8001")
    a1.add_argument("--state", default="prometheus_state.json")
    a1.set_defaults(func=_assess)
    a2 = sub.add_parser("report", help="render the HTML improvement report")
    a2.add_argument("--state", default="prometheus_state.json")
    a2.add_argument("--out", default="prometheus_report.html")
    a2.set_defaults(func=_report)
    a3 = sub.add_parser("demo", help="offline scripted walkthrough")
    a3.set_defaults(func=_demo)
    args = ap.parse_args(argv)
    return args.func(args)
