"""python -m ctxpack demo · python -m ctxpack pack --docs file.json --chars N"""
from __future__ import annotations

import argparse
import json
import sys

from .budget import Budget
from .doc import Doc
from .packer import pack_density, pack_exact, pack_greedy
from .render import render, render_report

ALGOS = {"greedy": pack_greedy, "density": pack_density, "exact": pack_exact}


def _demo(_a) -> int:
    docs = [
        Doc("policy", "Returns are accepted within 30 days with the original receipt. "
            "Sale items are final after 14 days. Refunds go to the original payment method.", 7.0),
        Doc(" sprawling essay", "A very long document about the history of " + "packing " * 90, 9.0),
        Doc("faq", "Q: Where is my order? A: Track it from the confirmation email.", 5.0),
        Doc("dense-note", "30-day window; receipt required.", 4.5),
        Doc("contact", "Support: open 9-17, Tbilisi time.", 2.0),
    ]
    budget = Budget(chars=320, header_chars=24, per_doc_overhead=30)
    print(f"{'algorithm':<10} {'docs':<14} {'score':>6} {'chars':>8}")
    for name, algo in ALGOS.items():
        pk = algo(docs, budget)
        ids = ",".join(d.id for d in pk.included) or "-"
        print(f"{name:<10} {ids:<14} {pk.total_score:>6.1f} {pk.used_chars:>8}")
    best = max(ALGOS.values(), key=lambda f: f(docs, budget).total_score)(docs, budget)
    print("\n--- best packing (render) ---")
    print(render(best, header="Answer using these documents."))
    print("\n--- audit ---")
    print(render_report(best, budget.chars))
    return 0


def _pack(a) -> int:
    raw = json.loads(open(a.docs, encoding="utf-8").read())
    docs = [Doc(d["id"], d["text"], float(d["score"])) for d in raw]
    budget = Budget(chars=a.chars, per_doc_overhead=a.overhead)
    pk = ALGOS[a.algorithm](docs, budget)
    print(render(pk))
    print("\n--- audit ---")
    print(render_report(pk, budget.chars))
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ctxpack", description="pack maximum doc value into a context window")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.set_defaults(func=_demo)
    p = sub.add_parser("pack")
    p.add_argument("--docs", required=True, help="JSON file: [{id,text,score},…]")
    p.add_argument("--chars", type=int, required=True)
    p.add_argument("--overhead", dest="overhead", type=int, default=30)
    p.add_argument("--algorithm", choices=list(ALGOS), default="exact")
    p.set_defaults(func=_pack)
    args = ap.parse_args(argv)
    return args.func(args)
