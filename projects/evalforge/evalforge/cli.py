"""python -m evalforge demo · generate --docs file.json --name myset --out file.json"""
from __future__ import annotations

import argparse
import json
import json

from .generate import generate_pairs
from .quality import QualityFilter
from .set import EvalSet


def _demo(_a) -> int:
    docs = {
        "faq": ("The return policy is 30 days with a receipt. The store has 3 branches in Tbilisi. "
                "Sale items are final after 14 days."),
        "security": ("Two-factor authentication is an extra layer of security beyond the password. "
                     "A firewall filters network traffic. Phishing tricks users into revealing credentials."),
        "python": ("A decorator is a function that wraps another function. "
                   "The list has 5 methods for mutation."),
    }
    es = EvalSet(name="demo-set")
    for doc_id, text in docs.items():
        pairs = generate_pairs(doc_id, text)
        es.add(pairs)
        print(f"{doc_id}: {len(pairs)} pairs generated")
    print("\n--- sample cases ---")
    for c in es.to_json()["cases"][:6]:
        print(f"  [{c['kind']}] {c['question']}\n      → {c['answer']}")
    print(f"\ntotal: {len(es.pairs)} unique pairs (dedup + quality-filtered)")
    print("margin export shape:", es.to_margin_cases()[0])
    return 0


def _generate(a) -> int:
    raw = json.loads(open(a.docs, encoding="utf-8").read())
    es = EvalSet(name=a.name)
    flt = QualityFilter(min_question_len=a.min_q)
    for d in raw:
        es.add(generate_pairs(d["id"], d["text"], flt))
    es.save(a.out)
    kinds = {}
    for p in es.pairs:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    print(f"{a.out}: {len(es.pairs)} pairs · by kind: {kinds}")
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalforge", description="documents → golden-set eval pairs, offline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.set_defaults(func=_demo)
    g = sub.add_parser("generate")
    g.add_argument("--docs", required=True, help='JSON: [{"id","text"},…]')
    g.add_argument("--name", default="evalforge-set")
    g.add_argument("--out", default="evalset.json")
    g.add_argument("--min-q", type=int, default=15)
    g.set_defaults(func=_generate)
    args = ap.parse_args(argv)
    return args.func(args)
