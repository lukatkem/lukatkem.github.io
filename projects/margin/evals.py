"""Deterministic eval harness + CI gate.

Metrics per golden question:
  - recall@5      — gold doc present in top-5 retrieved docs
  - precision@3   — gold doc present in top-3 (context quality)
  - mrr           — 1/rank of the first gold-doc hit
  - answer_hit    — generated answer contains every must_contain string
                    (case-insensitive) — requires LLM unless --retrieval-only

Exit code 1 when thresholds fail → GitHub Actions blocks the PR.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from .config import EVAL_ANSWER_THRESHOLD, EVAL_RECALL_THRESHOLD, EVALS_REPORT_PATH, GOLDEN_PATH
from .generate import answer
from .retrieve import Retriever


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_answer(text: str, must_contain: list) -> bool:
    """Every element must be satisfied. An element is either a string (all
    literal, case-insensitive) or a list of alternatives (any one matches).
    e.g. ["5%", ["three", "3"]] requires "5%" and ("three" or "3")."""
    low = text.lower()
    for item in must_contain:
        alts = item if isinstance(item, list) else [item]
        if not any(a.lower() in low for a in alts):
            return False
    return True


def run(retrieval_only: bool = False, limit: int | None = None, verbose: bool = True) -> dict:
    golden = load_golden(GOLDEN_PATH)
    if limit:
        golden = golden[:limit]
    r = Retriever()

    per_type: dict[str, list] = defaultdict(list)
    results = []
    for q in golden:
        t0 = time.perf_counter()
        hits = r.search(q["question"], top_k=5)
        latency = (time.perf_counter() - t0) * 1000

        docs_ranked = [h.doc for h in hits]
        gold = q["gold_doc"]
        recall5 = 1.0 if gold in docs_ranked else 0.0
        p3 = 1.0 if gold in docs_ranked[:3] else 0.0
        mrr = 0.0
        for rank, doc in enumerate(docs_ranked, 1):
            if doc == gold:
                mrr = 1.0 / rank
                break

        answer_hit = None
        model = "n/a"
        if not retrieval_only:
            text, model, _ = answer(q["question"], hits)
            answer_hit = 1.0 if check_answer(text, q["must_contain"]) else 0.0

        rec = {
            "id": q["id"],
            "question": q["question"],
            "gold_doc": gold,
            "retrieved": docs_ranked[:5],
            "recall5": recall5,
            "precision3": p3,
            "mrr": mrr,
            "latency_ms": round(latency, 1),
            "answer_hit": answer_hit,
            "model": model,
        }
        results.append(rec)
        per_type[q.get("type", "factual")].append(rec)
        if verbose:
            mark = "✓" if recall5 else "✗"
            ans = f" a:{'✓' if answer_hit else '✗'}" if answer_hit is not None else ""
            if not recall5 or answer_hit == 0.0:
                print(f"  {mark}{ans} {q['id']} gold={gold} got={docs_ranked[:3]}")

    n = len(results)
    summary = {
        "n": n,
        "recall_at_5": sum(x["recall5"] for x in results) / n,
        "precision_at_3": sum(x["precision3"] for x in results) / n,
        "mrr": sum(x["mrr"] for x in results) / n,
        "median_latency_ms": sorted(x["latency_ms"] for x in results)[n // 2],
    }
    if not retrieval_only:
        summary["answer_accuracy"] = sum(x["answer_hit"] for x in results) / n
    summary["by_type"] = {
        t: {
            "n": len(v),
            "recall_at_5": sum(x["recall5"] for x in v) / len(v),
            **({"answer_accuracy": sum(x["answer_hit"] for x in v) / len(v)} if not retrieval_only else {}),
        }
        for t, v in sorted(per_type.items())
    }
    summary["thresholds"] = {
        "recall_at_5": EVAL_RECALL_THRESHOLD,
        "answer_accuracy": EVAL_ANSWER_THRESHOLD if not retrieval_only else None,
    }
    summary["passed"] = summary["recall_at_5"] >= EVAL_RECALL_THRESHOLD and (
        retrieval_only or summary["answer_accuracy"] >= EVAL_ANSWER_THRESHOLD
    )
    return {"summary": summary, "results": results}


def write_report(report: dict, path: Path) -> None:
    s = report["summary"]
    lines = [
        "# Margin — Eval Report",
        "",
        f"**Verdict: {'PASS' if s['passed'] else 'FAIL'}** · {s['n']} questions · median retrieval latency {s['median_latency_ms']}ms",
        "",
        "| Metric | Score | Threshold |",
        "|--------|-------|-----------|",
        f"| recall@5 | {s['recall_at_5']:.1%} | {s['thresholds']['recall_at_5']:.0%} |",
        f"| precision@3 | {s['precision_at_3']:.1%} | — |",
        f"| MRR | {s['mrr']:.3f} | — |",
    ]
    if "answer_accuracy" in s:
        lines.append(f"| answer must-contain | {s['answer_accuracy']:.1%} | {s['thresholds']['answer_accuracy']:.0%} |")
    lines += ["", "## By question type", "", "| Type | n | recall@5 | answer |", "|------|---|----------|--------|"]
    for t, v in s["by_type"].items():
        ans = v.get("answer_accuracy", "—")
        ans_cell = f"{ans:.1%}" if isinstance(ans, float) else ans
        lines.append(f"| {t} | {v['n']} | {v['recall_at_5']:.1%} | {ans_cell} |")
    fails = [r for r in report["results"] if not r["recall5"] or r.get("answer_hit") == 0.0]
    if fails:
        lines += ["", "## Failures", ""]
        for r in fails:
            lines.append(f"- **{r['id']}**: `{r['question']}` — gold `{r['gold_doc']}`, got {r['retrieved'][:3]}")
    path.write_text("\n".join(lines) + "\n")


def save_report(report: dict) -> Path:
    """Persist the structured run report for GET /api/evals.

    Writes {example: false, generated_at, summary, results} to
    margin/evals/report.json (EVALS_REPORT_PATH). The human-readable markdown
    report written by write_report()/--report is unaffected — call both from
    main() to get the full pair.
    """
    payload = {
        "example": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **report,
    }
    EVALS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALS_REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return EVALS_REPORT_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description="Margin eval harness")
    ap.add_argument("--retrieval-only", action="store_true", help="skip LLM answer grading (fast, deterministic — CI mode)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    report = run(retrieval_only=args.retrieval_only, limit=args.limit, verbose=not args.quiet)
    s = report["summary"]
    print(json.dumps({k: v for k, v in s.items() if k != "by_type"}, indent=2))
    json_path = save_report(report)  # structured snapshot → GET /api/evals
    print(f"json report → {json_path}")
    if args.report:
        write_report(report, Path(args.report))
        print(f"report → {args.report}")
    sys.exit(0 if s["passed"] else 1)


if __name__ == "__main__":
    main()
