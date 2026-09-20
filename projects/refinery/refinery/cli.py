#!/usr/bin/env python3
"""Refinery CLI — refine text files into a training-grade corpus.

    python -m refinery.cli --input DIR [--input FILE…] --out clean.txt --report report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import Doc, Refinery
from .report import report

TEXT_SUFFIXES = {".txt", ".md"}


def load_inputs(inputs: list[Path]) -> list[Doc]:
    docs: list[Doc] = []
    for p in inputs:
        if p.is_dir():
            files = sorted(q for q in p.rglob("*") if q.suffix.lower() in TEXT_SUFFIXES)
        else:
            files = [p]
        for f in files:
            try:
                docs.append(Doc(id=f.name, source=str(f), text=f.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description="raw text in, training-grade corpus out")
    ap.add_argument("--input", action="append", required=True, help="file or directory (repeatable)")
    ap.add_argument("--out", default="corpus.clean.txt")
    ap.add_argument("--report", default="refinery-report.md")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--threshold", type=float, default=0.8, help="near-dup jaccard threshold")
    ap.add_argument("--min-quality", type=float, default=0.6)
    ap.add_argument("--no-scrub", action="store_true", help="skip PII/secret scrubbing (not recommended)")
    args = ap.parse_args()

    docs = load_inputs([Path(p) for p in args.input])
    if not docs:
        raise SystemExit("no .txt/.md documents found in the given inputs")

    engine = Refinery(lang=args.lang, dedup_threshold=args.threshold,
                      min_quality=args.min_quality, scrub_pii=not args.no_scrub)
    result = engine.run(docs)

    out = Path(args.out)
    out.write_text("\n\n".join(v.doc.text for v in result.kept) + "\n", encoding="utf-8")
    Path(args.report).write_text(report(result, [str(p) for p in args.input]), encoding="utf-8")

    kept = len(result.kept)
    print(f"refined {len(result.verdicts)} docs → kept {kept} ({kept / max(len(result.verdicts), 1):.0%}) "
          f"in {result.elapsed_s:.2f}s")
    print(f"clean corpus → {out}  ·  report → {args.report}")
    for stage, n in result.funnel.items():
        print(f"  {stage:<22} {n}")


if __name__ == "__main__":
    main()
