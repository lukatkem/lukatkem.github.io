#!/usr/bin/env python3
"""Knowledge distillation — frontier teachers in the cloud, student on this Mac.

This is how small models are actually made in the field: a frontier model
(GLM 5.3 etc., running on the provider's servers — nothing downloads) writes
high-quality training text AND grades it, and only the examples that pass
reach the student's corpus. Hydro-1 then retrains on the distilled set.

    python -m hydro1.distill --stories 200          # ~10–30 min, costs provider tokens
    python -m hydro1.train --corpus corpus/distilled.txt --steps 8000

Without TEACHER_API_KEY the script falls back to the local Ollama teacher —
slower and smaller, but it works and proves the pipeline end-to-end.
"""
from __future__ import annotations

import argparse
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import all_teachers, cloud_teachers
from .teachers import TeacherError, teacher_chat

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "corpus"

WRITE_SYSTEM = (
    "You write original short children's stories for training a tiny language "
    "model. Each story: 150-260 words, simple sentences, warm tone, a named "
    "character, a small everyday adventure with a beginning, middle and end. "
    "Plain prose only — no titles, no lists, no commentary. Output exactly one "
    "story."
)

GRADE_SYSTEM = (
    "You grade children's stories for training-data quality. A story passes if "
    "it is coherent from start to end, grammatical, appropriate for children, "
    "and between 100 and 300 words. Think as needed, then end your reply with "
    "exactly 'VERDICT: PASS' or 'VERDICT: FAIL'."
)

# topic seeds keep generations diverse instead of 200 near-clones
SEEDS = [
    "a lost mitten", "the first snow", "a paper boat", "a sleepless firefly",
    "a puddle after rain", "the bakery cat", "a kite that wouldn't land",
    "grandma's radio", "a snail race", "the missing button", "a moonlit garden",
    "the bus-stop dog", "a jar of fireflies", "the last apple of autumn",
    "a cardboard rocket", "the lighthouse keeper's lunch", "a windy Tuesday",
    "the duckling who feared water", "a map drawn in chalk", "the quiet drum",
]


def write_story(model_id: str, seed: str) -> str | None:
    prompt = f"Write one story about {seed}."
    # frontier teachers are reasoning models — headroom for thinking + prose
    text, _ = teacher_chat(model_id, prompt, WRITE_SYSTEM, temperature=1.0, max_tokens=2000)
    text = text.strip()
    words = len(text.split())
    if not (80 <= words <= 400):
        return None
    if re.search(r"^(#|\d+\.|here('|’)s|sure|certainly)", text, re.I):
        return None  # chatty preamble, not a story
    return text


def parse_verdict(text: str) -> bool:
    """The graded verdict is the LAST 'VERDICT:' marker — reasoning models
    deliberate first and answer last, and their verdict may land in the
    reasoning text rather than the content field."""
    v = text.upper()
    p, f = v.rfind("VERDICT: PASS"), v.rfind("VERDICT: FAIL")
    if p == -1 and f == -1:
        return False  # never reached a verdict → treat as failed
    return p > f


def grade_story(model_id: str, story: str) -> bool:
    try:
        verdict, _ = teacher_chat(model_id, story[:2000], GRADE_SYSTEM, temperature=0.0, max_tokens=2048)
        return parse_verdict(verdict)
    except TeacherError:
        return False  # teacher hiccup → drop the example, never train on ungraded text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=int, default=200)
    ap.add_argument("--grade", action="store_true", help="second-pass quality filter (doubles teacher calls)")
    ap.add_argument("--model", type=str, default=None, help="teacher model id (default: first frontier)")
    ap.add_argument("--grade-model", type=str, default=None, help="grader model id (default: first fast tier)")
    ap.add_argument("--out", type=str, default="distilled.txt")
    args = ap.parse_args()

    roster = all_teachers()
    cloud_r = cloud_teachers()
    model_id = (
        args.model
        or next((t["id"] for t in cloud_r if t.get("tier") == "frontier"), None)
        or (cloud_r[0]["id"] if cloud_r else roster[0]["id"])
    )
    # grading only needs a verdict — use the cheap fast tier when available
    grade_model = args.grade_model or next((t["id"] for t in cloud_r if t.get("tier") == "fast"), model_id)
    mode = "cloud" if cloud_r else "local-fallback (no provider key set)"
    print(f"teacher: {model_id} · grader: {grade_model} · mode: {mode}")

    seeds = [SEEDS[i % len(SEEDS)] for i in range(args.stories)]
    random.shuffle(seeds)
    stories: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(write_story, model_id, s): s for s in seeds}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                s = fut.result()
                if s:
                    stories.append(s)
            except TeacherError as e:
                print(f"  ! {e}")
            if i % 10 == 0:
                print(f"  generated {i}/{len(seeds)} → kept {len(stories)}")

    if args.grade:
        before = len(stories)
        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = list(pool.map(lambda s: grade_story(grade_model, s), stories))
        stories = [s for s, ok in zip(stories, verdicts) if ok]
        print(f"grading: {before} → {len(stories)} passed")

    out = OUT_DIR / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n\n".join(stories) + "\n", encoding="utf-8")
    print(f"wrote {len(stories)} stories · {out.stat().st_size/1024:.0f} KB → {out}")
    print(f"next: python -m hydro1.train --corpus {out} --steps 8000 --out-suffix distilled")


if __name__ == "__main__":
    main()
