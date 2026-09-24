"""Deterministic question generation from document sentences."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .pairs import Pair
from .quality import QualityFilter

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_DEFINITION = re.compile(
    r"^(?:the\s+)?(?P<subject>[A-Za-z][\w\s-]{2,60}?)\s+"
    r"(?P<verb>is|are|was|were)\s+(?P<predicate>[a-z0-9].{8,200}?)[.\s]*$", re.I)
_NUMBER = re.compile(r".*?(?P<subject>[A-Za-z][\w\s-]{2,50}?)\s+"
                     r"(?:has|have|had|contains?|includes?|supports?)\s+"
                     r"(?P<num>\d[\d,]*)(?P<unit>\s+\w+)?")
_ENTITY = re.compile(r"\b([A-Z][a-z]{2,})\b")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if 30 < len(s.strip()) < 400]


def generate_pairs(doc_id: str, text: str, flt: QualityFilter | None = None) -> list[Pair]:
    flt = flt or QualityFilter()
    pairs: list[Pair] = []
    seen_q: set[str] = set()
    for sent in _sentences(text):
        m = _DEFINITION.match(sent)
        if m:
            subj, verb, pred = (m.group("subject").strip(), m.group("verb").strip(),
                                m.group("predicate").strip())
            lead_num = re.match(r"^(\d[\d,]*)\s+(\w+)", pred)
            if lead_num:
                subj_l = subj[0].lower() + subj[1:] if subj else subj
                pair = Pair(doc_id, "numeric",
                            f"How many {lead_num.group(2)} does {subj_l} have?",
                            lead_num.group(1), sent)
                if not flt.accept(pair) and pair.question.lower() not in seen_q:
                    seen_q.add(pair.question.lower())
                    pairs.append(pair)
                continue
            q = f"What {verb} {subj}?"
            a = f"{subj} {verb} {pred}"
            a = a[0].upper() + a[1:]
            pair = Pair(doc_id, "definition", q, a.rstrip(".") + ".", sent)
            if not flt.accept(pair) and pair.question.lower() not in seen_q:
                seen_q.add(pair.question.lower())
                pairs.append(pair)
            continue
        m = _NUMBER.match(sent)
        if m:
            unit = (m.group("unit") or "").strip()
            q = f"How many {unit} does {m.group('subject').strip()} have?" if unit \
                else f"What number is associated with {m.group('subject').strip()}?"
            pair = Pair(doc_id, "numeric", q, m.group("num"), sent)
            if not flt.accept(pair) and pair.question.lower() not in seen_q:
                seen_q.add(pair.question.lower())
                pairs.append(pair)
            continue
        entities = [e for e in _ENTITY.findall(sent) if e.lower() not in
                    ("the", "she", "he", "they", "it", "a", "an", "one", "his", "her")]
        if len(entities) >= 1 and len(sent.split()) >= 8:
            entity = entities[0]
            cloze = sent.replace(entity, "_____", 1)
            q = f"Fill in the blank: {cloze}"
            pair = Pair(doc_id, "cloze", q, entity, sent)
            if not flt.accept(pair) and pair.question.lower() not in seen_q:
                seen_q.add(pair.question.lower())
                pairs.append(pair)
    return pairs
