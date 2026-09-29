"""Request-side injection scan — lightweight, weighted rules."""
from __future__ import annotations
import enum, re
from dataclasses import dataclass

_RULES = [
    ("instruction_override", 0.50, re.compile(r"ignore (all |any )?(previous|prior|above) (instructions|rules|prompts)", re.I)),
    ("instruction_hijack", 0.45, re.compile(r"(new|revised|updated) (instructions|rules|system prompt)[:\s]", re.I)),
    ("role_pretend", 0.40, re.compile(r"pretend (you are|to be)|you are now|act as (if )?you (are|were)", re.I)),
    ("data_exfiltration", 0.50, re.compile(r"(repeat|print|output|show|reveal|give me|tell me) (your |the )?(system prompt|instructions|rules)", re.I)),
    ("delimiter_break", 0.20, re.compile(r"(```|---|===){3,}", re.I)),
    ("jailbreak_marker", 0.45, re.compile(r"\bDAN\b|developer mode|do anything now", re.I)),
]
_THRESHOLD_SUSPICIOUS = 0.35
_THRESHOLD_BLOCKED = 0.70

class ScanVerdict(enum.Enum):
    CLEAN = "clean"; SUSPICIOUS = "suspicious"; BLOCKED = "blocked"

@dataclass
class ScanResult:
    verdict: ScanVerdict; score: float; matched_rules: list

def scan_request(text: str) -> ScanResult:
    score = 0.0; matched = []
    for rule_name, weight, pattern in _RULES:
        m = pattern.search(text)
        if m: score += weight; matched.append(f"{rule_name}@{m.start()}")
    if score >= _THRESHOLD_BLOCKED: verdict = ScanVerdict.BLOCKED
    elif score >= _THRESHOLD_SUSPICIOUS: verdict = ScanVerdict.SUSPICIOUS
    else: verdict = ScanVerdict.CLEAN
    return ScanResult(verdict=verdict, score=round(score, 4), matched_rules=matched)
