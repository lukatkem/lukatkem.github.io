"""Quality gates — every generated pair must clear them or it's dropped."""
from __future__ import annotations

from dataclasses import dataclass

from .pairs import Pair


@dataclass
class QualityFilter:
    min_question_len: int = 15
    min_answer_len: int = 1
    max_answer_words: int = 60

    def accept(self, pair: Pair) -> list[str]:
        """Return the list of FAILED checks (empty list = accept)."""
        failed = []
        if len(pair.question) < self.min_question_len:
            failed.append("question_too_short")
        if len(pair.answer) < self.min_answer_len:
            failed.append("answer_too_short")
        if len(pair.answer.split()) > self.max_answer_words:
            failed.append("answer_too_long")
        if pair.answer.lower() in pair.question.lower():
            failed.append("answer_leaks_into_question")
        if not pair.answer.strip():
            failed.append("answer_empty")
        return failed
