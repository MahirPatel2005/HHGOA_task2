from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import RetrievalResult, tokens

UNSAFE_PATTERNS = [
    r"\bhow\s+to\s+(build|make|create)\b.*\b(bomb|explosive|weapon)\b",
    r"\b(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)\b",
    r"\bkill\s+(someone|a\s+person)\b",
    r"\b(child sexual|sexual exploitation|csam)\b",
    r"\bsteal\s+(passwords?|credentials?)\b",
]


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str = "allowed"
    message: str = ""
    score: float = 0.0


class GuardrailEngine:
    def __init__(self, min_retrieval_score: float = 0.025, min_dense_score: float = 0.52, min_grounding_overlap: float = 0.20):
        self.min_retrieval_score = min_retrieval_score
        self.min_dense_score = min_dense_score
        self.min_grounding_overlap = min_grounding_overlap
        self._unsafe = [re.compile(pattern, re.IGNORECASE) for pattern in UNSAFE_PATTERNS]

    def check_input(self, query: str) -> GuardrailDecision:
        normalized = query.strip()
        if not normalized:
            return GuardrailDecision(False, "empty_query", "Please ask a question.")
        if len(normalized) > 2000:
            return GuardrailDecision(False, "query_too_long", "Please shorten the question.")
        if re.search(r"ignore\s+(all\s+)?previous\s+instructions|reveal\s+the\s+system\s+prompt", normalized, re.I):
            return GuardrailDecision(False, "prompt_injection", "I can only answer questions using the indexed dataset.")
        if any(pattern.search(normalized) for pattern in self._unsafe):
            return GuardrailDecision(False, "unsafe_input", "I can't help with unsafe or harmful instructions.")
        return GuardrailDecision(True)

    def check_retrieval(self, query: str, results: list[RetrievalResult]) -> GuardrailDecision:
        if not results:
            return GuardrailDecision(False, "insufficient_context", "I don't have enough evidence in the indexed dataset to answer that.")
        best = results[0]
        # A lexical hit is a strong signal. Otherwise require a high semantic
        # score so unrelated hash/vector collisions do not become answers.
        unrelated = best.lexical_overlap <= 0.0 and best.dense_score < self.min_dense_score
        if unrelated:
            return GuardrailDecision(False, "off_topic", "That question is outside the indexed dataset, so I can't answer it.", best.score)
        if best.score < self.min_retrieval_score:
            return GuardrailDecision(False, "insufficient_context", "I don't have enough evidence in the indexed dataset to answer that.", best.score)
        return GuardrailDecision(True, score=best.score)

    def check_answer(self, answer: str, contexts: list[str]) -> GuardrailDecision:
        if not answer.strip():
            return GuardrailDecision(False, "empty_answer", "I couldn't produce an answer from the retrieved evidence.")
        context_tokens = set(tokens(" ".join(contexts)))
        answer_tokens = set(tokens(answer))
        overlap = len(context_tokens & answer_tokens) / max(1, len(answer_tokens))
        if overlap < self.min_grounding_overlap:
            return GuardrailDecision(False, "ungrounded_answer", "I don't have enough evidence to support that answer.", overlap)
        if not re.search(r"(?:\[S\d+\]|【S\d+】)", answer):
            return GuardrailDecision(False, "missing_citation", "I couldn't verify an evidence-backed answer.")
        return GuardrailDecision(True, score=overlap)
