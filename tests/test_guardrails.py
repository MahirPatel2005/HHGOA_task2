from app.chunking import Chunk
from app.guardrails import GuardrailEngine
from app.retrieval import RetrievalResult


def test_retrieval_guardrail_labels_unrelated_queries_off_topic():
    result = RetrievalResult(
        chunk=Chunk("c1", "d1", "Goa is in India.", "paragraph", {}, 0, 4),
        score=0.03,
        dense_score=0.10,
        lexical_score=0.0,
        lexical_overlap=0.0,
    )
    decision = GuardrailEngine().check_retrieval("Hi, how are you?", [result])
    assert not decision.allowed
    assert decision.reason == "off_topic"
