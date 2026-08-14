from app.chunking import Chunk
from app.embedding import HashingEmbedder
from app.guardrails import GuardrailEngine
from app.harness import RAGOrchestrator
from app.providers import TemplateAnswerGenerator
from app.retrieval import HybridRetriever
from app.schemas import PipelineRequest
from app.vector_store import LocalVectorStore


def make_orchestrator():
    embedder = HashingEmbedder(dim=96)
    store = LocalVectorStore(dim=96)
    chunk = Chunk(
        "c1", "d1", "Paris is the capital of France.", "paragraph", {"source": "fixture"}, 0, 6
    )
    store.upsert([(chunk.id, embedder.embed(chunk.text), chunk)])
    return RAGOrchestrator(
        retriever=HybridRetriever(store, embedder),
        answer_generator=TemplateAnswerGenerator(),
        guardrails=GuardrailEngine(min_retrieval_score=0.01),
    )


def test_harness_returns_structured_grounded_answer():
    response = make_orchestrator().run(PipelineRequest(query_text="What is the capital of France?"))
    assert response.status == "answered"
    assert response.answer
    assert [citation.chunk_id for citation in response.citations] == ["c1"]
    assert response.stage_latency_ms["retrieval"] >= 0
    assert response.request_id


def test_harness_abstains_when_context_is_missing():
    response = make_orchestrator().run(PipelineRequest(query_text="What is the tallest mountain on Mars?"))
    assert response.status == "abstained"
    assert response.refusal_reason in {"off_topic", "insufficient_context", "ungrounded_answer"}
