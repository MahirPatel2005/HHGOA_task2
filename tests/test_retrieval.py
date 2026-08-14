from app.chunking import Chunk
from app.embedding import HashingEmbedder
from app.retrieval import HybridRetriever
from app.vector_store import LocalVectorStore


def test_hybrid_retrieval_fuses_dense_and_lexical_results():
    embedder = HashingEmbedder(dim=96)
    store = LocalVectorStore(dim=96)
    chunks = [
        Chunk("c1", "d1", "The Eiffel Tower is in Paris, France.", "paragraph", {}, 0, 8),
        Chunk("c2", "d2", "Python is a programming language.", "paragraph", {}, 0, 6),
        Chunk("c3", "d3", "The Seine river flows through Paris.", "semantic", {}, 0, 7),
    ]
    store.upsert([(chunk.id, embedder.embed(chunk.text), chunk) for chunk in chunks])
    retriever = HybridRetriever(store=store, embedder=embedder)
    results = retriever.search("Where is the Eiffel Tower?", top_k=2)

    assert results
    assert results[0].chunk.id == "c1"
    assert results[0].score > 0
    assert results[0].retrieval_sources
