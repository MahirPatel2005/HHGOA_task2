from app.chunking import Chunk
from app.embedding import HashingEmbedder
from app.retrieval import HybridRetriever


class _RemoteLikeStore:
    records = {}

    def search(self, vector, limit=10):
        return [
            (
                Chunk(
                    "remote-1",
                    "doc-1",
                    "Panaji is the capital of Goa.",
                    "paragraph",
                    {},
                    0,
                    1,
                ),
                0.9,
            )
        ]


def test_qdrant_dense_hits_are_usable_without_local_records():
    retriever = HybridRetriever(_RemoteLikeStore(), HashingEmbedder(dim=8))

    results = retriever.search("What is the capital of Goa?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "remote-1"
