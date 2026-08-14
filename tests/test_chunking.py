from app.chunking import Document, MultiStrategyChunker
from app.embedding import HashingEmbedder


def test_multi_strategy_chunker_builds_distinct_views_with_parent_links():
    document = Document(
        id="doc-1",
        text=(
            "Paris is the capital of France. It is known for the Eiffel Tower.\n\n"
            "The city lies on the Seine river. Visitors often travel there in spring."
        ),
        title="Paris travel facts",
        metadata={"source": "fixture", "language": "en"},
    )

    chunks = MultiStrategyChunker(
        embedder=HashingEmbedder(dim=64),
        token_size=12,
        token_overlap=3,
        semantic_threshold=0.25,
    ).chunk(document)

    strategies = {chunk.strategy for chunk in chunks}
    assert {"paragraph", "sentence_window", "token", "semantic", "metadata"} <= strategies
    assert len(chunks) >= 5
    assert all(chunk.parent_id == "doc-1" for chunk in chunks)
    assert any(chunk.metadata["source"] == "fixture" for chunk in chunks)
    assert any("Paris travel facts" in chunk.text for chunk in chunks if chunk.strategy == "metadata")


def test_token_chunks_have_real_overlap_when_document_is_long():
    document = Document(id="doc-2", text=" ".join(f"word{i}" for i in range(30)))
    chunks = MultiStrategyChunker(
        embedder=HashingEmbedder(dim=32),
        token_size=10,
        token_overlap=3,
        semantic_threshold=0.25,
    ).chunk(document)

    token_chunks = [chunk for chunk in chunks if chunk.strategy == "token"]
    assert len(token_chunks) >= 3
    assert set(token_chunks[0].text.split()[-3:]) & set(token_chunks[1].text.split()[:3])
