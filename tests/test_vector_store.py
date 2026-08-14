from app.chunking import Chunk
from app.vector_store import QdrantVectorStore


def test_qdrant_upsert_sends_small_remote_batches(tmp_path):
    store = QdrantVectorStore(2, path=str(tmp_path / "qdrant"), collection="batch-test", batch_size=2, timeout_s=90)
    calls = []
    store.client.upsert = lambda **kwargs: calls.append(kwargs)
    records = [
        (
            f"c{i}",
            [1.0, 0.0],
            Chunk(f"c{i}", "d", f"text {i}", "paragraph", {}, i, i + 1),
        )
        for i in range(5)
    ]

    store.upsert(records)

    assert [len(call["points"]) for call in calls] == [2, 2, 1]
    store.close()
