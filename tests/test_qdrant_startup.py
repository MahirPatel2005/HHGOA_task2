import sys
from types import SimpleNamespace

from app.vector_store import QdrantVectorStore


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scroll_calls = 0

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name="voice_rag_demo")])

    def scroll(self, **kwargs):
        self.scroll_calls += 1
        raise AssertionError("remote startup must not scan every Qdrant payload")

    def close(self):
        pass


class _FakeModels:
    class VectorParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Distance:
        COSINE = "Cosine"



def test_remote_qdrant_startup_does_not_scroll_entire_collection(monkeypatch):
    fake_module = SimpleNamespace(QdrantClient=_FakeClient, models=_FakeModels)
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_module)

    store = QdrantVectorStore(
        dim=384,
        collection="voice_rag_demo",
        url="https://qdrant.example.com",
        api_key="test-key",
    )

    assert store.client.scroll_calls == 0
