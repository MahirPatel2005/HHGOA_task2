import app.main as main
from app.providers import OllamaCloudAnswerGenerator


def test_create_orchestrator_selects_ollama_cloud(monkeypatch, tmp_path):
    monkeypatch.setenv("GENERATION_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-test-key")
    monkeypatch.setenv("VECTOR_BACKEND", "local")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LOCAL_INDEX_PATH", str(tmp_path / "index.json"))
    monkeypatch.setattr(main, "sample_documents", lambda: [main.Document(id="d1", text="Goa is in India.")])

    orchestrator = main.create_orchestrator()

    assert isinstance(orchestrator.answer_generator, OllamaCloudAnswerGenerator)
    assert orchestrator.answer_generator.model == "gpt-oss:120b"
