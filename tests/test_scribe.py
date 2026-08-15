import app.main as main
from fastapi.testclient import TestClient


def test_scribe_token_fails_without_key(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    api = main.create_app()
    with TestClient(api) as client:
        response = client.get("/api/voice/scribe-token")
        assert response.status_code == 400
        assert "not configured" in response.json()["error"]


def test_query_route_accepts_mode_parameter(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_INDEX_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("VECTOR_BACKEND", "local")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    
    # Prepopulate documents
    monkeypatch.setattr(main, "sample_documents", lambda: [main.Document(id="d1", text="Goa is a state in western India.")])
    
    api = main.create_app()
    with TestClient(api) as client:
        # Check that post with mode=fast is accepted
        response = client.post(
            "/api/query",
            json={"query": "Where is Goa?", "language_code": "en", "mode": "fast"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "answered"
        assert "Goa is a state" in data["answer"]
