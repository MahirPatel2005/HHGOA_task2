import app.main as main
from fastapi.testclient import TestClient


def test_health_and_config_do_not_construct_pipeline(monkeypatch):
    calls = []

    def unexpected_pipeline_build():
        calls.append(True)
        raise AssertionError("pipeline construction must be lazy")

    monkeypatch.setattr(main, "create_orchestrator", unexpected_pipeline_build)
    api = main.create_app()

    with TestClient(api) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/config").status_code == 200

    assert calls == []
