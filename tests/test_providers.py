import app.providers as providers


def test_ollama_cloud_generator_uses_native_chat_api(monkeypatch):
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"role": "assistant", "content": "Grounded answer [S1]"}}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(providers.httpx, "post", fake_post)
    generator = providers.OllamaCloudAnswerGenerator(
        api_key="ollama-test-key",
        model="gpt-oss:120b",
        base_url="https://ollama.com",
    )

    answer = generator.generate("What is Goa?", ["Goa is a state in India."])

    assert answer == "Grounded answer [S1]"
    assert seen["url"] == "https://ollama.com/api/chat"
    assert seen["kwargs"]["headers"] == {"Authorization": "Bearer ollama-test-key"}
    assert seen["kwargs"]["json"]["model"] == "gpt-oss:120b"
    assert seen["kwargs"]["json"]["stream"] is False
    assert "Goa is a state in India." in seen["kwargs"]["json"]["messages"][1]["content"]
