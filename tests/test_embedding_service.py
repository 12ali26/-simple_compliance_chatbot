from src.embedding_service import (
    DEFAULT_OPENROUTER_EMBEDDING_MODEL,
    embed_openrouter_texts,
    embed_texts,
    get_embedding_settings,
    get_openrouter_embedding_model,
)


def test_openrouter_embedding_model_default(monkeypatch):
    monkeypatch.delenv("OPENROUTER_EMBEDDING_MODEL", raising=False)

    assert get_openrouter_embedding_model() == DEFAULT_OPENROUTER_EMBEDDING_MODEL


def test_embedding_settings_prefer_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    settings = get_embedding_settings()

    assert settings is not None
    assert settings.provider == "openrouter"
    assert settings.model == DEFAULT_OPENROUTER_EMBEDDING_MODEL


def test_openrouter_embedding_request_shape(monkeypatch):
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    def fake_post(url, headers, json, timeout):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return Response()

    monkeypatch.setattr("src.embedding_service.requests.post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    embeddings = embed_openrouter_texts(["hello"])

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert calls["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert calls["headers"]["Authorization"] == "Bearer sk-or-real"
    assert calls["json"]["model"] == "openai/text-embedding-3-small"
    assert calls["json"]["input"] == ["hello"]


def test_embed_texts_prefers_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr("src.embedding_service.embed_openrouter_texts", lambda texts: [[1, 2, 3]])
    monkeypatch.setattr("src.embedding_service.embed_openai_texts", lambda texts, settings: [[4, 5, 6]])

    assert embed_texts(["question"]) == [[1, 2, 3]]
