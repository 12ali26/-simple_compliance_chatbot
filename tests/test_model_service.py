from src.model_service import (
    DEFAULT_OPENROUTER_MODEL,
    build_grounded_messages,
    generate_grounded_answer,
    generate_openrouter_grounded_answer,
    get_openrouter_settings,
)


def test_openrouter_settings_use_default_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")

    settings = get_openrouter_settings()

    assert settings is not None
    assert settings.api_key == "sk-or-real"
    assert settings.model == DEFAULT_OPENROUTER_MODEL


def test_openrouter_settings_use_configured_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")

    settings = get_openrouter_settings()

    assert settings is not None
    assert settings.model == "anthropic/claude-3.5-haiku"


def test_build_grounded_messages_contains_guardrails_and_sources():
    messages = build_grounded_messages(
        "What PPE is required?",
        [
            {
                "document_title": "PPE SOP",
                "section_title": "Isolation",
                "page_number": 2,
                "chunk_text": "Wear gloves and eye protection.",
            }
        ],
    )

    assert messages[0]["role"] == "system"
    assert "Do not invent" in messages[0]["content"]
    assert "PPE SOP -> Isolation -> Page 2" in messages[1]["content"]


def test_openrouter_request_shape(monkeypatch):
    calls = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Grounded answer"}}]}

    def fake_post(url, headers, json, timeout):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return Response()

    monkeypatch.setattr("src.model_service.requests.post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "http://localhost:8501")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "Cleaning SOP Assistant")

    answer = generate_openrouter_grounded_answer(
        "How do I clean a spill?",
        [{"document_title": "Spill SOP", "section_title": "Page 1", "chunk_text": "Use absorbent material."}],
    )

    assert answer == "Grounded answer"
    assert calls["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer sk-or-real"
    assert calls["headers"]["HTTP-Referer"] == "http://localhost:8501"
    assert calls["headers"]["X-Title"] == "Cleaning SOP Assistant"
    assert calls["json"]["model"] == "openai/gpt-4o-mini"
    assert calls["json"]["messages"][0]["role"] == "system"


def test_generate_grounded_answer_prefers_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(
        "src.model_service.generate_openrouter_grounded_answer",
        lambda question, chunks, settings: "openrouter answer",
    )
    monkeypatch.setattr(
        "src.model_service.generate_openai_grounded_answer",
        lambda question, chunks, settings: "openai answer",
    )

    answer = generate_grounded_answer("Question?", [{"chunk_text": "Context"}])

    assert answer == "openrouter answer"
