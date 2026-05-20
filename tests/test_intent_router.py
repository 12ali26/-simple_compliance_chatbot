from src.intent_router import Intent, keyword_route, route_message


def test_hello_routes_to_casual(monkeypatch):
    monkeypatch.setattr("src.intent_router.model_route", lambda message: None)

    result = route_message("hello")

    assert result.intent == Intent.CASUAL
    assert "approved cleaning procedures" in result.response


def test_thanks_routes_to_casual():
    result = keyword_route("thanks")

    assert result.intent == Intent.CASUAL
    assert result.response == "You are welcome."


def test_blood_spill_routes_to_sop_question():
    result = keyword_route("How do I clean a blood spill?")

    assert result.intent == Intent.SOP_QUESTION
    assert "blood spill" in result.search_query.lower()


def test_leave_routes_to_unsupported():
    result = keyword_route("How do I apply for leave?")

    assert result.intent == Intent.UNSUPPORTED
    assert "approved cleaning procedures" in result.response


def test_model_cannot_override_keyword_sop_question_as_unsupported(monkeypatch):
    monkeypatch.setenv("ENABLE_MODEL_ROUTER", "true")
    monkeypatch.setattr(
        "src.intent_router.model_route",
        lambda message: type(
            "ModelResult",
            (),
            {
                "intent": Intent.UNSUPPORTED,
                "response": "Unsupported.",
                "search_query": "",
                "reason": "Model rejected it.",
            },
        )(),
    )

    result = route_message("What PPE is required for isolation room cleaning?")

    assert result.intent == Intent.SOP_QUESTION
    assert "isolation room cleaning" in result.search_query.lower()


def test_model_router_is_disabled_by_default(monkeypatch):
    def fail_if_called(message):
        raise AssertionError("model router should not be called")

    monkeypatch.delenv("ENABLE_MODEL_ROUTER", raising=False)
    monkeypatch.setattr("src.intent_router.model_route", fail_if_called)

    result = route_message("What PPE is required for isolation room cleaning?")

    assert result.intent == Intent.SOP_QUESTION
