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
