from src.logging_store import (
    append_chat_log,
    append_unanswered_question,
    delete_chat_log,
    delete_unanswered_question,
    list_chat_logs,
    list_unanswered_questions,
)


def test_list_and_delete_chat_logs_preserves_header(tmp_path):
    append_chat_log(
        tmp_path,
        {
            "id": "keep",
            "question": "How do I clean a sink?",
            "answer": "Use the approved sink procedure.",
            "category": "washroom cleaning",
            "source_used": "SOP",
            "helpful": "",
            "created_at": "2026-05-20T00:00:00Z",
        },
    )
    append_chat_log(
        tmp_path,
        {
            "id": "delete",
            "question": "Delete me",
            "answer": "Answer",
            "category": "general",
            "source_used": "",
            "helpful": "",
            "created_at": "2026-05-20T00:01:00Z",
        },
    )

    assert len(list_chat_logs(tmp_path)) == 2
    assert delete_chat_log(tmp_path, "delete") is True
    assert delete_chat_log(tmp_path, "missing") is False

    rows = list_chat_logs(tmp_path)
    assert [row["id"] for row in rows] == ["keep"]
    assert (tmp_path / "chat_logs.csv").read_text(encoding="utf-8").startswith("id,question,answer")


def test_list_and_delete_unanswered_questions_preserves_header(tmp_path):
    append_unanswered_question(
        tmp_path,
        {
            "id": "keep",
            "question": "Known gap?",
            "reason": "No source found.",
            "created_at": "2026-05-20T00:00:00Z",
        },
    )
    append_unanswered_question(
        tmp_path,
        {
            "id": "delete",
            "question": "Delete me",
            "reason": "No source found.",
            "created_at": "2026-05-20T00:01:00Z",
        },
    )

    assert len(list_unanswered_questions(tmp_path)) == 2
    assert delete_unanswered_question(tmp_path, "delete") is True
    assert delete_unanswered_question(tmp_path, "missing") is False

    rows = list_unanswered_questions(tmp_path)
    assert [row["id"] for row in rows] == ["keep"]
    assert (tmp_path / "unanswered_questions.csv").read_text(encoding="utf-8").startswith("id,question,reason")
