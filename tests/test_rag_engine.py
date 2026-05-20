from src.rag_engine import (
    answer_question_best_available,
    normalize_query,
    retrieve_best_evidence,
)
from src.sop_engine import FALLBACK_RESPONSE, DocumentChunk


class FakeStore:
    pass


def make_chunks():
    return [
        DocumentChunk(
            document_title="Housekeeping MS",
            category="ppe",
            source_name="Test SOP",
            section_title="Donning & Doffing",
            chunk_text=(
                "Required Materials include Disposable Mask and N95 Face Mask - Optional. "
                "Put on N95 face mask or disposable face mask during donning."
            ),
            file_path="test.md",
            page_number=2,
        ),
        DocumentChunk(
            document_title="Housekeeping MS",
            category="isolation room cleaning",
            source_name="Test SOP",
            section_title="Isolation Room Terminal Cleaning",
            chunk_text=(
                "Task Description Terminal Cleaning for Droplet, Airborne Isolation & Protective Environment. "
                "2. Required Personal Protective Equipment (PPE). "
                "Surgical Mask (For Droplet Isolation & Protective Environment). "
                "N95 Mask (For Airborne Isolation). "
                "Disposable Gown. Disposable Shoe Cover. Disposable Head Cover."
            ),
            file_path="test.md",
            page_number=77,
        ),
        DocumentChunk(
            document_title="Housekeeping MS",
            category="isolation room cleaning",
            source_name="Test SOP",
            section_title="Isolation Room Terminal Cleaning Contact",
            chunk_text=(
                "Task Description Terminal Cleaning for Contact Isolation. "
                "2. Required Personal Protective Equipment (PPE). "
                "Surgical Mask (For Droplet Isolation & Protective Environment). "
                "N95 Mask (For Airborne Isolation). "
                "Disposable Gown. Disposable Shoe Cover. Disposable Head Cover."
            ),
            file_path="test.md",
            page_number=84,
        ),
        DocumentChunk(
            document_title="Housekeeping MS",
            category="spill response",
            source_name="Test SOP",
            section_title="Spill Procedure",
            chunk_text="Stop work and prevent people from walking through the spill area.",
            file_path="test.md",
            page_number=1,
        ),
    ]


def test_normalize_query_handles_common_staff_variants():
    assert normalize_query("what PPEs are used in I solation room cleaning") == (
        "what ppe are used in isolation room cleaning"
    )
    assert normalize_query("when do I wear an N95 respirator") == "when do i wear an n95 mask"
    assert normalize_query("downing and doff steps") == "donning and doffing steps"


def test_best_evidence_reranks_exact_ppe_table_above_general_donning_page():
    evidence = retrieve_best_evidence(
        "which isolation type requires wearing an N95 mask",
        make_chunks(),
        FakeStore(),
        semantic_enabled=False,
    )

    assert evidence[0].page_number == 77
    assert "Required Personal Protective Equipment" in evidence[0].chunk_text
    assert "Airborne Isolation" in evidence[0].chunk_text


def test_best_available_falls_back_to_local_chunks_when_semantic_has_no_match():
    result = answer_question_best_available(
        "What procedure should I follow when a spill occurs?",
        make_chunks(),
        FakeStore(),
        semantic_enabled=True,
        model_enabled=False,
    )

    assert result.answer != FALLBACK_RESPONSE
    assert result.sources[0]["document_title"] == "Housekeeping MS"


def test_n95_question_variants_answer_airborne_isolation():
    questions = [
        "which isolation type requires wearing an N95 mask",
        "when do I wear an N95 respirator",
        "is N95 for droplet or airborne isolation",
        "what mask should I wear for airborne isolation cleaning",
    ]

    for question in questions:
        result = answer_question_best_available(
            question,
            make_chunks(),
            FakeStore(),
            semantic_enabled=False,
            model_enabled=False,
        )

        assert result.answer == "N95 Mask is for Airborne Isolation."
        assert any(source["page_number"] == "77" for source in result.sources)


def test_contact_isolation_question_does_not_say_contact_requires_n95():
    result = answer_question_best_available(
        "does contact isolation require N95",
        make_chunks(),
        FakeStore(),
        semantic_enabled=False,
        model_enabled=False,
    )

    assert "N95 Mask for Airborne Isolation" in result.answer
    assert "Contact Isolation requires an N95" not in result.answer
    assert "does not list N95 as the Contact Isolation requirement" in result.answer
    assert any(source["page_number"] == "84" for source in result.sources)


def test_droplet_mask_question_answers_surgical_mask():
    result = answer_question_best_available(
        "what mask is used for droplet isolation",
        make_chunks(),
        FakeStore(),
        semantic_enabled=False,
        model_enabled=False,
    )

    assert result.answer == "Surgical Mask is for Droplet Isolation and Protective Environment."
    assert any(source["page_number"] == "77" for source in result.sources)
