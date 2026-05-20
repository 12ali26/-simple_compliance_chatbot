from pathlib import Path

from src.sop_engine import FALLBACK_RESPONSE, answer_question, load_knowledge_base


ROOT = Path(__file__).resolve().parents[1]


def test_answers_blood_spill_with_sources():
    chunks = load_knowledge_base(ROOT / "knowledge_base")

    result = answer_question("How do I clean a blood spill?", chunks)

    assert result.answer != FALLBACK_RESPONSE
    assert "approved procedure" in result.answer
    assert any(source["document_title"] == "Spill Response SOP" for source in result.sources)


def test_unknown_question_falls_back():
    chunks = load_knowledge_base(ROOT / "knowledge_base")

    result = answer_question("How do I apply for annual leave?", chunks)

    assert result.answer == FALLBACK_RESPONSE
    assert result.sources == []


def test_chemical_ratios_are_not_invented():
    chunks = load_knowledge_base(ROOT / "knowledge_base")

    result = answer_question("What dilution ratio should I use for chemicals?", chunks)

    assert "Do not guess chemical ratios" in result.answer
    assert "1:" not in result.answer
