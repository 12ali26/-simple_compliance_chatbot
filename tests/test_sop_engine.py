from src.sop_engine import FALLBACK_RESPONSE, answer_question, load_knowledge_base


def make_knowledge_base(tmp_path):
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "spill_response.md").write_text(
        """# Spill Response SOP
Category: spill response
Source: Test Spill SOP

## Blood Or Body Fluid Spill
- Cover the spill with approved absorbent material.
- Clean and disinfect the area again after the spill material is removed.

## General Spill Safety
- Stop work and prevent people from walking through the spill area.
""",
        encoding="utf-8",
    )
    (markdown_dir / "chemical_dilution.md").write_text(
        """# Chemical Dilution SOP
Category: chemical dilution
Source: Test Chemical SOP

## Dilution Rules
- Follow the site dilution chart or dispensing system.
- Do not guess chemical ratios.
- Never mix chemicals together.
""",
        encoding="utf-8",
    )
    return tmp_path


def test_answers_blood_spill_with_sources(tmp_path):
    chunks = load_knowledge_base(make_knowledge_base(tmp_path))

    result = answer_question("How do I clean a blood spill?", chunks)

    assert result.answer != FALLBACK_RESPONSE
    assert "approved procedure" in result.answer
    assert any(source["document_title"] == "Spill Response SOP" for source in result.sources)


def test_unknown_question_falls_back(tmp_path):
    chunks = load_knowledge_base(make_knowledge_base(tmp_path))

    result = answer_question("How do I apply for annual leave?", chunks)

    assert result.answer == FALLBACK_RESPONSE
    assert result.sources == []


def test_chemical_ratios_are_not_invented(tmp_path):
    chunks = load_knowledge_base(make_knowledge_base(tmp_path))

    result = answer_question("What dilution ratio should I use for chemicals?", chunks)

    assert "Do not guess chemical ratios" in result.answer
    assert "1:" not in result.answer


def test_empty_uploaded_knowledge_base_has_no_chunks(tmp_path):
    (tmp_path / "markdown").mkdir()

    assert load_knowledge_base(tmp_path) == []
