from pathlib import Path

import pytest

from src import pdf_ingestion
from src.pdf_ingestion import (
    chunk_markdown,
    ingest_pdf_bytes,
    pages_to_markdown,
    sha256_bytes,
)


fitz = pytest.importorskip("fitz")


def make_pdf(text: str | None) -> bytes:
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def test_pdf_text_extraction_returns_markdown(tmp_path: Path):
    content = make_pdf("Blood spill cleanup requires gloves and approved disinfectant.")

    result = ingest_pdf_bytes(
        filename="spill-sop.pdf",
        content=content,
        title="Spill SOP",
        category="spill response",
        source_name="Approved Spill PDF",
        knowledge_dir=tmp_path,
    )

    assert result.raw_pdf_path.exists()
    assert result.markdown_path.exists()
    assert "# Spill SOP" in result.markdown
    assert "## Page 1" in result.markdown
    assert "Blood spill cleanup" in result.markdown
    assert result.pages[0].used_ocr is False
    assert result.chunks[0].page_number == 1


def test_ocr_fallback_is_triggered_for_image_only_page(tmp_path: Path, monkeypatch):
    content = make_pdf(None)
    monkeypatch.setattr(pdf_ingestion, "ocr_page", lambda page: "OCR extracted isolation room PPE.")

    result = ingest_pdf_bytes(
        filename="scan.pdf",
        content=content,
        title="Scanned SOP",
        category="ppe",
        source_name="Scanned PDF",
        knowledge_dir=tmp_path,
    )

    assert result.pages[0].used_ocr is True
    assert "OCR extracted isolation room PPE." in result.markdown


def test_chunking_preserves_page_section_and_page_number():
    markdown = pages_to_markdown(
        title="Floor SOP",
        category="floor scrubbing",
        source_name="Floor PDF",
        file_hash="abc123",
        raw_pdf_path=Path("knowledge_base/raw/floor.pdf"),
        uploaded_at="2026-05-20T00:00:00+00:00",
        pages=[
            pdf_ingestion.PageExtraction(
                page_number=2,
                text="Place wet floor signs before scrubbing. Recover dirty solution.",
                used_ocr=False,
            )
        ],
    )

    chunks = chunk_markdown(markdown)

    assert chunks[0].section_title == "Page 2"
    assert chunks[0].page_number == 2
    assert "wet floor signs" in chunks[0].chunk_text


def test_duplicate_pdf_hash_is_deterministic_and_flagged(tmp_path: Path):
    content = make_pdf("Duplicate SOP content with enough text to avoid OCR fallback.")

    first = ingest_pdf_bytes("dup.pdf", content, "Dup SOP", "general", "Dup PDF", tmp_path)
    second = ingest_pdf_bytes("dup.pdf", content, "Dup SOP", "general", "Dup PDF", tmp_path)

    assert sha256_bytes(content) == first.file_hash == second.file_hash
    assert first.raw_pdf_path == second.raw_pdf_path
    assert second.duplicate is True


def test_unsupported_non_pdf_upload_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Only valid PDF"):
        ingest_pdf_bytes(
            filename="not-a-pdf.txt",
            content=b"hello",
            title="Bad",
            category="general",
            source_name="Bad",
            knowledge_dir=tmp_path,
        )
