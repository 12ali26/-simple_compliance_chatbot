from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


MIN_TEXT_CHARS_FOR_OCR = 40


@dataclass(frozen=True)
class PageExtraction:
    page_number: int
    text: str
    used_ocr: bool


@dataclass(frozen=True)
class IngestedDocument:
    title: str
    category: str
    source_name: str
    slug: str
    file_hash: str
    raw_pdf_path: Path
    markdown_path: Path
    markdown: str
    pages: list[PageExtraction]
    chunks: list["MarkdownChunk"]
    duplicate: bool = False


@dataclass(frozen=True)
class MarkdownChunk:
    chunk_text: str
    section_title: str
    page_number: int | None
    chunk_index: int
    token_count: int


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def slugify(value: str, fallback: str = "document") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def ensure_pdf(filename: str, content: bytes) -> None:
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF"):
        raise ValueError("Only valid PDF files can be uploaded.")


def make_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        subprocess.run(
            ["tesseract", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def ocr_page(page: object) -> str:
    if not tesseract_available():
        raise RuntimeError(
            "This page needs OCR, but the tesseract executable is not installed."
        )

    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("OCR dependencies are missing. Install Pillow and pytesseract.") from exc

    import fitz

    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image).strip()


def extract_pdf_pages(pdf_path: Path) -> list[PageExtraction]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF ingestion. Install pymupdf.") from exc

    pages: list[PageExtraction] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            used_ocr = False
            if len(text) < MIN_TEXT_CHARS_FOR_OCR:
                ocr_text = ocr_page(page)
                if ocr_text:
                    text = ocr_text
                    used_ocr = True
            pages.append(PageExtraction(page_number=index, text=normalize_text(text), used_ocr=used_ocr))
    return pages


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def frontmatter_value(value: str) -> str:
    return value.replace('"', '\\"')


def pages_to_markdown(
    title: str,
    category: str,
    source_name: str,
    file_hash: str,
    raw_pdf_path: Path,
    uploaded_at: str,
    pages: list[PageExtraction],
) -> str:
    parts = [
        "---",
        f'title: "{frontmatter_value(title)}"',
        f'category: "{frontmatter_value(category)}"',
        f'source_name: "{frontmatter_value(source_name)}"',
        f'file_hash: "{file_hash}"',
        f'raw_pdf: "{raw_pdf_path.as_posix()}"',
        f'uploaded_at: "{uploaded_at}"',
        "---",
        "",
        f"# {title}",
        "",
    ]

    for page in pages:
        parts.append(f"## Page {page.page_number}")
        parts.append("")
        if page.text:
            parts.append(page.text)
        else:
            parts.append("_No readable text was extracted from this page._")
        parts.append("")

    return "\n".join(parts).strip() + "\n"


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def split_oversized_text(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    words = re.findall(r"\S+", text)
    if len(words) <= target_tokens:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + target_tokens)
        chunks.append(" ".join(words[start:end]).strip())
        if end == len(words):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def chunk_markdown(
    markdown: str,
    target_tokens: int = 850,
    overlap_tokens: int = 120,
) -> list[MarkdownChunk]:
    page_pattern = re.compile(r"^##\s+Page\s+(\d+)\s*$", re.MULTILINE)
    matches = list(page_pattern.finditer(markdown))
    chunks: list[MarkdownChunk] = []

    for match_index, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(markdown)
        section_text = markdown[start:end].strip()
        if not section_text or section_text.startswith("_No readable text"):
            continue

        for piece in split_oversized_text(section_text, target_tokens, overlap_tokens):
            chunks.append(
                MarkdownChunk(
                    chunk_text=piece,
                    section_title=f"Page {page_number}",
                    page_number=page_number,
                    chunk_index=len(chunks),
                    token_count=estimate_tokens(piece),
                )
            )

    if chunks:
        return chunks

    body = re.sub(r"^---.*?---", "", markdown, count=1, flags=re.DOTALL).strip()
    body = re.sub(r"^#.*$", "", body, count=1, flags=re.MULTILINE).strip()
    for piece in split_oversized_text(body, target_tokens, overlap_tokens):
        if piece:
            chunks.append(
                MarkdownChunk(
                    chunk_text=piece,
                    section_title="General Procedure",
                    page_number=None,
                    chunk_index=len(chunks),
                    token_count=estimate_tokens(piece),
                )
            )
    return chunks


def ingest_pdf_bytes(
    filename: str,
    content: bytes,
    title: str,
    category: str,
    source_name: str,
    knowledge_dir: Path,
) -> IngestedDocument:
    ensure_pdf(filename, content)
    file_hash = sha256_bytes(content)
    clean_title = title.strip() or Path(filename).stem.replace("_", " ").replace("-", " ").title()
    clean_category = category.strip() or "general"
    clean_source_name = source_name.strip() or filename
    slug = f"{slugify(clean_title)}-{file_hash[:10]}"

    raw_dir = knowledge_dir / "raw"
    markdown_dir = knowledge_dir / "markdown"
    raw_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    raw_pdf_path = raw_dir / f"{slug}.pdf"
    markdown_path = markdown_dir / f"{slug}.md"
    duplicate = raw_pdf_path.exists() or markdown_path.exists()
    raw_pdf_path.write_bytes(content)

    uploaded_at = datetime.now(timezone.utc).isoformat()
    pages = extract_pdf_pages(raw_pdf_path)
    markdown = pages_to_markdown(
        clean_title,
        clean_category,
        clean_source_name,
        file_hash,
        raw_pdf_path,
        uploaded_at,
        pages,
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    chunks = chunk_markdown(markdown)

    return IngestedDocument(
        title=clean_title,
        category=clean_category,
        source_name=clean_source_name,
        slug=slug,
        file_hash=file_hash,
        raw_pdf_path=raw_pdf_path,
        markdown_path=markdown_path,
        markdown=markdown,
        pages=pages,
        chunks=chunks,
        duplicate=duplicate,
    )
