from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FALLBACK_RESPONSE = (
    "I could not find this information in the approved procedures. "
    "Please consult your supervisor."
)

STOP_WORDS = {
    "a",
    "about",
    "after",
    "am",
    "an",
    "and",
    "annual",
    "apply",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "clean",
    "cleaning",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "leave",
    "me",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "use",
    "what",
    "when",
    "which",
    "with",
}

CATEGORY_TERMS = {
    "ppe": {"ppe", "gloves", "gown", "mask", "goggles", "visor", "apron"},
    "spill response": {"spill", "blood", "vomit", "body", "fluid", "absorbent"},
    "washroom cleaning": {"washroom", "toilet", "urinal", "sink", "basin"},
    "floor scrubbing": {"floor", "scrub", "scrubber", "mop", "wet"},
    "waste handling": {"waste", "bag", "bin", "sharps", "trash", "rubbish"},
    "chemical dilution": {"chemical", "dilution", "dilute", "ratio", "label"},
    "isolation room cleaning": {"isolation", "room", "discharge", "terminal"},
}

SYNONYMS = {
    "restroom": "washroom",
    "bathroom": "washroom",
    "loo": "washroom",
    "lavatory": "washroom",
    "toilets": "toilet",
    "chemicals": "chemical",
    "dilute": "dilution",
    "diluted": "dilution",
    "clean": "cleaning",
    "cleans": "cleaning",
    "cleaned": "cleaning",
    "scrubbing": "scrub",
    "scrubber": "scrub",
    "garbage": "waste",
    "trash": "waste",
    "rubbish": "waste",
    "bodily": "body",
    "fluids": "fluid",
}


@dataclass(frozen=True)
class DocumentChunk:
    document_title: str
    category: str
    source_name: str
    section_title: str
    chunk_text: str
    file_path: str
    page_number: int | None = None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[dict[str, str]]
    category: str
    reason: str


def tokenize(text: str) -> list[str]:
    raw_terms = re.findall(r"[a-z0-9]+", text.lower())
    terms = []
    for term in raw_terms:
        normalized = SYNONYMS.get(term, term)
        if normalized not in STOP_WORDS and len(normalized) > 1:
            terms.append(normalized)
    return terms


def parse_markdown_document(path: Path) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").title()
    category = "general"
    source_name = path.name

    frontmatter = re.search(r"^---\n(.*?)\n---", text, re.DOTALL)
    if frontmatter:
        fields = dict(
            re.findall(r"^([a-zA-Z_]+):\s*\"?(.+?)\"?\s*$", frontmatter.group(1), re.MULTILINE)
        )
        title = fields.get("title", title)
        category = fields.get("category", category)
        source_name = fields.get("source_name", source_name)

    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    category_match = re.search(r"^Category:\s*(.+)$", text, re.MULTILINE)
    if category_match:
        category = category_match.group(1).strip()

    source_match = re.search(r"^Source:\s*(.+)$", text, re.MULTILINE)
    if source_match:
        source_name = source_match.group(1).strip()

    section_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(section_pattern.finditer(text))
    chunks = []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_title = match.group(1).strip()
        page_match = re.match(r"Page\s+(\d+)", section_title, re.IGNORECASE)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    document_title=title,
                    category=category,
                    source_name=source_name,
                    section_title=section_title,
                    chunk_text=chunk_text,
                    file_path=str(path),
                    page_number=int(page_match.group(1)) if page_match else None,
                )
            )

    if not chunks:
        body = re.sub(r"^#.*$", "", text, count=1, flags=re.MULTILINE).strip()
        if body:
            chunks.append(
                DocumentChunk(
                    document_title=title,
                    category=category,
                    source_name=source_name,
                    section_title="General Procedure",
                    chunk_text=body,
                    file_path=str(path),
                    page_number=None,
                )
            )

    return chunks


def load_knowledge_base(knowledge_dir: Path) -> list[DocumentChunk]:
    if not knowledge_dir.exists():
        return []

    search_dir = knowledge_dir / "markdown" if (knowledge_dir / "markdown").exists() else knowledge_dir
    chunks: list[DocumentChunk] = []
    for path in sorted(search_dir.glob("*.md")):
        chunks.extend(parse_markdown_document(path))
    return chunks


def detect_category(question_terms: set[str]) -> str:
    best_category = "general"
    best_hits = 0
    for category, terms in CATEGORY_TERMS.items():
        hits = len(question_terms & terms)
        if hits > best_hits:
            best_category = category
            best_hits = hits
    return best_category


def score_chunk(question_terms: set[str], chunk: DocumentChunk) -> int:
    searchable = " ".join(
        [chunk.document_title, chunk.category, chunk.source_name, chunk.section_title, chunk.chunk_text]
    )
    chunk_terms = set(tokenize(searchable))
    overlap = question_terms & chunk_terms
    if not overlap:
        return 0

    title_terms = set(tokenize(chunk.document_title + " " + chunk.section_title + " " + chunk.category))
    weighted_overlap = len(overlap) * 2 + len(overlap & title_terms) * 3
    return weighted_overlap


def retrieve(question: str, chunks: list[DocumentChunk], limit: int = 2) -> list[DocumentChunk]:
    question_terms = set(tokenize(question))
    if not question_terms:
        return []

    scored = [
        (score_chunk(question_terms, chunk), chunk)
        for chunk in chunks
    ]
    relevant = [(score, chunk) for score, chunk in scored if score >= 2]
    relevant.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in relevant[:limit]]


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*]\s+", "", line)
    line = re.sub(r"^\d+\.\s+", "", line)
    return line.strip()


def select_evidence_lines(question: str, chunks: list[DocumentChunk], max_lines: int = 5) -> list[str]:
    question_terms = set(tokenize(question))
    selected: list[str] = []

    for chunk in chunks:
        lines = [clean_line(line) for line in chunk.chunk_text.splitlines()]
        lines = [line for line in lines if line and not line.startswith("#")]
        ranked = sorted(
            lines,
            key=lambda line: len(question_terms & set(tokenize(line))),
            reverse=True,
        )
        for line in ranked:
            if len(question_terms & set(tokenize(line))) > 0 and line not in selected:
                selected.append(line)
            if len(selected) >= max_lines:
                return selected

    if not selected:
        for chunk in chunks:
            for line in [clean_line(line) for line in chunk.chunk_text.splitlines()]:
                if line and line not in selected:
                    selected.append(line)
                if len(selected) >= max_lines:
                    return selected

    return selected


def build_grounded_answer(question: str, chunks: list[DocumentChunk]) -> str:
    lines = select_evidence_lines(question, chunks)
    if not lines:
        return FALLBACK_RESPONSE

    intro = "According to the approved procedure:"
    bullets = "\n".join(f"- {line}" for line in lines)
    escalation = (
        "\n\nIf the situation is unclear, stop and consult your supervisor before continuing."
    )
    return f"{intro}\n\n{bullets}{escalation}"


def answer_question(question: str, chunks: list[DocumentChunk]) -> AnswerResult:
    question_terms = set(tokenize(question))
    category = detect_category(question_terms)
    matches = retrieve(question, chunks)

    if not matches:
        return AnswerResult(
            answer=FALLBACK_RESPONSE,
            sources=[],
            category=category,
            reason="No approved SOP section matched the question.",
        )

    sources = [
        {
            "document_title": chunk.document_title,
            "section_title": chunk.section_title,
            "source_name": chunk.source_name,
            "page_number": str(chunk.page_number or ""),
        }
        for chunk in matches
    ]

    return AnswerResult(
        answer=build_grounded_answer(question, matches),
        sources=sources,
        category=category,
        reason="Answered from matched approved SOP sections.",
    )
