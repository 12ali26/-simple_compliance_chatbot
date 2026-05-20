from __future__ import annotations

from dataclasses import dataclass

from src.openai_service import embed_texts, generate_grounded_answer, get_openai_settings
from src.sop_engine import FALLBACK_RESPONSE, AnswerResult, detect_category, tokenize
from src.supabase_store import SupabaseStore


@dataclass(frozen=True)
class RetrievalConfig:
    match_count: int = 5
    similarity_threshold: float = 0.72


def format_source(chunk: dict) -> dict[str, str]:
    page = chunk.get("page_number")
    section = chunk.get("section_title") or "Unknown Section"
    return {
        "document_title": str(chunk.get("document_title") or chunk.get("title") or "Unknown Document"),
        "section_title": section,
        "source_name": str(chunk.get("source_name") or ""),
        "page_number": str(page or ""),
    }


def answer_question_semantic(
    question: str,
    store: SupabaseStore,
    config: RetrievalConfig | None = None,
) -> AnswerResult:
    settings = get_openai_settings()
    if not store.enabled or settings is None:
        raise RuntimeError("Supabase and OpenAI must both be configured for semantic answers.")

    config = config or RetrievalConfig()
    query_embedding = embed_texts([question], settings=settings)[0]
    matches = store.match_chunks(
        query_embedding=query_embedding,
        match_count=config.match_count,
        similarity_threshold=config.similarity_threshold,
    )
    category = detect_category(set(tokenize(question)))

    if not matches:
        return AnswerResult(
            answer=FALLBACK_RESPONSE,
            sources=[],
            category=category,
            reason="No Supabase vector match met the similarity threshold.",
        )

    answer = generate_grounded_answer(question, matches, settings=settings)
    if not answer:
        answer = FALLBACK_RESPONSE

    return AnswerResult(
        answer=answer,
        sources=[format_source(match) for match in matches],
        category=category,
        reason="Answered from Supabase semantic retrieval.",
    )
