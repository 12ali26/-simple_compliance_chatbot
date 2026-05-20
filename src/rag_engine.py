from __future__ import annotations

from dataclasses import dataclass

from src.embedding_service import embed_texts, get_embedding_settings
from src.model_service import generate_grounded_answer
from src.sop_engine import FALLBACK_RESPONSE, AnswerResult, DocumentChunk, detect_category, retrieve, tokenize
from src.supabase_store import SupabaseStore


@dataclass(frozen=True)
class RetrievalConfig:
    match_count: int = 8
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
    settings = get_embedding_settings()
    if not store.enabled or settings is None:
        raise RuntimeError("Supabase and an embedding provider must both be configured for semantic answers.")

    config = config or RetrievalConfig()
    query_embedding = embed_texts([question])[0]
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

    answer = generate_grounded_answer(question, matches)
    if not answer:
        answer = FALLBACK_RESPONSE

    return AnswerResult(
        answer=answer,
        sources=[format_source(match) for match in matches],
        category=category,
        reason="Answered from Supabase semantic retrieval.",
    )


def local_chunk_to_model_context(chunk: DocumentChunk) -> dict:
    return {
        "document_title": chunk.document_title,
        "section_title": chunk.section_title,
        "source_name": chunk.source_name,
        "page_number": chunk.page_number,
        "chunk_text": chunk.chunk_text,
    }


def answer_question_local_model(
    question: str,
    chunks: list[DocumentChunk],
) -> AnswerResult:
    matches = retrieve(question, chunks)
    category = detect_category(set(tokenize(question)))

    if not matches:
        return AnswerResult(
            answer=FALLBACK_RESPONSE,
            sources=[],
            category=category,
            reason="No local SOP section matched the question.",
        )

    contexts = [local_chunk_to_model_context(match) for match in matches]
    answer = generate_grounded_answer(question, contexts)
    if not answer:
        answer = FALLBACK_RESPONSE

    return AnswerResult(
        answer=answer,
        sources=[format_source(context) for context in contexts],
        category=category,
        reason="Answered from local keyword retrieval and configured model provider.",
    )
