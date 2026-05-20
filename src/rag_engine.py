from __future__ import annotations

import re
from dataclasses import dataclass

from src.embedding_service import embed_texts, get_embedding_settings
from src.model_service import generate_grounded_answer
from src.sop_engine import FALLBACK_RESPONSE, AnswerResult, DocumentChunk, answer_question, detect_category, retrieve, tokenize
from src.supabase_store import SupabaseStore


@dataclass(frozen=True)
class RetrievalConfig:
    match_count: int = 8
    similarity_threshold: float = 0.35


@dataclass(frozen=True)
class EvidenceChunk:
    document_title: str
    section_title: str
    source_name: str
    page_number: int | None
    chunk_text: str
    score: int
    source_type: str


QUERY_REPLACEMENTS = [
    (r"\bi\s+solation\b", "isolation"),
    (r"\bppes\b", "ppe"),
    (r"\bdowning\b", "donning"),
    (r"\bdoff\b", "doffing"),
    (r"\bdon\b", "donning"),
    (r"\brespirator\b", "n95 mask"),
    (r"\bface\s*mask\b", "mask"),
]


PPE_ISOLATION_TERMS = {
    "airborne",
    "contact",
    "droplet",
    "isolation",
    "mask",
    "n95",
    "ppe",
    "protective",
    "respirator",
    "surgical",
}


def normalize_query(question: str) -> str:
    normalized = question.lower().replace("’", "'")
    for pattern, replacement in QUERY_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bn\s*95\b", "n95", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bn95\s+n95 mask\b", "n95 mask", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_evidence_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_ppe_isolation_question(question: str) -> bool:
    terms = set(re.findall(r"[a-z0-9]+", normalize_query(question)))
    return bool(terms & PPE_ISOLATION_TERMS)


def format_source(chunk: dict) -> dict[str, str]:
    page = chunk.get("page_number")
    section = chunk.get("section_title") or "Unknown Section"
    return {
        "document_title": str(chunk.get("document_title") or chunk.get("title") or "Unknown Document"),
        "section_title": section,
        "source_name": str(chunk.get("source_name") or ""),
        "page_number": str(page or ""),
    }


def evidence_to_source(evidence: EvidenceChunk) -> dict[str, str]:
    return {
        "document_title": evidence.document_title,
        "section_title": evidence.section_title,
        "source_name": evidence.source_name,
        "page_number": str(evidence.page_number or ""),
    }


def local_chunk_to_evidence(chunk: DocumentChunk, score: int = 0, source_type: str = "local") -> EvidenceChunk:
    return EvidenceChunk(
        document_title=chunk.document_title,
        section_title=chunk.section_title,
        source_name=chunk.source_name,
        page_number=chunk.page_number,
        chunk_text=chunk.chunk_text,
        score=score,
        source_type=source_type,
    )


def semantic_match_to_evidence(match: dict, score: int = 0) -> EvidenceChunk:
    page_number = match.get("page_number")
    return EvidenceChunk(
        document_title=str(match.get("document_title") or match.get("title") or "Unknown Document"),
        section_title=str(match.get("section_title") or "Unknown Section"),
        source_name=str(match.get("source_name") or ""),
        page_number=int(page_number) if page_number else None,
        chunk_text=str(match.get("chunk_text") or ""),
        score=score,
        source_type="semantic",
    )


def evidence_key(evidence: EvidenceChunk) -> tuple[str, str, int | None]:
    return (evidence.document_title, evidence.section_title, evidence.page_number)


def score_evidence(question: str, evidence: EvidenceChunk) -> int:
    normalized_question = normalize_query(question)
    question_terms = set(tokenize(normalized_question))
    text = normalize_evidence_text(
        " ".join(
            [
                evidence.document_title,
                evidence.section_title,
                evidence.source_name,
                evidence.chunk_text,
            ]
        )
    )
    text_terms = set(tokenize(text))

    score = len(question_terms & text_terms) * 4
    if evidence.source_type == "semantic":
        score += 3

    if is_ppe_isolation_question(normalized_question):
        has_n95 = "n95 mask" in text or "n95" in text
        has_airborne = "airborne isolation" in text
        has_surgical = "surgical mask" in text
        has_droplet = "droplet isolation" in text
        has_protective = "protective environment" in text
        has_contact = "contact isolation" in text
        has_ppe_table = "required personal protective equipment" in text
        has_terminal_isolation = "isolation room terminal cleaning" in text

        if has_ppe_table:
            score += 45
        if has_terminal_isolation:
            score += 25
        if has_n95:
            score += 25
        if has_airborne:
            score += 35
        if has_n95 and has_airborne:
            score += 60
        if has_surgical:
            score += 15
        if has_droplet:
            score += 20
        if has_protective:
            score += 15
        if "donning & doffing" in text or "donning and doffing" in text:
            score -= 30

        if "contact" in normalized_question and has_contact:
            score += 35
        if "airborne" in normalized_question and has_airborne:
            score += 35
        if "droplet" in normalized_question and has_droplet:
            score += 35
        if "protective" in normalized_question and has_protective:
            score += 25

    return score


def collect_semantic_evidence(question: str, store: SupabaseStore, semantic_enabled: bool) -> list[EvidenceChunk]:
    if not semantic_enabled:
        return []
    try:
        query_embedding = embed_texts([question])[0]
        matches = store.match_chunks(
            query_embedding=query_embedding,
            match_count=12,
            similarity_threshold=0.0,
        )
    except Exception:
        return []
    return [semantic_match_to_evidence(match) for match in matches]


def collect_local_evidence(question: str, chunks: list[DocumentChunk]) -> list[EvidenceChunk]:
    normalized_question = normalize_query(question)
    candidates = [local_chunk_to_evidence(chunk) for chunk in retrieve(normalized_question, chunks, limit=20)]

    if is_ppe_isolation_question(normalized_question):
        for chunk in chunks:
            text = normalize_evidence_text(chunk.chunk_text)
            if any(
                phrase in text
                for phrase in [
                    "required personal protective equipment",
                    "n95 mask",
                    "surgical mask",
                    "airborne isolation",
                    "droplet isolation",
                    "protective environment",
                    "contact isolation",
                    "isolation room terminal cleaning",
                ]
            ):
                candidates.append(local_chunk_to_evidence(chunk))

    return candidates


def retrieve_best_evidence(
    question: str,
    chunks: list[DocumentChunk],
    store: SupabaseStore,
    semantic_enabled: bool,
    limit: int = 4,
) -> list[EvidenceChunk]:
    normalized_question = normalize_query(question)
    candidates = collect_local_evidence(normalized_question, chunks)
    candidates.extend(collect_semantic_evidence(normalized_question, store, semantic_enabled))

    best_by_key: dict[tuple[str, str, int | None], EvidenceChunk] = {}
    for candidate in candidates:
        score = score_evidence(normalized_question, candidate)
        scored = EvidenceChunk(
            document_title=candidate.document_title,
            section_title=candidate.section_title,
            source_name=candidate.source_name,
            page_number=candidate.page_number,
            chunk_text=candidate.chunk_text,
            score=score,
            source_type=candidate.source_type,
        )
        key = evidence_key(scored)
        existing = best_by_key.get(key)
        if existing is None or scored.score > existing.score:
            best_by_key[key] = scored

    evidence = sorted(best_by_key.values(), key=lambda chunk: chunk.score, reverse=True)
    return [chunk for chunk in evidence if chunk.score > 0][:limit]


def build_direct_ppe_isolation_answer(question: str, evidence: list[EvidenceChunk]) -> str | None:
    normalized_question = normalize_query(question)
    if not is_ppe_isolation_question(normalized_question):
        return None

    combined = normalize_evidence_text(" ".join(chunk.chunk_text for chunk in evidence))
    has_n95_airborne = "n95 mask" in combined and "airborne isolation" in combined
    has_surgical_droplet = "surgical mask" in combined and "droplet isolation" in combined
    has_protective = "protective environment" in combined

    if "contact" in normalized_question and ("n95" in normalized_question or "mask" in normalized_question):
        if has_n95_airborne:
            answer = (
                "The approved PPE table lists N95 Mask for Airborne Isolation. "
                "It does not list N95 as the Contact Isolation requirement."
            )
            if has_surgical_droplet:
                answer += " It lists Surgical Mask for Droplet Isolation"
                if has_protective:
                    answer += " and Protective Environment"
                answer += "."
            return answer

    if "n95" in normalized_question or "respirator" in normalized_question:
        if has_n95_airborne:
            return "N95 Mask is for Airborne Isolation."

    if "surgical" in normalized_question or "droplet" in normalized_question or "protective" in normalized_question:
        if has_surgical_droplet:
            answer = "Surgical Mask is for Droplet Isolation"
            if has_protective:
                answer += " and Protective Environment"
            answer += "."
            return answer

    if "airborne" in normalized_question and "mask" in normalized_question and has_n95_airborne:
        return "N95 Mask is for Airborne Isolation."

    return None


def direct_answer_evidence(question: str, evidence: list[EvidenceChunk]) -> list[EvidenceChunk]:
    normalized_question = normalize_query(question)
    selected = []
    for chunk in evidence:
        text = normalize_evidence_text(chunk.chunk_text)
        if "contact" in normalized_question and "contact isolation" in text:
            selected.append(chunk)
        elif "n95" in normalized_question and "n95 mask" in text and "airborne isolation" in text:
            selected.append(chunk)
        elif "airborne" in normalized_question and "n95 mask" in text and "airborne isolation" in text:
            selected.append(chunk)
        elif (
            ("droplet" in normalized_question or "surgical" in normalized_question or "protective" in normalized_question)
            and "surgical mask" in text
            and "droplet isolation" in text
        ):
            selected.append(chunk)

    return selected[:2] or evidence[:2]


def evidence_to_model_context(evidence: EvidenceChunk) -> dict:
    return {
        "document_title": evidence.document_title,
        "section_title": evidence.section_title,
        "source_name": evidence.source_name,
        "page_number": evidence.page_number,
        "chunk_text": evidence.chunk_text,
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


def answer_question_best_available(
    question: str,
    chunks: list[DocumentChunk],
    store: SupabaseStore,
    semantic_enabled: bool,
    model_enabled: bool,
) -> AnswerResult:
    normalized_question = normalize_query(question)
    category = detect_category(set(tokenize(normalized_question)))
    evidence = retrieve_best_evidence(normalized_question, chunks, store, semantic_enabled)

    if evidence:
        direct_answer = build_direct_ppe_isolation_answer(normalized_question, evidence)
        if direct_answer:
            direct_evidence = direct_answer_evidence(normalized_question, evidence)
            return AnswerResult(
                answer=direct_answer,
                sources=[evidence_to_source(chunk) for chunk in direct_evidence],
                category=category,
                reason="Answered directly from ranked PPE/isolation evidence.",
            )

    if evidence and model_enabled:
        try:
            contexts = [evidence_to_model_context(chunk) for chunk in evidence]
            answer = generate_grounded_answer(normalized_question, contexts)
            if answer and answer != FALLBACK_RESPONSE:
                return AnswerResult(
                    answer=answer,
                    sources=[evidence_to_source(chunk) for chunk in evidence],
                    category=category,
                    reason="Answered from ranked hybrid evidence and configured model provider.",
                )
        except Exception:
            pass

    if evidence:
        sources = [evidence_to_source(chunk) for chunk in evidence]
        local_chunks = [
            DocumentChunk(
                document_title=chunk.document_title,
                category=category,
                source_name=chunk.source_name,
                section_title=chunk.section_title,
                chunk_text=chunk.chunk_text,
                file_path="",
                page_number=chunk.page_number,
            )
            for chunk in evidence
        ]
        local_response = answer_question(normalized_question, local_chunks)
        return AnswerResult(
            answer=local_response.answer,
            sources=sources if local_response.answer != FALLBACK_RESPONSE else [],
            category=category,
            reason="Answered from ranked hybrid evidence without model provider.",
        )

    return AnswerResult(
        answer=FALLBACK_RESPONSE,
        sources=[],
        category=category,
        reason="No approved SOP section matched the question.",
    )
