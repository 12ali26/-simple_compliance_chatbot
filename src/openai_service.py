from __future__ import annotations

import os
from dataclasses import dataclass


EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_model: str = EMBEDDING_MODEL


def get_openai_settings() -> OpenAISettings | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAISettings(
        api_key=api_key,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
    )


def get_client(settings: OpenAISettings | None = None):
    settings = settings or get_openai_settings()
    if settings is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed.") from exc
    return OpenAI(api_key=settings.api_key)


def embed_texts(texts: list[str], settings: OpenAISettings | None = None) -> list[list[float]]:
    if not texts:
        return []
    settings = settings or get_openai_settings()
    if settings is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    client = get_client(settings)
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def generate_grounded_answer(
    question: str,
    chunks: list[dict],
    settings: OpenAISettings | None = None,
) -> str:
    settings = settings or get_openai_settings()
    if settings is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    client = get_client(settings)

    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        page_label = f"Page {chunk.get('page_number')}" if chunk.get("page_number") else "Page unknown"
        context_parts.append(
            "\n".join(
                [
                    f"[{index}] {chunk.get('document_title', 'Unknown Document')} -> "
                    f"{chunk.get('section_title', 'Unknown Section')} -> {page_label}",
                    str(chunk.get("chunk_text", "")),
                ]
            )
        )

    system_prompt = (
        "You are the Cleaning SOP Assistant for janitorial and housekeeping staff. "
        "Answer only from the provided approved procedure chunks. "
        "Do not invent procedures, chemical names, PPE requirements, contact times, or dilution ratios. "
        "If the chunks do not contain the answer, say: "
        "\"I could not find this information in the approved procedures. Please consult your supervisor.\" "
        "Keep answers simple, practical, and concise. Include source citations using the exact labels provided."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        "Approved procedure chunks:\n\n"
        + "\n\n---\n\n".join(context_parts)
    )

    response = client.responses.create(
        model=settings.chat_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text.strip()
