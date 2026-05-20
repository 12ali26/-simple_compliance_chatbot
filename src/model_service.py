from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from src.config import is_real_value
from src.openai_service import generate_openai_grounded_answer, get_openai_settings


DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    model: str = DEFAULT_OPENROUTER_MODEL
    site_url: str | None = None
    app_name: str | None = None


def get_openrouter_settings() -> OpenRouterSettings | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not is_real_value(api_key):
        return None
    return OpenRouterSettings(
        api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        site_url=os.getenv("OPENROUTER_SITE_URL") or None,
        app_name=os.getenv("OPENROUTER_APP_NAME") or "Cleaning SOP Assistant",
    )


def build_grounded_messages(question: str, chunks: list[dict]) -> list[dict[str, str]]:
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
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_openrouter_grounded_answer(
    question: str,
    chunks: list[dict],
    settings: OpenRouterSettings | None = None,
) -> str:
    settings = settings or get_openrouter_settings()
    if settings is None:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    if settings.site_url:
        headers["HTTP-Referer"] = settings.site_url
    if settings.app_name:
        headers["X-Title"] = settings.app_name

    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers=headers,
        json={
            "model": settings.model,
            "messages": build_grounded_messages(question, chunks),
            "temperature": 0.1,
            "max_tokens": 700,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content", "")).strip()


def generate_grounded_answer(question: str, chunks: list[dict]) -> str:
    openrouter_settings = get_openrouter_settings()
    if openrouter_settings:
        return generate_openrouter_grounded_answer(question, chunks, openrouter_settings)

    openai_settings = get_openai_settings()
    if openai_settings:
        return generate_openai_grounded_answer(question, chunks, openai_settings)

    raise RuntimeError("No model provider is configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
