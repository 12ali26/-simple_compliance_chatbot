from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from src.config import is_real_value
from src.openai_service import embed_texts as embed_openai_texts
from src.openai_service import get_openai_settings


DEFAULT_OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str


def get_openrouter_embedding_model() -> str:
    return os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_OPENROUTER_EMBEDDING_MODEL)


def has_openrouter_embeddings() -> bool:
    return is_real_value(os.getenv("OPENROUTER_API_KEY"))


def has_openai_embeddings() -> bool:
    return get_openai_settings() is not None


def get_embedding_settings() -> EmbeddingSettings | None:
    if has_openrouter_embeddings():
        return EmbeddingSettings(provider="openrouter", model=get_openrouter_embedding_model())
    openai_settings = get_openai_settings()
    if openai_settings:
        return EmbeddingSettings(provider="openai", model=openai_settings.embedding_model)
    return None


def embed_openrouter_texts(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not is_real_value(api_key):
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    if not texts:
        return []

    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": get_openrouter_embedding_model(),
            "input": texts,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or []
    return [item["embedding"] for item in data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if has_openrouter_embeddings():
        return embed_openrouter_texts(texts)

    openai_settings = get_openai_settings()
    if openai_settings:
        return embed_openai_texts(texts, settings=openai_settings)

    raise RuntimeError("No embedding provider is configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
