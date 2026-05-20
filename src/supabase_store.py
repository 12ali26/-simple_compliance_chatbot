from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pdf_ingestion import IngestedDocument, MarkdownChunk
from src.config import is_real_value


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    backend_key: str


def get_supabase_settings() -> SupabaseSettings | None:
    url = os.getenv("SUPABASE_URL")
    backend_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if (
        not is_real_value(url)
        or not is_real_value(backend_key)
        or str(backend_key).strip().lower().startswith("sb_publishable_")
    ):
        return None
    return SupabaseSettings(url=url, backend_key=backend_key)


def get_client(settings: SupabaseSettings | None = None):
    settings = settings or get_supabase_settings()
    if settings is None:
        raise RuntimeError("Supabase is not configured.")
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError("The supabase package is not installed.") from exc
    return create_client(settings.url, settings.backend_key)


class SupabaseStore:
    def __init__(self, settings: SupabaseSettings | None = None):
        self.settings = settings or get_supabase_settings()
        self.client = get_client(self.settings) if self.settings else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def document_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        response = (
            self.client.table("documents")
            .select("*")
            .eq("file_hash", file_hash)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def create_document(self, document: IngestedDocument) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Supabase is not configured.")

        existing = self.document_by_hash(document.file_hash)
        if existing:
            return existing

        payload = {
            "title": document.title,
            "category": document.category,
            "source_name": document.source_name,
            "raw_pdf_path": document.raw_pdf_path.as_posix(),
            "markdown_path": document.markdown_path.as_posix(),
            "file_hash": document.file_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        response = self.client.table("documents").insert(payload).execute()
        if not response.data:
            raise RuntimeError("Supabase did not return the inserted document.")
        return response.data[0]

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: list[MarkdownChunk],
        embeddings: list[list[float]],
    ) -> int:
        if not self.enabled:
            raise RuntimeError("Supabase is not configured.")
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")

        self.client.table("document_chunks").delete().eq("document_id", document_id).execute()
        if not chunks:
            return 0

        payloads = []
        for chunk, embedding in zip(chunks, embeddings):
            payloads.append(
                {
                    "document_id": document_id,
                    "chunk_text": chunk.chunk_text,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "embedding": embedding,
                }
            )
        self.client.table("document_chunks").insert(payloads).execute()
        return len(payloads)

    def match_chunks(
        self,
        query_embedding: list[float],
        match_count: int = 5,
        similarity_threshold: float = 0.72,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        response = self.client.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "similarity_threshold": similarity_threshold,
            },
        ).execute()
        return response.data or []

    def append_chat_log(self, row: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None
        response = self.client.table("chat_logs").insert(row).execute()
        if response.data:
            return str(response.data[0].get("id", ""))
        return None

    def append_unanswered_question(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.client.table("unanswered_questions").insert(row).execute()

    def update_helpfulness(self, log_id: str, helpful: bool) -> None:
        if not self.enabled:
            return
        self.client.table("chat_logs").update({"helpful": helpful}).eq("id", log_id).execute()


def path_to_str(path: Path | str) -> str:
    return path.as_posix() if isinstance(path, Path) else path
