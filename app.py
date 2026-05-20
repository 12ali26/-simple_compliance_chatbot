from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.config import get_config_status, load_dotenv_file, sync_mapping_to_environ
from src.logging_store import append_chat_log, append_unanswered_question, update_helpfulness
from src.openai_service import embed_texts, get_openai_settings
from src.pdf_ingestion import ingest_pdf_bytes, sha256_bytes
from src.rag_engine import answer_question_semantic
from src.sop_engine import FALLBACK_RESPONSE, answer_question, load_knowledge_base
from src.supabase_store import SupabaseStore


ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT / "knowledge_base"
DATA_DIR = ROOT / "data"


st.set_page_config(
    page_title="Cleaning SOP Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding: 1rem 0.85rem 5rem;
    }
    h1 {
        font-size: clamp(1.8rem, 6vw, 2.45rem);
        letter-spacing: 0;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #4b5563;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        border-radius: 8px;
    }
    .source-box {
        border-left: 4px solid #0f766e;
        background: #f0fdfa;
        padding: 0.75rem 0.85rem;
        border-radius: 6px;
        margin-top: 0.65rem;
        color: #134e4a;
        font-size: 0.95rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    div[data-testid="stButton"] button {
        min-height: 2.65rem;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def sync_streamlit_secrets() -> None:
    try:
        sync_mapping_to_environ(st.secrets)
    except Exception:
        return


def status_label(value: bool) -> str:
    return "configured" if value else "not configured"


def render_connection_status(store: SupabaseStore) -> None:
    status = get_config_status()
    st.subheader("Connection Status")
    st.write(f"Admin password: {status_label(status.admin_password)}")
    st.write(f"Supabase credentials: {status_label(status.supabase)}")
    st.write(f"Supabase client: {'connected' if store.enabled else 'not connected'}")
    st.write(f"OpenAI API key: {status_label(status.openai)}")
    st.write(f"Semantic search: {'active' if status.semantic_search and store.enabled else 'inactive'}")


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me about approved cleaning procedures, PPE, spills, chemicals, waste, or room cleaning.",
                "sources": [],
                "log_id": None,
            }
        ]
    if "last_log_id" not in st.session_state:
        st.session_state.last_log_id = None
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    source_lines = []
    seen = set()
    for source in sources:
        section_title = source["section_title"]
        page_number = source.get("page_number")
        page_suffix = ""
        if page_number and f"Page {page_number}" not in section_title:
            page_suffix = f" -> Page {page_number}"
        label = f"{source['document_title']} -> {section_title}{page_suffix}"
        if label not in seen:
            seen.add(label)
            source_lines.append(label)

    st.markdown(
        "<div class='source-box'><strong>Source:</strong><br>"
        + "<br>".join(source_lines)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_feedback(log_id: str | None, store: SupabaseStore) -> None:
    if not log_id:
        return

    st.caption("Was this answer helpful?")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Helpful", use_container_width=True, key=f"helpful-{log_id}"):
            if store.enabled:
                store.update_helpfulness(log_id, True)
            else:
                update_helpfulness(DATA_DIR, log_id, True)
            st.toast("Feedback saved.")
    with col_no:
        if st.button("Not helpful", use_container_width=True, key=f"not-helpful-{log_id}"):
            if store.enabled:
                store.update_helpfulness(log_id, False)
            else:
                update_helpfulness(DATA_DIR, log_id, False)
            st.toast("Thanks. This will be flagged for review.")


def source_string(sources: list[dict]) -> str:
    labels = []
    for source in sources:
        section_title = source["section_title"]
        page_number = source.get("page_number")
        page_suffix = ""
        if page_number and f"Page {page_number}" not in section_title:
            page_suffix = f" -> Page {page_number}"
        labels.append(f"{source['document_title']} -> {section_title}{page_suffix}")
    return "; ".join(labels)


def save_chat_log(store: SupabaseStore, row: dict) -> str:
    if store.enabled:
        supabase_id = store.append_chat_log(
            {
                "question": row["question"],
                "answer": row["answer"],
                "source_used": row["source_used"],
                "helpful": None,
                "created_at": row["created_at"],
            }
        )
        return supabase_id or row["id"]

    append_chat_log(DATA_DIR, row)
    return row["id"]


def save_unanswered(store: SupabaseStore, row: dict) -> None:
    if store.enabled:
        store.append_unanswered_question(
            {
                "question": row["question"],
                "reason": row["reason"],
                "created_at": row["created_at"],
            }
        )
    else:
        append_unanswered_question(DATA_DIR, row)


def render_chat_tab(chunks, store: SupabaseStore) -> None:
    semantic_enabled = store.enabled and get_openai_settings() is not None

    st.title("Cleaning SOP Assistant")
    st.markdown(
        "<div class='subtitle'>Fast answers from approved cleaning procedures.</div>",
        unsafe_allow_html=True,
    )

    if not semantic_enabled and not chunks:
        st.error("No SOP documents were found. Upload a PDF or add Markdown files to knowledge_base/markdown.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    if prompt := st.chat_input("Ask a cleaning procedure question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking approved procedures..."):
                if semantic_enabled:
                    response = answer_question_semantic(prompt, store)
                else:
                    response = answer_question(prompt, chunks)

            st.markdown(response.answer)
            render_sources(response.sources)

            created_at = datetime.now(timezone.utc).isoformat()
            row = {
                "id": str(uuid4()),
                "question": prompt,
                "answer": response.answer,
                "category": response.category,
                "source_used": source_string(response.sources),
                "helpful": "",
                "created_at": created_at,
            }
            log_id = save_chat_log(store, row)

            if response.answer == FALLBACK_RESPONSE:
                save_unanswered(
                    store,
                    {
                        "id": str(uuid4()),
                        "question": prompt,
                        "reason": response.reason,
                        "created_at": created_at,
                    },
                )

            st.session_state.last_log_id = log_id
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "log_id": log_id,
                }
            )
            render_feedback(log_id, store)


def render_admin_tab(store: SupabaseStore) -> None:
    st.title("Admin Upload")
    st.markdown(
        "<div class='subtitle'>Upload approved SOP PDFs and publish them to the searchable knowledge base.</div>",
        unsafe_allow_html=True,
    )
    render_connection_status(store)

    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password:
        st.warning("ADMIN_PASSWORD is not configured. Set it in Streamlit secrets or the environment.")
        return

    if not st.session_state.admin_authenticated:
        entered = st.text_input("Admin password", type="password")
        if st.button("Unlock admin", use_container_width=True):
            st.session_state.admin_authenticated = entered == admin_password
            if not st.session_state.admin_authenticated:
                st.error("Incorrect admin password.")
            else:
                st.rerun()
        return

    st.success("Admin unlocked.")
    uploaded_pdf = st.file_uploader("Upload approved SOP PDF", type=["pdf"])
    title = st.text_input("Document title")
    category = st.text_input("Category", placeholder="spill response")
    source_name = st.text_input("Source name", placeholder="Approved SOP name or document code")

    if not uploaded_pdf:
        return

    content = uploaded_pdf.getvalue()
    file_hash = sha256_bytes(content)
    existing = store.document_by_hash(file_hash) if store.enabled else None
    if existing:
        st.info(f"This PDF is already published as {existing.get('title', 'an existing document')}.")
        return

    if st.button("Convert and publish", type="primary", use_container_width=True):
        with st.spinner("Saving PDF, converting to Markdown, chunking, and publishing..."):
            try:
                document = ingest_pdf_bytes(
                    filename=uploaded_pdf.name,
                    content=content,
                    title=title,
                    category=category,
                    source_name=source_name,
                    knowledge_dir=KNOWLEDGE_DIR,
                )

                supabase_status = "Skipped: Supabase/OpenAI not configured"
                if store.enabled and get_openai_settings() is not None:
                    document_row = store.create_document(document)
                    embeddings = embed_texts([chunk.chunk_text for chunk in document.chunks])
                    chunk_count = store.replace_document_chunks(
                        document_id=str(document_row["id"]),
                        chunks=document.chunks,
                        embeddings=embeddings,
                    )
                    supabase_status = f"Published {chunk_count} chunks"

                st.success("PDF ingested.")
                if document.duplicate:
                    st.info("A local file with the same hash already existed and was refreshed.")
                st.write(f"Pages processed: {len(document.pages)}")
                st.write(f"OCR pages: {sum(1 for page in document.pages if page.used_ocr)}")
                st.write(f"Chunks created: {len(document.chunks)}")
                st.write(f"Raw PDF: `{document.raw_pdf_path}`")
                st.write(f"Markdown: `{document.markdown_path}`")
                st.write(f"Supabase: {supabase_status}")
            except Exception as exc:
                st.error(str(exc))


load_dotenv_file(ROOT)
sync_streamlit_secrets()
init_state()
store = SupabaseStore()
chunks = load_knowledge_base(KNOWLEDGE_DIR)

chat_tab, admin_tab = st.tabs(["Chat", "Admin"])

with chat_tab:
    render_chat_tab(chunks, store)

with admin_tab:
    render_admin_tab(store)

with st.sidebar:
    status = get_config_status()
    semantic_enabled = store.enabled and get_openai_settings() is not None
    st.header("Supervisor Snapshot")
    st.markdown(
        "<p class='small-note'>Logs use Supabase when configured, otherwise local CSV files.</p>",
        unsafe_allow_html=True,
    )
    st.write(f"Loaded local SOP sections: {len(chunks)}")
    st.write(f"Supabase: {'connected' if store.enabled else status_label(status.supabase)}")
    st.write(f"OpenAI: {status_label(status.openai)}")
    st.write(f"Retrieval: {'semantic vector search' if semantic_enabled else 'local keyword fallback'}")
