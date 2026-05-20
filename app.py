from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.config import get_config_status, is_real_value, load_dotenv_file, sync_mapping_to_environ
from src.embedding_service import embed_texts, get_embedding_settings
from src.intent_router import Intent, route_message
from src.logging_store import (
    append_chat_log,
    append_unanswered_question,
    delete_chat_log,
    delete_unanswered_question,
    list_chat_logs,
    list_unanswered_questions,
    update_helpfulness,
)
from src.pdf_ingestion import ingest_pdf_bytes, sha256_bytes
from src.rag_engine import answer_question_best_available
from src.sop_engine import FALLBACK_RESPONSE, load_knowledge_base
from src.supabase_store import SupabaseStore


ROOT = Path(__file__).parent
KNOWLEDGE_DIR = ROOT / "knowledge_base"
DATA_DIR = ROOT / "data"


class SimpleResponse:
    def __init__(self, answer: str, category: str, reason: str):
        self.answer = answer
        self.sources = []
        self.category = category
        self.reason = reason


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


def current_role() -> str | None:
    return st.session_state.get("auth_role")


def is_admin() -> bool:
    return current_role() == "admin"


def configured_password(role: str) -> str | None:
    key = "ADMIN_PASSWORD" if role == "admin" else "STAFF_PASSWORD"
    value = os.getenv(key)
    return value if is_real_value(value) else None


def render_connection_status(store: SupabaseStore) -> None:
    status = get_config_status()
    embedding_settings = get_embedding_settings()
    st.subheader("Connection Status")
    st.write(f"Admin password: {status_label(status.admin_password)}")
    st.write(f"Staff password: {status_label(status.staff_password)}")
    st.write(f"Supabase credentials: {status_label(status.supabase)}")
    st.write(f"Supabase client: {'connected' if store.enabled else 'not connected'}")
    st.write(f"Embedding provider: {embedding_settings.provider if embedding_settings else 'not configured'}")
    st.write(f"OpenRouter API key: {status_label(status.openrouter)}")
    st.write(f"Model access: {status_label(status.model_access)}")
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
    if "auth_role" not in st.session_state:
        st.session_state.auth_role = None


def reset_auth() -> None:
    st.session_state.auth_role = None
    st.session_state.admin_authenticated = False
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me about approved cleaning procedures, PPE, spills, chemicals, waste, or room cleaning.",
            "sources": [],
            "log_id": None,
        }
    ]


def render_login() -> None:
    st.title("Cleaning SOP Assistant")
    st.markdown(
        "<div class='subtitle'>Sign in to use approved cleaning procedures.</div>",
        unsafe_allow_html=True,
    )

    status = get_config_status()
    if not status.admin_password:
        st.warning("Admin login is not configured. Set ADMIN_PASSWORD in .env, Streamlit secrets, or the environment.")
    if not status.staff_password:
        st.warning("Staff login is not configured. Set STAFF_PASSWORD in .env, Streamlit secrets, or the environment.")

    login_tab, staff_tab = st.tabs(["Supervisor / Admin", "Staff"])
    with login_tab:
        render_role_login("admin", "Supervisor / admin password")
    with staff_tab:
        render_role_login("staff", "Staff password")


def render_role_login(role: str, label: str) -> None:
    password = configured_password(role)
    entered = st.text_input(label, type="password", key=f"{role}-login-password")
    if st.button("Sign in", use_container_width=True, key=f"{role}-login-button"):
        if password is None:
            st.error("This login is not configured yet.")
            return
        if entered == password:
            st.session_state.auth_role = role
            st.session_state.admin_authenticated = role == "admin"
            st.rerun()
        st.error("Incorrect password.")


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


def parse_frontmatter(markdown: str) -> dict[str, str]:
    match = re.search(r"^---\n(.*?)\n---", markdown, re.DOTALL)
    if not match:
        return {}
    return dict(
        re.findall(r"^([a-zA-Z_]+):\s*\"?(.+?)\"?\s*$", match.group(1), re.MULTILINE)
    )


def local_documents() -> list[dict]:
    documents = []
    markdown_dir = KNOWLEDGE_DIR / "markdown"
    if not markdown_dir.exists():
        return documents

    for path in sorted(markdown_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fields = parse_frontmatter(text)
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        raw_pdf = fields.get("raw_pdf", "")
        documents.append(
            {
                "id": path.name,
                "title": fields.get("title") or (title_match.group(1).strip() if title_match else path.stem),
                "category": fields.get("category", "general"),
                "source_name": fields.get("source_name", path.name),
                "raw_pdf_path": raw_pdf,
                "markdown_path": path.as_posix(),
                "file_hash": fields.get("file_hash", ""),
                "created_at": fields.get("uploaded_at", ""),
                "local": True,
            }
        )
    return documents


def update_local_document(document_id: str, updates: dict[str, str]) -> None:
    path = KNOWLEDGE_DIR / "markdown" / document_id
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    replacements = {
        "title": updates.get("title", "").strip(),
        "category": updates.get("category", "").strip(),
        "source_name": updates.get("source_name", "").strip(),
    }
    for field, value in replacements.items():
        if value:
            text = re.sub(
                rf"^{field}:\s*\"?.*?\"?\s*$",
                f'{field}: "{value.replace(chr(34), chr(92) + chr(34))}"',
                text,
                count=1,
                flags=re.MULTILINE,
            )
    if replacements["title"]:
        text = re.sub(r"^#\s+.+$", f"# {replacements['title']}", text, count=1, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


def delete_local_document(document_id: str) -> None:
    path = KNOWLEDGE_DIR / "markdown" / document_id
    if not path.exists():
        return

    fields = parse_frontmatter(path.read_text(encoding="utf-8"))
    raw_pdf = fields.get("raw_pdf")
    path.unlink()
    if raw_pdf:
        raw_path = ROOT / raw_pdf if not Path(raw_pdf).is_absolute() else Path(raw_pdf)
        try:
            if raw_path.exists() and raw_path.is_file() and raw_path.is_relative_to(KNOWLEDGE_DIR / "raw"):
                raw_path.unlink()
        except ValueError:
            return


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
    status = get_config_status()
    embedding_settings = get_embedding_settings()
    semantic_enabled = store.enabled and embedding_settings is not None and status.model_access
    model_enabled = status.model_access

    st.title("Cleaning SOP Assistant")
    st.markdown(
        "<div class='subtitle'>Fast answers from approved cleaning procedures.</div>",
        unsafe_allow_html=True,
    )

    has_documents = bool(chunks) or semantic_enabled
    if not has_documents:
        st.warning("No approved SOP documents have been uploaded yet. Please ask a supervisor to add them.")

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
                routed = route_message(prompt)
                if routed.intent == Intent.CASUAL:
                    response = SimpleResponse(routed.response, "casual", routed.reason)
                elif routed.intent == Intent.UNSUPPORTED:
                    response = SimpleResponse(routed.response, "unsupported", routed.reason)
                elif not has_documents:
                    response = SimpleResponse(
                        "No approved SOP documents have been uploaded yet. Please ask an admin to upload the approved procedures.",
                        "unavailable",
                        "No approved SOP documents uploaded.",
                    )
                else:
                    response = answer_question_best_available(
                        routed.search_query,
                        chunks,
                        store,
                        semantic_enabled=semantic_enabled,
                        model_enabled=model_enabled,
                    )

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

            if response.answer == FALLBACK_RESPONSE or row["category"] in {"unsupported", "unavailable"}:
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


def render_admin_console(store: SupabaseStore) -> None:
    st.title("Supervisor Console")
    st.markdown(
        "<div class='subtitle'>Manage approved SOP documents and review chat activity.</div>",
        unsafe_allow_html=True,
    )

    documents_tab, logs_tab, status_tab = st.tabs(["SOP Documents", "Logs", "Status"])
    with documents_tab:
        render_document_admin(store)
    with logs_tab:
        render_log_admin(store)
    with status_tab:
        render_connection_status(store)


def render_document_admin(store: SupabaseStore) -> None:
    st.subheader("Published SOP documents")
    documents = store.list_documents() if store.enabled else local_documents()
    if documents:
        for document in documents:
            title = document.get("title") or document.get("source_name") or document.get("id")
            with st.expander(str(title)):
                st.write(f"Category: {document.get('category', 'general')}")
                st.write(f"Source: {document.get('source_name', '')}")
                if document.get("markdown_path"):
                    st.write(f"Markdown: `{document.get('markdown_path')}`")
                if document.get("raw_pdf_path"):
                    st.write(f"Raw PDF: `{document.get('raw_pdf_path')}`")

                with st.form(f"edit-document-{document['id']}"):
                    new_title = st.text_input("Title", value=str(document.get("title", "")))
                    new_category = st.text_input("Category", value=str(document.get("category", "")))
                    new_source = st.text_input("Source name", value=str(document.get("source_name", "")))
                    submitted = st.form_submit_button("Save changes", use_container_width=True)
                    if submitted:
                        updates = {
                            "title": new_title,
                            "category": new_category,
                            "source_name": new_source,
                        }
                        if store.enabled:
                            store.update_document(str(document["id"]), updates)
                        else:
                            update_local_document(str(document["id"]), updates)
                        st.success("Document updated.")
                        st.rerun()

                delete_key = f"delete-document-{document['id']}"
                if st.button("Delete document", key=delete_key, use_container_width=True):
                    if store.enabled:
                        store.delete_document(str(document["id"]))
                    else:
                        delete_local_document(str(document["id"]))
                    st.success("Document deleted.")
                    st.rerun()

                replacement = st.file_uploader(
                    "Replace with revised PDF",
                    type=["pdf"],
                    key=f"replace-upload-{document['id']}",
                )
                if replacement and st.button(
                    "Replace document",
                    key=f"replace-document-{document['id']}",
                    use_container_width=True,
                ):
                    with st.spinner("Replacing SOP document..."):
                        try:
                            if store.enabled:
                                store.delete_document(str(document["id"]))
                            else:
                                delete_local_document(str(document["id"]))
                            publish_pdf(
                                store=store,
                                uploaded_pdf=replacement,
                                title=str(document.get("title", "")),
                                category=str(document.get("category", "")),
                                source_name=str(document.get("source_name", "")),
                            )
                            st.success("Document replaced.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
    else:
        st.info("No SOP documents have been published yet.")

    st.divider()
    render_upload_form(store)


def render_upload_form(store: SupabaseStore) -> None:
    st.subheader("Upload approved SOP PDF")
    uploaded_pdf = st.file_uploader(
        "Upload approved SOP PDF",
        type=["pdf"],
        help="PDF uploads are accepted up to the server limit configured in .streamlit/config.toml.",
    )
    title = st.text_input("Document title")
    category = st.text_input("Category", placeholder="spill response")
    source_name = st.text_input("Source name", placeholder="Approved SOP name or document code")

    if not uploaded_pdf:
        st.info("Choose a PDF before publishing.")
        st.button("Convert and publish", type="primary", use_container_width=True, disabled=True)
        return

    content = uploaded_pdf.getvalue()
    st.caption(f"Selected file: {uploaded_pdf.name} ({len(content) / (1024 * 1024):.1f} MB)")
    file_hash = sha256_bytes(content)
    existing = store.document_by_hash(file_hash) if store.enabled else None
    if existing:
        st.info(f"This PDF is already published as {existing.get('title', 'an existing document')}.")
        return

    if st.button("Convert and publish", type="primary", use_container_width=True):
        with st.spinner("Saving PDF, converting to Markdown, chunking, and publishing..."):
            try:
                document, supabase_status = publish_pdf(store, uploaded_pdf, title, category, source_name)
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


def publish_pdf(
    store: SupabaseStore,
    uploaded_pdf,
    title: str,
    category: str,
    source_name: str,
) -> tuple[object, str]:
    document = ingest_pdf_bytes(
        filename=uploaded_pdf.name,
        content=uploaded_pdf.getvalue(),
        title=title,
        category=category,
        source_name=source_name,
        knowledge_dir=KNOWLEDGE_DIR,
    )

    supabase_status = "Skipped: Supabase or embedding provider not configured"
    if store.enabled and get_embedding_settings() is not None:
        document_row = store.create_document(document)
        embeddings = embed_texts([chunk.chunk_text for chunk in document.chunks])
        chunk_count = store.replace_document_chunks(
            document_id=str(document_row["id"]),
            chunks=document.chunks,
            embeddings=embeddings,
        )
        supabase_status = f"Published {chunk_count} chunks"

    return document, supabase_status


def render_log_admin(store: SupabaseStore) -> None:
    st.subheader("Chat logs")
    chat_logs = store.list_chat_logs() if store.enabled else list_chat_logs(DATA_DIR)
    render_log_table(chat_logs, "chat", store)

    st.subheader("Unanswered questions")
    unanswered = store.list_unanswered_questions() if store.enabled else list_unanswered_questions(DATA_DIR)
    render_log_table(unanswered, "unanswered", store)


def render_log_table(rows: list[dict], row_type: str, store: SupabaseStore) -> None:
    if not rows:
        st.info("No records found.")
        return

    query = st.text_input("Filter records", key=f"{row_type}-filter")
    filtered = rows
    if query:
        normalized = query.lower()
        filtered = [
            row for row in rows
            if normalized in " ".join(str(value) for value in row.values()).lower()
        ]

    for row in filtered:
        label = row.get("question") or row.get("id", "Record")
        with st.expander(str(label)):
            for key, value in row.items():
                st.write(f"{key}: {value}")
            if st.button("Delete record", key=f"delete-{row_type}-{row.get('id')}", use_container_width=True):
                if store.enabled and row_type == "chat":
                    store.delete_chat_log(str(row.get("id")))
                elif store.enabled:
                    store.delete_unanswered_question(str(row.get("id")))
                elif row_type == "chat":
                    delete_chat_log(DATA_DIR, str(row.get("id")))
                else:
                    delete_unanswered_question(DATA_DIR, str(row.get("id")))
                st.success("Record deleted.")
                st.rerun()


load_dotenv_file(ROOT)
sync_streamlit_secrets()
init_state()
store = SupabaseStore()
chunks = load_knowledge_base(KNOWLEDGE_DIR)

if current_role() is None:
    render_login()
else:
    header_col, logout_col = st.columns([0.72, 0.28])
    with header_col:
        st.caption(f"Signed in as {'Supervisor / Admin' if is_admin() else 'Staff'}")
    with logout_col:
        if st.button("Log out", use_container_width=True):
            reset_auth()
            st.rerun()

    if is_admin():
        chat_tab, admin_tab = st.tabs(["Chat", "Admin"])
        with chat_tab:
            render_chat_tab(chunks, store)
        with admin_tab:
            render_admin_console(store)
    else:
        render_chat_tab(chunks, store)
