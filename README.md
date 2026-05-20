# Cleaning SOP Assistant Chatbot MVP

A mobile-friendly Streamlit chatbot that answers cleaning procedure questions from approved SOPs. Staff sign in for chat-only access, while supervisors/admins can upload and manage PDF procedures, review logs, keep the raw PDF for reference, convert it to Obsidian-style Markdown, chunk it, and publish it to Supabase vector search.

## What Is Included

- Streamlit chat interface
- Two-password role login for staff chat access and supervisor/admin CRUD access
- Admin PDF upload and document/log management
- Raw PDF storage in `knowledge_base/raw/`
- Markdown knowledge base in `knowledge_base/markdown/`
- PDF text extraction with OCR fallback for scanned pages
- Supabase + pgvector document/chunk storage
- OpenRouter embeddings for semantic search, with OpenAI fallback
- OpenRouter or OpenAI grounded answer generation
- Local keyword fallback when Supabase/embeddings are not configured
- Grounded fallback for unsupported questions
- Chat, source, category, unanswered question, and helpfulness logging
- Focused tests for retrieval and guardrails

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open the forwarded port or local URL shown by Streamlit.
PDF uploads are configured up to 1024 MB in `.streamlit/config.toml`. Restart Streamlit after changing that limit.

OCR requires the system `tesseract` executable. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

## Configuration

For local development, copy the example file and fill in your real values:

```bash
cp .env.example .env
```

```env
ADMIN_PASSWORD=change-me
STAFF_PASSWORD=change-me
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SECRET_KEY=your-sb-secret-key
SUPABASE_SERVICE_ROLE_KEY=
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_SITE_URL=http://localhost:8501
OPENROUTER_APP_NAME=Cleaning SOP Assistant
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-5.4-mini
```

The real `.env` file is ignored by git. You can also set the same values as environment variables or in `.streamlit/secrets.toml`:

```toml
ADMIN_PASSWORD = "change-me"
STAFF_PASSWORD = "change-me"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SECRET_KEY = "your-sb-secret-key"
# Optional legacy fallback:
SUPABASE_SERVICE_ROLE_KEY = ""
OPENROUTER_API_KEY = "your-openrouter-api-key"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENROUTER_SITE_URL = "http://localhost:8501"
OPENROUTER_APP_NAME = "Cleaning SOP Assistant"
OPENAI_API_KEY = ""
OPENAI_CHAT_MODEL = "gpt-5.4-mini"
```

`OPENAI_CHAT_MODEL` is optional. If missing, the app defaults to `gpt-5.4-mini`.

### Supabase Setup

1. Create or open your Supabase project.
2. In Supabase, open the SQL Editor.
3. Copy the full contents of `supabase_schema.sql`.
4. Run the SQL. This creates the `documents`, `document_chunks`, `chat_logs`, `unanswered_questions` tables and the `match_document_chunks` RPC.
5. In Supabase project settings, copy the Project URL into `SUPABASE_URL`.
6. In API Keys, create/copy the backend secret key that starts with `sb_secret_` into `SUPABASE_SECRET_KEY`.
7. If your project only shows legacy keys, copy the `service_role` key into `SUPABASE_SERVICE_ROLE_KEY` instead.
8. Do not use the `sb_publishable_...` key for this Streamlit admin ingestion flow. It is for public/client-side access and is not enough for server-side document inserts/upserts.
9. Restart Streamlit after changing `.env`.

If a Supabase MCP tool is available in your Codex session later, it can be used to apply `supabase_schema.sql` and verify the database. In this session, no Supabase MCP tool is exposed, so the manual SQL Editor path is the reliable setup.

### OpenRouter Model And Embedding Access

To use OpenRouter for answer generation and semantic retrieval, set:

```env
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

OpenRouter is preferred for both answer generation and embeddings when `OPENROUTER_API_KEY` is present. The app sends chat requests to `https://openrouter.ai/api/v1/chat/completions` and embedding requests to `https://openrouter.ai/api/v1/embeddings`.

### OpenAI API Fallback

`OPENAI_API_KEY` is optional and only used as a fallback if OpenRouter is not configured. A ChatGPT Plus/Pro/Business subscription cannot be used directly as an API key; OpenAI API billing is separate from ChatGPT subscriptions. See OpenAI's API pricing FAQ and help article:

- https://openai.com/api/pricing/
- https://help.openai.com/en/articles/8156019

## Try These Questions

- How do I clean a blood spill?
- What PPE is required for isolation rooms?
- What chemical should I use for toilets?
- How often should washrooms be checked?
- What dilution ratio should I use for chemicals?

## Knowledge Base

Approved Markdown procedures live in `knowledge_base/markdown/`.
Uploaded PDFs are stored unchanged in `knowledge_base/raw/`.
The production knowledge base starts empty. Supervisors/admins add approved procedures from the Admin tab by uploading PDFs. Staff users only see the chat experience and answer source citations.

Each file can include:

```markdown
# Procedure Title
Category: spill response
Source: Approved SOP Name

## Section Title
- Procedure step
- Safety instruction
```

Uploaded PDFs are converted to Markdown with frontmatter:

```markdown
---
title: "Procedure Title"
category: "spill response"
source_name: "Approved SOP Name"
file_hash: "..."
raw_pdf: "knowledge_base/raw/example.pdf"
uploaded_at: "..."
---
# Procedure Title

## Page 1
...
```

The assistant cites the matched procedure title, section title, and page number when available.

## Logs

When Supabase is configured, logs are written to Supabase. Without Supabase, MVP logs are written locally:

- `data/chat_logs.csv`
- `data/unanswered_questions.csv`

These files are created at runtime and can later be replaced with Supabase tables matching the PRD.
Supervisors/admins can review and delete these records in the Admin tab.

## Tests

```bash
pytest
```

## MVP Notes

This version supports the intended Supabase/OpenAI path and still runs locally without secrets. Without Supabase/OpenAI, chat falls back to local Markdown keyword retrieval so development remains easy.
