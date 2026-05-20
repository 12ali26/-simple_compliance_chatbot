# Cleaning SOP Assistant Chatbot MVP

A mobile-friendly Streamlit chatbot that answers cleaning procedure questions from approved SOPs. Admins can upload PDF procedures, keep the raw PDF for reference, convert it to Obsidian-style Markdown, chunk it, and publish it to Supabase vector search.

## What Is Included

- Streamlit chat interface
- Admin PDF upload with shared password gate
- Raw PDF storage in `knowledge_base/raw/`
- Markdown knowledge base in `knowledge_base/markdown/`
- PDF text extraction with OCR fallback for scanned pages
- Supabase + pgvector document/chunk storage
- OpenAI embeddings and grounded answer generation
- Local keyword fallback when Supabase/OpenAI are not configured
- Grounded fallback for unsupported questions
- Chat, source, category, unanswered question, and helpfulness logging
- Focused tests for retrieval and guardrails

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open the forwarded port or local URL shown by Streamlit.

OCR requires the system `tesseract` executable. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

## Configuration

Set these in `.streamlit/secrets.toml` or environment variables:

```toml
ADMIN_PASSWORD = "change-me"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
OPENAI_API_KEY = "your-openai-key"
OPENAI_CHAT_MODEL = "gpt-5.4-mini"
```

`OPENAI_CHAT_MODEL` is optional. If missing, the app defaults to `gpt-5.4-mini`.

Run the SQL in `supabase_schema.sql` in Supabase before publishing PDFs.

## Try These Questions

- How do I clean a blood spill?
- What PPE is required for isolation rooms?
- What chemical should I use for toilets?
- How often should washrooms be checked?
- What dilution ratio should I use for chemicals?

## Knowledge Base

Approved Markdown procedures live in `knowledge_base/markdown/`.
Uploaded PDFs are stored unchanged in `knowledge_base/raw/`.

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

## Tests

```bash
pytest
```

## MVP Notes

This version supports the intended Supabase/OpenAI path and still runs locally without secrets. Without Supabase/OpenAI, chat falls back to local Markdown keyword retrieval so development remains easy.
