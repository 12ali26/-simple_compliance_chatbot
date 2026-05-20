create extension if not exists vector;

create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    category text not null default 'general',
    source_name text not null,
    raw_pdf_path text not null,
    markdown_path text not null,
    file_hash text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents(id) on delete cascade,
    chunk_text text not null,
    section_title text not null,
    page_number integer,
    chunk_index integer not null default 0,
    embedding vector(1536) not null,
    created_at timestamptz not null default now()
);

create index if not exists document_chunks_embedding_idx
on document_chunks using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

create index if not exists document_chunks_document_id_idx
on document_chunks(document_id);

create table if not exists chat_logs (
    id uuid primary key default gen_random_uuid(),
    question text not null,
    answer text not null,
    source_used text,
    helpful boolean,
    created_at timestamptz not null default now()
);

create table if not exists unanswered_questions (
    id uuid primary key default gen_random_uuid(),
    question text not null,
    reason text not null,
    created_at timestamptz not null default now()
);

create or replace function match_document_chunks(
    query_embedding vector(1536),
    match_count int default 5,
    similarity_threshold float default 0.72
)
returns table (
    id uuid,
    document_id uuid,
    document_title text,
    source_name text,
    raw_pdf_path text,
    markdown_path text,
    chunk_text text,
    section_title text,
    page_number integer,
    similarity float
)
language sql stable
as $$
    select
        dc.id,
        dc.document_id,
        d.title as document_title,
        d.source_name,
        d.raw_pdf_path,
        d.markdown_path,
        dc.chunk_text,
        dc.section_title,
        dc.page_number,
        1 - (dc.embedding <=> query_embedding) as similarity
    from document_chunks dc
    join documents d on d.id = dc.document_id
    where 1 - (dc.embedding <=> query_embedding) >= similarity_threshold
    order by dc.embedding <=> query_embedding
    limit match_count;
$$;
