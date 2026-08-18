-- RAG application schema for Supabase/Postgres
-- Run this entire file in the Supabase SQL Editor.
-- The backend should use the Supabase service-role key for ingestion and audit writes.

create extension if not exists pgcrypto;
create extension if not exists vector;

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

create table if not exists public.projects (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid references auth.users(id) on delete set null,
  name text not null check (char_length(name) between 1 and 120),
  description text not null default '',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  uploaded_by uuid references auth.users(id) on delete set null,
  filename text not null,
  storage_path text,
  content_type text not null default 'application/pdf',
  size_bytes bigint not null default 0 check (size_bytes >= 0),
  checksum text,
  publisher text not null default 'UNKNOWN',
  source_url text not null default '',
  status text not null default 'uploaded' check (status in ('uploaded', 'queued', 'ingesting', 'ready', 'failed')),
  job_id uuid,
  chunk_count integer not null default 0 check (chunk_count >= 0),
  error text,
  ingested_at timestamptz,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references public.documents(id) on delete cascade,
  project_id uuid not null references public.projects(id) on delete cascade,
  chunk_id text not null,
  page_number integer not null default 0,
  section_title text not null default 'General',
  text_content text not null,
  token_count integer,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(384),
  created_at timestamptz not null default timezone('utc', now()),
  unique (document_id, chunk_id)
);

create table if not exists public.ingestion_jobs (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.projects(id) on delete cascade,
  document_id uuid not null references public.documents(id) on delete cascade,
  status text not null default 'queued' check (status in ('queued', 'running', 'succeeded', 'failed')),
  started_at timestamptz,
  finished_at timestamptz,
  result jsonb,
  error text,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references public.projects(id) on delete set null,
  user_id uuid references auth.users(id) on delete set null,
  title text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null default '',
  answer jsonb,
  refusal boolean not null default false,
  request_id text,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.retrieval_runs (
  id uuid primary key default gen_random_uuid(),
  request_id text not null,
  project_id uuid references public.projects(id) on delete set null,
  conversation_id uuid references public.conversations(id) on delete set null,
  query text not null,
  method text not null default 'hybrid' check (method in ('semantic', 'bm25', 'hybrid')),
  requested_k integer not null default 5 check (requested_k between 1 and 20),
  safety_threshold real not null,
  max_similarity real not null default 0,
  safe_to_generate boolean not null default false,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.retrieval_chunks (
  id uuid primary key default gen_random_uuid(),
  retrieval_run_id uuid not null references public.retrieval_runs(id) on delete cascade,
  chunk_id text not null,
  document_id uuid references public.documents(id) on delete set null,
  rank integer not null,
  similarity_score real not null default 0,
  excerpt text not null default '',
  metadata jsonb not null default '{}'::jsonb
);

create table if not exists public.evaluations (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references public.projects(id) on delete set null,
  conversation_id uuid references public.conversations(id) on delete set null,
  message_id uuid references public.messages(id) on delete set null,
  query text not null,
  answer jsonb,
  expected_chunk_ids jsonb not null default '[]'::jsonb,
  citation_precision real,
  citation_recall real,
  grounded boolean,
  notes jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.pipeline_logs (
  id uuid primary key default gen_random_uuid(),
  request_id text not null,
  project_id uuid references public.projects(id) on delete set null,
  conversation_id uuid references public.conversations(id) on delete set null,
  message_id uuid references public.messages(id) on delete set null,
  stage text not null check (stage in (
    'request_received', 'auth_validated', 'input_validated', 'context_retrieved',
    'safety_threshold_checked', 'generation_started', 'answer_validated',
    'citations_validated', 'refusal_returned', 'answer_returned', 'error'
  )),
  status text not null default 'ok' check (status in ('ok', 'blocked', 'error')),
  latency_ms integer,
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now())
);

create index if not exists idx_projects_owner on public.projects(owner_id);
create index if not exists idx_documents_project on public.documents(project_id);
create index if not exists idx_documents_status on public.documents(status);
create index if not exists idx_chunks_project on public.document_chunks(project_id);
create index if not exists idx_chunks_document on public.document_chunks(document_id);
create index if not exists idx_jobs_document on public.ingestion_jobs(document_id);
create index if not exists idx_jobs_status on public.ingestion_jobs(status);
create index if not exists idx_conversations_project on public.conversations(project_id);
create index if not exists idx_messages_conversation_created on public.messages(conversation_id, created_at);
create index if not exists idx_retrieval_runs_request on public.retrieval_runs(request_id);
create index if not exists idx_retrieval_chunks_run_rank on public.retrieval_chunks(retrieval_run_id, rank);
create index if not exists idx_evaluations_project on public.evaluations(project_id);
create index if not exists idx_pipeline_logs_request on public.pipeline_logs(request_id);
create index if not exists idx_pipeline_logs_stage on public.pipeline_logs(stage);

-- Optional vector index for a future pgvector retrieval implementation.
-- It is intentionally not required by the current FAISS-backed engine.
create index if not exists idx_document_chunks_embedding
  on public.document_chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

 drop trigger if exists projects_set_updated_at on public.projects;
 create trigger projects_set_updated_at before update on public.projects
 for each row execute function public.set_updated_at();

 drop trigger if exists documents_set_updated_at on public.documents;
 create trigger documents_set_updated_at before update on public.documents
 for each row execute function public.set_updated_at();

 drop trigger if exists ingestion_jobs_set_updated_at on public.ingestion_jobs;
 create trigger ingestion_jobs_set_updated_at before update on public.ingestion_jobs
 for each row execute function public.set_updated_at();

 drop trigger if exists conversations_set_updated_at on public.conversations;
 create trigger conversations_set_updated_at before update on public.conversations
 for each row execute function public.set_updated_at();

-- RLS protects browser access. Server-side service-role requests bypass RLS.
alter table public.projects enable row level security;
alter table public.documents enable row level security;
alter table public.document_chunks enable row level security;
alter table public.ingestion_jobs enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.retrieval_runs enable row level security;
alter table public.retrieval_chunks enable row level security;
alter table public.evaluations enable row level security;
alter table public.pipeline_logs enable row level security;

 drop policy if exists projects_owner_select on public.projects;
 create policy projects_owner_select on public.projects for select using (owner_id = auth.uid() or owner_id is null);
 drop policy if exists projects_owner_write on public.projects;
 create policy projects_owner_write on public.projects for all using (owner_id = auth.uid() or owner_id is null) with check (owner_id = auth.uid() or owner_id is null);

 drop policy if exists documents_project_access on public.documents;
 create policy documents_project_access on public.documents for all using (
   exists (select 1 from public.projects p where p.id = documents.project_id and (p.owner_id = auth.uid() or p.owner_id is null))
 ) with check (
   exists (select 1 from public.projects p where p.id = documents.project_id and (p.owner_id = auth.uid() or p.owner_id is null))
 );

 drop policy if exists chunks_project_access on public.document_chunks;
 create policy chunks_project_access on public.document_chunks for all using (
   exists (select 1 from public.projects p where p.id = document_chunks.project_id and (p.owner_id = auth.uid() or p.owner_id is null))
 ) with check (
   exists (select 1 from public.projects p where p.id = document_chunks.project_id and (p.owner_id = auth.uid() or p.owner_id is null))
 );

 drop policy if exists jobs_project_access on public.ingestion_jobs;
 create policy jobs_project_access on public.ingestion_jobs for all using (
   exists (select 1 from public.projects p where p.id = ingestion_jobs.project_id and (p.owner_id = auth.uid() or p.owner_id is null))
 ) with check (
   exists (select 1 from public.projects p where p.id = ingestion_jobs.project_id and (p.owner_id = auth.uid() or p.owner_id is null))
 );

 drop policy if exists conversations_access on public.conversations;
 create policy conversations_access on public.conversations for all using (user_id = auth.uid() or user_id is null) with check (user_id = auth.uid() or user_id is null);

 drop policy if exists messages_access on public.messages;
 create policy messages_access on public.messages for all using (
   exists (select 1 from public.conversations c where c.id = messages.conversation_id and (c.user_id = auth.uid() or c.user_id is null))
 ) with check (
   exists (select 1 from public.conversations c where c.id = messages.conversation_id and (c.user_id = auth.uid() or c.user_id is null))
 );

-- Retrieval, evaluation, and pipeline tables are server-audit data.
-- No browser policies are added intentionally; the service role can still write them.

comment on table public.projects is 'RAG knowledge projects';
comment on table public.documents is 'Uploaded source documents and ingestion state';
comment on table public.document_chunks is 'Chunk metadata and optional pgvector embeddings';
comment on table public.ingestion_jobs is 'Document ingestion job lifecycle';
comment on table public.conversations is 'User/project conversations';
comment on table public.messages is 'Conversation messages and grounded responses/refusals';
comment on table public.retrieval_runs is 'Retrieval request-level trace';
comment on table public.retrieval_chunks is 'Ranked context returned by retrieval';
comment on table public.evaluations is 'Optional answer/citation evaluation results';
comment on table public.pipeline_logs is 'Audit trail for auth, retrieval, safety, generation, validation, and return stages';
