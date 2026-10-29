-- Prototype tables for the local Streamlit app.
-- Run this once in the Supabase SQL editor.
-- These policies allow the anon key to read and write. Do not use them for real patient data.

create table if not exists public.hospital_profile (
  id integer primary key default 1,
  name text not null,
  address text not null,
  phone text not null,
  email text not null,
  website text not null,
  logo_url text not null,
  colors text[] not null,
  fonts text[] not null,
  font_sizes text[] not null,
  font_weights text[] not null,
  font_styles text[] not null,
  font_variants text[] not null,
  updated_at timestamptz not null default now(),
  constraint hospital_profile_singleton check (id = 1)
);

create table if not exists public.chat_messages (
  id bigint generated always as identity primary key,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.document_chunks (
  id text primary key,
  source text not null,
  page integer not null,
  content text not null,
  approval_status text not null,
  document_version text not null,
  updated_at timestamptz not null default now()
);

alter table public.hospital_profile enable row level security;
alter table public.chat_messages enable row level security;
alter table public.document_chunks enable row level security;

drop policy if exists "prototype read hospital" on public.hospital_profile;
drop policy if exists "prototype insert hospital" on public.hospital_profile;
drop policy if exists "prototype update hospital" on public.hospital_profile;
drop policy if exists "prototype read chat" on public.chat_messages;
drop policy if exists "prototype insert chat" on public.chat_messages;
drop policy if exists "prototype delete chat" on public.chat_messages;
drop policy if exists "prototype read documents" on public.document_chunks;
drop policy if exists "prototype insert documents" on public.document_chunks;
drop policy if exists "prototype update documents" on public.document_chunks;
drop policy if exists "prototype delete documents" on public.document_chunks;

create policy "prototype read hospital"
  on public.hospital_profile for select
  using (true);

create policy "prototype insert hospital"
  on public.hospital_profile for insert
  with check (true);

create policy "prototype update hospital"
  on public.hospital_profile for update
  using (true)
  with check (true);

create policy "prototype read chat"
  on public.chat_messages for select
  using (true);

create policy "prototype insert chat"
  on public.chat_messages for insert
  with check (true);

create policy "prototype delete chat"
  on public.chat_messages for delete
  using (true);

create policy "prototype read documents"
  on public.document_chunks for select
  using (true);

create policy "prototype insert documents"
  on public.document_chunks for insert
  with check (true);

create policy "prototype update documents"
  on public.document_chunks for update
  using (true)
  with check (true);

create policy "prototype delete documents"
  on public.document_chunks for delete
  using (true);

grant select, insert, update on public.hospital_profile to anon, authenticated;
grant select, insert, delete on public.chat_messages to anon, authenticated;
grant select, insert, update, delete on public.document_chunks to anon, authenticated;
grant usage, select on sequence public.chat_messages_id_seq to anon, authenticated;
