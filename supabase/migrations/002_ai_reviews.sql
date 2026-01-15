-- 002_ai_reviews.sql
-- Tabla para guardar reviews IA por sesión y por semana

create table if not exists public.ai_reviews (
  id uuid primary key default gen_random_uuid(),
  scope text not null check (scope in ('session', 'weekly')),
  session_id uuid references public.sessions(id) on delete cascade,
  week_start date,
  week_end date,
  payload jsonb,
  review jsonb,
  model text,
  prompt_version text,
  payload_hash text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_ai_reviews_session_id
on public.ai_reviews(session_id);

create index if not exists idx_ai_reviews_week_start
on public.ai_reviews(week_start);

create index if not exists idx_ai_reviews_scope
on public.ai_reviews(scope);

create unique index if not exists uniq_ai_reviews_session
on public.ai_reviews(session_id)
where scope = 'session';

create unique index if not exists uniq_ai_reviews_week
on public.ai_reviews(week_start)
where scope = 'weekly';
