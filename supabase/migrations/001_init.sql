-- 001_init.sql
-- Tablas base para Trade Journal Pro
-- Ejecuta esto en Supabase SQL Editor.

-- Trades
create table if not exists public.trades (
  id uuid primary key default gen_random_uuid(),
  trade_time timestamptz not null default now(),
  trade_date date generated always as (trade_time::date) stored,
  asset text not null,
  timeframe text not null,
  amount numeric(12,2) not null check (amount >= 0),
  direction text not null check (direction in ('UP','DOWN')),
  outcome text not null check (outcome in ('WIN','LOSS','TIE')),
  payout_pct numeric(5,2) not null default 0 check (payout_pct >= 0),
  pnl numeric(12,2) not null default 0,
  emotion text,
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists idx_trades_trade_time on public.trades(trade_time desc);
create index if not exists idx_trades_trade_date on public.trades(trade_date);

-- Sessions (para cronómetro / disciplina)
create table if not exists public.sessions (
  id uuid primary key default gen_random_uuid(),
  start_time timestamptz not null,
  end_time timestamptz,
  duration_min numeric(10,2),
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists idx_sessions_start_time on public.sessions(start_time desc);

-- (Opcional) RLS:
-- alter table public.trades enable row level security;
-- alter table public.sessions enable row level security;
-- Luego crea policies según tu auth / service key.
