-- 001_init.sql
-- Tablas base para Trade Journal Pro
-- Ejecuta esto en Supabase SQL Editor.

-- Trades
-- ⚠️ SOLO si todavía no tienes datos y quieres recrear
-- drop table if exists public.trades cascade;
create table if not exists public.trades (
  id uuid primary key default gen_random_uuid(),
  trade_time timestamptz not null default now(),
  trade_date date not null, -- se setea por trigger (UTC)
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

create or replace function public.trades_set_trade_date()
returns trigger language plpgsql as $$
begin
  -- Fecha derivada en UTC (consistente y “no depende del cliente”)
  new.trade_date := (new.trade_time at time zone 'UTC')::date;
  return new;
end;
$$;

drop trigger if exists trg_trades_set_trade_date on public.trades;

create trigger trg_trades_set_trade_date
before insert or update of trade_time
on public.trades
for each row execute function public.trades_set_trade_date();

create index if not exists idx_trades_trade_date
on public.trades (trade_date);



create table if not exists public.sessions (
  id uuid primary key default gen_random_uuid(),
  start_time timestamptz not null,
  end_time timestamptz,
  duration_min numeric(10,2),
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists idx_sessions_start_time on public.sessions(start_time desc);
