-- 001_init.sql
-- Trade Journal Pro (Supabase / Postgres)
-- Recomendado: correr esto en un proyecto limpio o después de DROP TABLE.
-- Incluye: trades + sessions, columnas nuevas para métricas,
-- trade_date como generated column (sin trigger), checks y defaults “compatibles con legacy”.

-- ------------------------------------------------------------
-- Extensiones útiles (Supabase normalmente ya las tiene)
-- ------------------------------------------------------------
create extension if not exists pgcrypto;

-- ------------------------------------------------------------
-- DROP (si vas a borrar y recrear)
-- ------------------------------------------------------------
drop table if exists public.trades cascade;
drop table if exists public.sessions cascade;

-- ------------------------------------------------------------
-- Sessions
-- ------------------------------------------------------------
create table public.sessions (
  id uuid primary key default gen_random_uuid(),
  start_time timestamptz not null,
  end_time timestamptz,
  duration_min numeric(10,2),
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists idx_sessions_start_time
on public.sessions(start_time desc);

-- ------------------------------------------------------------
-- Trades
-- ------------------------------------------------------------
create table public.trades (
  id uuid primary key default gen_random_uuid(),

  -- Fecha/hora del trade (en UTC, pero timestamptz guarda TZ)
  trade_time timestamptz not null default now(),

  -- Fecha derivada en UTC (evita trigger e inconsistencias)
  trade_date date generated always as ((trade_time at time zone 'UTC')::date) stored,

  asset text not null,
  timeframe text not null,

  amount numeric(12,2) not null check (amount >= 0),

  direction text not null check (direction in ('UP','DOWN')),
  outcome   text not null check (outcome in ('WIN','LOSS','TIE')),

  payout_pct numeric(5,2) not null default 0 check (payout_pct >= 0),
  pnl numeric(12,2) not null default 0,

  -- Campos originales
  emotion text not null,
  notes text,

  created_at timestamptz not null default now(),

  -- ----------------------------------------------------------
  -- Campos extra (para dashboard/mejoras). Todos “legacy-friendly”
  -- (NULL permitido o default, para no romper imports viejos)
  -- ----------------------------------------------------------
  setup_tag text,            -- ejemplo: Breakout, Reversal, etc.
  market_regime text,        -- ejemplo: Trend, Range, Volatile
  quality_grade text,        -- ejemplo: A/B/C/D
  checklist_pass boolean not null default true,

  -- Sesión y evidencia
  session_id uuid references public.sessions(id) on delete set null,
  screenshot_url text,

  -- ----------------------------------------------------------
  -- Checks
  -- ----------------------------------------------------------
  constraint trades_emotion_check check (
    emotion in ('Neutral','Confiado','Enfocado','Ansioso','Impulsivo','Cansado','Frustrado')
  ),

  -- Si quieres forzar catálogos cerrados, descomenta estos checks,
  -- pero ojo: te puede volver a romper imports si no mapeas valores.
  -- constraint trades_quality_check check (quality_grade is null or quality_grade in ('A','B','C','D')),
  -- constraint trades_regime_check  check (market_regime is null or market_regime in ('Trend','Range','Volatile','LowVol')),

  -- Validación mínima útil
  constraint trades_asset_nonempty check (length(trim(asset)) > 0),
  constraint trades_timeframe_nonempty check (length(trim(timeframe)) > 0)
);

create index if not exists idx_trades_trade_date
on public.trades (trade_date);

create index if not exists idx_trades_trade_time
on public.trades (trade_time desc);

create index if not exists idx_trades_session_id
on public.trades (session_id);

create index if not exists idx_trades_asset
on public.trades (asset);

create index if not exists idx_trades_timeframe
on public.trades (timeframe);

create index if not exists idx_trades_emotion
on public.trades (emotion);
