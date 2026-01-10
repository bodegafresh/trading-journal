from __future__ import annotations

from datetime import date, datetime
from typing import Tuple

import pandas as pd
import pytz
import streamlit as st
from dotenv import load_dotenv

from trade_journal.data.supabase_client import load_supabase_from_env
from trade_journal.data.repositories import SessionRepository, TradeRepository

LOCAL_TZ = pytz.timezone("America/Santiago")

@st.cache_resource
def get_repos() -> Tuple[TradeRepository, SessionRepository]:
    load_dotenv()
    sb = load_supabase_from_env()
    return TradeRepository(sb), SessionRepository(sb)


@st.cache_data(ttl=15)
def get_recent_trades(limit: int = 2000) -> pd.DataFrame:
    trade_repo, _ = get_repos()
    rows = trade_repo.list_recent(limit=limit) or []
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # ✅ tolera ISO con/sin microsegundos, con/sin Z
    if "trade_time" in df.columns:
        dt = pd.to_datetime(df["trade_time"], utc=True, errors="coerce", format="mixed")
        # lo muestro en hora local pero sin tz (más cómodo en tablas)
        df["trade_time"] = dt.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    # trade_date si existe
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date

    return df


@st.cache_data(ttl=15)
def get_recent_sessions(limit: int = 500) -> pd.DataFrame:
    _, session_repo = get_repos()
    rows = session_repo.list_recent(limit=limit)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_convert(None)
        if "end_time" in df.columns:
            df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce").dt.tz_convert(None)
    return df

