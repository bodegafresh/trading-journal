from __future__ import annotations

from datetime import date, datetime
from typing import Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from trade_journal.data.supabase_client import load_supabase_from_env
from trade_journal.data.repositories import SessionRepository, TradeRepository


@st.cache_resource
def get_repos() -> Tuple[TradeRepository, SessionRepository]:
    load_dotenv()
    sb = load_supabase_from_env()
    return TradeRepository(sb), SessionRepository(sb)


@st.cache_data(ttl=15)
def get_recent_trades(limit: int = 2000) -> pd.DataFrame:
    trade_repo, _ = get_repos()
    rows = trade_repo.list_recent(limit=limit)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["trade_time"] = pd.to_datetime(df["trade_time"], utc=True).dt.tz_convert(None)
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
