from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from trade_journal.analytics.charts import equity_curve, pnl_by_asset
from trade_journal.analytics.kpis import compute_kpis
from trade_journal.app.utils import get_recent_trades

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Dashboard")

df = get_recent_trades()

colf1, colf2 = st.columns([1, 2])
with colf1:
    day = st.date_input("Día", value=date.today())
with colf2:
    st.caption("Los datos se leen desde Supabase (tabla `trades`).")

df_day = df[df.get("trade_date").astype(str) == str(day)] if not df.empty else df

k = compute_kpis(df_day if not df_day.empty else pd.DataFrame())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", k["trades"])
c2.metric("WinRate", f'{k["winrate"]:.1f}%')
c3.metric("PnL", f'{k["pnl"]:.2f}')
c4.metric("Avg PnL", f'{k["avg_pnl"]:.2f}')

st.divider()

left, right = st.columns(2)
with left:
    fig = equity_curve(df_day)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sin trades para el día seleccionado.")
with right:
    fig2 = pnl_by_asset(df_day)
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Sin datos para gráfico por activo.")

st.subheader("Trades del día")
if df_day.empty:
    st.write("—")
else:
    show_cols = [
        "trade_time","asset","timeframe","amount","direction","outcome","payout_pct","pnl","emotion","notes"
    ]
    cols = [c for c in show_cols if c in df_day.columns]
    st.dataframe(df_day[cols].sort_values("trade_time", ascending=False), use_container_width=True)
