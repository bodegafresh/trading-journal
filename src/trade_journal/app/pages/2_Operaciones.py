from __future__ import annotations

from datetime import datetime

import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_trades
from trade_journal.domain.models import Direction, Outcome, TradeCreate

st.set_page_config(page_title="Operaciones", page_icon="🧾", layout="wide")

st.title("🧾 Operaciones")

trade_repo, _ = get_repos()

with st.form("new_trade", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        asset = st.text_input("Activo", value="EUR/USD")
        timeframe = st.text_input("Timeframe", value="M1")
    with c2:
        amount = st.number_input("Monto (USD)", min_value=0.0, value=1.0, step=1.0)
        payout = st.number_input("Payout %", min_value=0.0, value=80.0, step=1.0)
    with c3:
        direction = st.selectbox("Dirección", options=[d.value for d in Direction], index=0)
        outcome = st.selectbox("Resultado", options=[o.value for o in Outcome], index=0)
    with c4:
        pnl = st.number_input("PnL (USD)", value=0.0, step=1.0)
        emotion = st.text_input("Emoción (opcional)", value="")
    notes = st.text_area("Notas (opcional)", height=80)

    submitted = st.form_submit_button("Guardar trade")

if submitted:
    try:
        trade = TradeCreate(
            trade_time=datetime.utcnow(),
            asset=asset.strip(),
            timeframe=timeframe.strip(),
            amount=float(amount),
            direction=Direction(direction),
            outcome=Outcome(outcome),
            payout_pct=float(payout),
            pnl=float(pnl),
            emotion=(emotion.strip() or None),
            notes=(notes.strip() or None),
        )
        inserted = trade_repo.create(trade)
        st.success(f"Trade guardado ✅ id={inserted.get('id')}")
        st.cache_data.clear()  # refrescar tablas
    except Exception as e:
        st.error(f"No se pudo guardar: {e}")

st.divider()

st.subheader("Trades recientes")
df = get_recent_trades()
if df.empty:
    st.write("—")
else:
    st.dataframe(
        df.sort_values("trade_time", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
