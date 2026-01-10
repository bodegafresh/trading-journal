from __future__ import annotations

from datetime import datetime

import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_trades
from trade_journal.domain.models import Direction, Outcome, TradeCreate

st.set_page_config(page_title="Operaciones", page_icon="🧾", layout="wide")

st.title("🧾 Operaciones")

# Dropdowns del script original
ASSETS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CAD",
    "EUR/JPY",
    "EUR/GBP",
    "BTC/USD",
    "Asia comoposites",
    "Euro composites",
    "Compound Index",
]
TIMEFRAMES = ["1m", "5m"]

# Emociones definidas por ti (obligatorias)
EMOTIONS = ["Neutral", "Confiado", "Enfocado", "Ansioso", "Impulsivo", "Cansado", "Frustrado"]


def calculate_pnl(amount: float, payout_pct: float, outcome: str) -> float:
    """PnL calculado:
    - WIN  -> + amount * payout%
    - LOSS -> - amount
    - TIE  -> 0
    """
    if outcome == Outcome.WIN.value:
        return round(amount * (payout_pct / 100.0), 2)
    if outcome == Outcome.LOSS.value:
        return round(-amount, 2)
    return 0.0


trade_repo, _ = get_repos()

with st.form("new_trade", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        asset = st.selectbox("Activo", options=ASSETS, index=0)
        timeframe = st.selectbox("Timeframe", options=TIMEFRAMES, index=0)

    with c2:
        amount = st.number_input("Monto (USD)", min_value=0.0, value=10.0, step=1.0)
        payout = st.number_input("Payout %", min_value=0.0, value=80.0, step=1.0)

    with c3:
        direction = st.selectbox("Dirección", options=[d.value for d in Direction], index=0)
        outcome = st.selectbox("Resultado", options=[o.value for o in Outcome], index=0)

    with c4:
        emotion = st.selectbox("Emoción", options=EMOTIONS, index=0)

    notes = st.text_area("Notas (opcional)", height=80)

    # PnL NO se muestra como input ni se edita. Se calcula al guardar.
    submitted = st.form_submit_button("Guardar trade")

if submitted:
    try:
        pnl = calculate_pnl(float(amount), float(payout), str(outcome))

        trade = TradeCreate(
            trade_time=datetime.utcnow(),
            asset=asset.strip(),
            timeframe=timeframe.strip(),
            amount=float(amount),
            direction=Direction(direction),
            outcome=Outcome(outcome),
            payout_pct=float(payout),
            pnl=float(pnl),
            emotion=emotion.strip(),  # obligatorio + validado por lista
            notes=(notes.strip() or None),
        )

        inserted = trade_repo.create(trade)
        st.success(f"Trade guardado ✅ id={inserted.get('id')} | PnL={pnl:.2f} USD")
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
