from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_trades
from trade_journal.domain.models import Direction, Outcome, TradeCreate

st.set_page_config(page_title="Operaciones", page_icon="🧾", layout="wide")
st.title("🧾 Operaciones")

ASSETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "EUR/JPY", "EUR/GBP", "BTC/USD",
    "Asia comoposites", "Euro composites", "Compound Index",
]
TIMEFRAMES = ["1m", "5m"]
EMOTIONS = ["Neutral", "Confiado", "Enfocado", "Ansioso", "Impulsivo", "Cansado", "Frustrado"]

# Segmentación (según tu BD nueva)
SETUPS = ["", "Breakout", "Reversal", "Trend", "Range", "News"]  # "" -> None
MARKET_REGIMES = ["", "Trend", "Range", "Volatile", "LowVol"]   # "" -> None
QUALITY_GRADES = ["", "A", "B", "C", "D"]                       # "" -> None


def calculate_pnl(amount: float, payout_pct: float, outcome: str) -> float:
    if outcome == Outcome.WIN.value:
        return round(amount * (payout_pct / 100.0), 2)
    if outcome == Outcome.LOSS.value:
        return round(-amount, 2)
    return 0.0


def empty_to_none(x) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


trade_repo, session_repo = get_repos()

# Sesiones (opcional)
sessions_opts: list[tuple[str, str | None]] = [("(Sin sesión)", None)]
try:
    sess_rows = session_repo.list_recent(limit=200) or []
    for r in sess_rows:
        sid = r.get("id")
        label = r.get("start_time") or sid
        if sid:
            sessions_opts.append((str(label), str(sid)))
except Exception:
    pass

with st.form("new_trade", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        asset = st.selectbox("Activo", options=ASSETS, index=0, key="asset")
        timeframe = st.selectbox("Timeframe", options=TIMEFRAMES, index=0, key="timeframe")

    with c2:
        amount = st.number_input("Monto (USD)", min_value=0.0, value=10.0, step=1.0, key="amount")
        payout = st.number_input("Payout %", min_value=0.0, value=80.0, step=1.0, key="payout")

    with c3:
        direction = st.selectbox("Dirección", options=[d.value for d in Direction], index=0, key="direction")
        outcome = st.selectbox("Resultado", options=[o.value for o in Outcome], index=0, key="outcome")

    with c4:
        emotion = st.selectbox("Emoción", options=EMOTIONS, index=0, key="emotion")
        checklist_passed = st.checkbox("Checklist PASS", value=True, key="checklist_passed")

    st.markdown("### 🧩 Contexto / Calidad (opcional)")
    s1, s2, s3 = st.columns([1.2, 1.2, 0.8])
    with s1:
        setup_tag = st.selectbox("Setup", options=SETUPS, index=0, key="setup_tag")
    with s2:
        market_regime = st.selectbox("Régimen de mercado", options=MARKET_REGIMES, index=0, key="market_regime")
    with s3:
        quality_grade = st.selectbox("Calidad", options=QUALITY_GRADES, index=0, key="quality_grade")

    st.markdown("### 🗂️ Sesión / Evidencia (opcional)")
    e1, e2 = st.columns([1, 3])
    with e1:
        session_choice = st.selectbox("Sesión", options=sessions_opts, index=0, key="session_choice")
        session_id = session_choice[1]
    with e2:
        screenshot_url = st.text_input("Screenshot/Link evidencia", value="", key="screenshot_url")

    notes = st.text_area("Notas (opcional)", height=90, key="notes")

    submitted = st.form_submit_button("Guardar trade")

if submitted:
    try:
        pnl = calculate_pnl(float(amount), float(payout), str(outcome))

        trade = TradeCreate(
            trade_time=datetime.now(timezone.utc),

            asset=str(asset).strip(),
            timeframe=str(timeframe).strip(),
            amount=float(amount),
            direction=Direction(direction),
            outcome=Outcome(outcome),
            payout_pct=float(payout),
            pnl=float(pnl),
            emotion=str(emotion).strip(),
            notes=empty_to_none(notes),

            # Nuevos campos
            setup_tag=empty_to_none(setup_tag),
            market_regime=empty_to_none(market_regime),
            quality_grade=empty_to_none(quality_grade),
            checklist_passed=bool(checklist_passed),
            session_id=session_id,
            screenshot_url=empty_to_none(screenshot_url),
        )

        inserted = trade_repo.create(trade)
        st.success(f"Trade guardado ✅ id={inserted.get('id')} | PnL={pnl:.2f} USD")
        st.cache_data.clear()
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
