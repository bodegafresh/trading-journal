from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_trades
from trade_journal.domain.models import Direction, Outcome, TradeCreate

st.set_page_config(page_title="Operaciones", page_icon="🧾", layout="wide")
st.title("🧾 Operaciones")

LOCAL_TZ = ZoneInfo("America/Santiago")

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


def _parse_dt(dt_val) -> datetime | None:
    """Parse robusto a datetime aware (UTC) si viene string o datetime."""
    if not dt_val:
        return None
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val
    try:
        # strings tipo ISO "2025-10-05T10:00:00+00:00"
        return datetime.fromisoformat(str(dt_val).replace("Z", "+00:00"))
    except Exception:
        return None


def _session_local_date(session_row: dict) -> datetime.date | None:
    stt = _parse_dt(session_row.get("start_time"))
    if not stt:
        return None
    return stt.astimezone(LOCAL_TZ).date()


def get_open_session(session_repo) -> tuple[str | None, str]:
    """
    Devuelve (session_id, status_message).
    Reglas:
      - debe existir EXACTAMENTE 1 sesión abierta (end_time is null)
      - esa sesión debe corresponder al día local actual (America/Santiago)
    """
    # Traer sesiones recientes y filtrar abiertas.
    # Si tu repo tiene un método específico para abiertas, lo usamos.
    rows = None
    if hasattr(session_repo, "list_open"):
        rows = session_repo.list_open()  # ideal si existe
    else:
        rows = session_repo.list_recent(limit=50)

    rows = rows or []
    open_rows = [r for r in rows if r.get("end_time") in (None, "", "null")]

    if len(open_rows) == 0:
        return None, "No hay sesión abierta (debes iniciar desde Telegram)."

    # Si hay más de una abierta: estado inválido (y hay que arreglarlo desde bot/BD)
    if len(open_rows) > 1:
        return None, f"⚠️ Hay {len(open_rows)} sesiones abiertas. Debe existir solo 1. Cierra desde el bot."

    open_sess = open_rows[0]
    sid = open_sess.get("id")
    if not sid:
        return None, "Sesión abierta sin ID (inconsistencia)."

    today_local = datetime.now(LOCAL_TZ).date()
    sess_date_local = _session_local_date(open_sess)

    if sess_date_local != today_local:
        # Según tus reglas: si pasó medianoche, debería auto-cerrarse.
        # Aún no implementamos el auto-close, así que en la app lo tratamos como inválido.
        return None, (
            f"Hay una sesión abierta pero es de otro día (sesión: {sess_date_local}, hoy: {today_local}). "
            "Debe cerrarse automáticamente o vía bot."
        )

    start_time = open_sess.get("start_time") or "?"
    return str(sid), f"Sesión abierta ✅ (start: {start_time} | día local: {sess_date_local})"


trade_repo, session_repo = get_repos()

# --- Estado de sesión (bloquea guardado si no está OK) ---
open_session_id, session_status = get_open_session(session_repo)

if open_session_id:
    st.success(session_status)
else:
    st.error(session_status)

# ---------------------------------------------------------
# Formulario
# ---------------------------------------------------------
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

    st.markdown("### 🗂️ Evidencia (opcional)")
    screenshot_url = st.text_input("Screenshot/Link evidencia", value="", key="screenshot_url")

    notes = st.text_area("Notas (opcional)", height=90, key="notes")

    # IMPORTANTE: si no hay sesión abierta válida, deshabilitamos el botón
    submitted = st.form_submit_button("Guardar trade", disabled=(open_session_id is None))

# ---------------------------------------------------------
# Guardado (enforce session)
# ---------------------------------------------------------
if submitted:
    if open_session_id is None:
        st.error("No se puede guardar: no hay una sesión abierta válida para hoy (America/Santiago).")
    else:
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
                session_id=str(open_session_id),  # ✅ forzado a la sesión abierta
                screenshot_url=empty_to_none(screenshot_url),
            )

            inserted = trade_repo.create(trade)
            st.success(f"Trade guardado ✅ id={inserted.get('id')} | PnL={pnl:.2f} USD | session_id={open_session_id}")
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
