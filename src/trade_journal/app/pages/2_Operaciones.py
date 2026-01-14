from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import os
import uuid

import streamlit as st
import requests

from trade_journal.app.utils import get_repos, get_recent_trades
from trade_journal.domain.models import Direction, Outcome
from trade_journal.infra.storage import GitHubPagesStorage

st.set_page_config(page_title="Operaciones", page_icon="🧾", layout="wide")
st.title("🧾 Operaciones")

LOCAL_TZ = ZoneInfo("America/Santiago")

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
EMOTIONS = ["Neutral", "Confiado", "Enfocado", "Ansioso", "Impulsivo", "Cansado", "Frustrado"]

# Segmentación (según tu BD)
SETUPS = ["", "Breakout", "Reversal", "Trend", "Range", "News"]  # "" -> None
MARKET_REGIMES = ["", "Trend", "Range", "Volatile", "LowVol"]   # "" -> None
QUALITY_GRADES = ["", "A", "B", "C", "D"]                       # "" -> None


# ---------------------------------------------------------
# Storage (GitHub Pages) - escalable (puedes cambiar proveedor luego)
# ---------------------------------------------------------
def _build_evidence_storage() -> GitHubPagesStorage | None:
    token = os.getenv("GITHUB_TOKEN")
    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_REPO")
    pages_base = os.getenv("GITHUB_PAGES_BASE")

    if not token or not owner or not repo or not pages_base:
        return None

    return GitHubPagesStorage(
        owner=owner,
        repo=repo,
        token=token,
        pages_base_url=pages_base,
    )


EVIDENCE_STORAGE = _build_evidence_storage()


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
    if hasattr(session_repo, "list_open"):
        rows = session_repo.list_open()
    else:
        rows = session_repo.list_recent(limit=50)

    rows = rows or []
    open_rows = [r for r in rows if r.get("end_time") in (None, "", "null")]

    if len(open_rows) == 0:
        return None, "No hay sesión abierta (debes iniciar desde Telegram)."

    if len(open_rows) > 1:
        return None, f"⚠️ Hay {len(open_rows)} sesiones abiertas. Debe existir solo 1. Cierra desde el bot."

    open_sess = open_rows[0]
    sid = open_sess.get("id")
    if not sid:
        return None, "Sesión abierta sin ID (inconsistencia)."

    today_local = datetime.now(LOCAL_TZ).date()
    sess_date_local = _session_local_date(open_sess)

    if sess_date_local != today_local:
        return None, (
            f"Hay una sesión abierta pero es de otro día (sesión: {sess_date_local}, hoy: {today_local}). "
            "Debe cerrarse automáticamente o vía bot."
        )

    start_time = open_sess.get("start_time") or "?"
    return str(sid), f"Sesión abierta ✅ (start: {start_time} | día local: {sess_date_local})"


def _normalize_insert_result(result):
    """
    Supabase/PostgREST a veces devuelve:
      - dict
      - list[dict]
    Normalizamos a dict.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        return result[0]
    return {}  # fallback


def _safe_repo_insert_trade(trade_repo, payload: dict):
    """
    Inserta trade de forma robusta.
    Preferimos enviar un dict con nombres EXACTOS de columnas de la BD
    para evitar que TradeCreate/serialización "pierda" session_id.
    """
    # 1) Si el repo soporta insertar dict directo
    if hasattr(trade_repo, "create_dict"):
        return trade_repo.create_dict(payload)

    # 2) Si expone cliente supabase interno
    if hasattr(trade_repo, "sb") and hasattr(trade_repo.sb, "insert"):
        return trade_repo.sb.insert("trades", payload)

    # 3) Último recurso: create() con dict (muchos repos lo aceptan)
    return trade_repo.create(payload)


trade_repo, session_repo = get_repos()

# --- Estado de sesión (bloquea guardado si no está OK) ---
open_session_id, session_status = get_open_session(session_repo)

if open_session_id:
    st.success(session_status)
else:
    st.error(session_status)

# ---------------------------------------------------------
# Evidencia (FUERA del form) => evita que quede "pendiente" hasta submit
# ---------------------------------------------------------
st.markdown("### 🗂️ Evidencia (opcional)")

with st.container():
    if EVIDENCE_STORAGE is None:
        st.info(
            "Subida rápida deshabilitada: configura tu .env con "
            "GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO, GITHUB_PAGES_BASE."
        )
        uploaded_evidence = None
    else:
        uploaded_evidence = st.file_uploader(
            "Subir screenshot (png/jpg/webp)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="evidence_file",
        )

    overwrite_evidence = st.checkbox(
        "Reemplazar imagen si ya existe (corregir)",
        value=False,
        key="overwrite_evidence",
    )

    if uploaded_evidence is not None:
        st.image(uploaded_evidence, caption="Preview evidencia", width=400)

    can_upload = bool(EVIDENCE_STORAGE is not None and uploaded_evidence is not None)

    c_up1, c_up2 = st.columns([1, 3])
    with c_up1:
        if st.button("⬆️ Subir evidencia", disabled=not can_upload):
            try:
                trade_date_local = datetime.now(LOCAL_TZ).date()

                path = EVIDENCE_STORAGE.build_path(
                    trade_date=trade_date_local,
                    asset=st.session_state.get("asset", ASSETS[0]),
                    filename=uploaded_evidence.name,
                )

                url = EVIDENCE_STORAGE.upload(
                    path=path,
                    content=uploaded_evidence.getvalue(),
                    overwrite=bool(overwrite_evidence),
                )

                st.session_state["screenshot_url"] = url
                st.success("Evidencia subida ✅")
                st.info(f"URL copiada al formulario. Ahora puedes guardar el trade.")
                st.code(url, language="text")
                # Forzar rerun para actualizar el formulario
                st.rerun()

            except Exception as e:
                st.error(f"No se pudo subir evidencia: {e}")

    with c_up2:
        st.caption(
            "Tip: sube la evidencia primero y luego guarda el trade. "
            "El campo 'Screenshot/Link evidencia' se llenará automáticamente."
        )

# Mostrar URL actual si existe
if "screenshot_url" in st.session_state and st.session_state["screenshot_url"]:
    st.success("📎 Evidencia cargada (copia la URL para pegarla en el formulario):")

    col_url, col_btn = st.columns([5, 1])
    with col_url:
        st.text_input(
            "URL de la evidencia",
            value=st.session_state["screenshot_url"],
            key="display_screenshot_url",
            disabled=False,
            label_visibility="collapsed",
            help="Copia esta URL (Ctrl+C / Cmd+C) y pégala en el campo 'Screenshot/Link evidencia' del formulario"
        )
    with col_btn:
        if st.button("🗑️ Limpiar", use_container_width=True):
            del st.session_state["screenshot_url"]
            st.rerun()

st.divider()
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
        checklist_pass = st.checkbox("Checklist PASS", value=True, key="checklist_pass")  # ✅ columna BD

    st.markdown("### 🧩 Contexto / Calidad (opcional)")
    s1, s2, s3 = st.columns([1.2, 1.2, 0.8])
    with s1:
        setup_tag = st.selectbox("Setup", options=SETUPS, index=0, key="setup_tag")
    with s2:
        market_regime = st.selectbox("Régimen de mercado", options=MARKET_REGIMES, index=0, key="market_regime")
    with s3:
        quality_grade = st.selectbox("Calidad", options=QUALITY_GRADES, index=0, key="quality_grade")

    # Campo final que se guarda a la BD
    screenshot_url = st.text_input(
        "Screenshot/Link evidencia",
        value="",
        key="screenshot_url_input",
        placeholder="Pega aquí la URL de la evidencia si subiste una imagen arriba ⬆️",
        help="Si subiste evidencia arriba, copia y pega la URL aquí",
    )

    notes = st.text_area("Notas (opcional)", height=90, key="notes")

    submitted = st.form_submit_button("Guardar trade", disabled=(open_session_id is None))

# ---------------------------------------------------------
# Guardado (ENFORCE session)
# ---------------------------------------------------------
if submitted:
    if open_session_id is None:
        st.error("No se puede guardar: no hay una sesión abierta válida para hoy (America/Santiago).")
    else:
        try:
            uuid.UUID(str(open_session_id))  # valida UUID real

            pnl = calculate_pnl(float(amount), float(payout), str(outcome))

            payload = {
                "trade_time": datetime.now(timezone.utc).isoformat(),
                "asset": str(asset).strip(),
                "timeframe": str(timeframe).strip(),
                "amount": float(amount),
                "direction": Direction(direction).value,
                "outcome": Outcome(outcome).value,
                "payout_pct": float(payout),
                "pnl": float(pnl),
                "emotion": str(emotion).strip(),
                "notes": empty_to_none(notes),

                "setup_tag": empty_to_none(setup_tag),
                "market_regime": empty_to_none(market_regime),
                "quality_grade": empty_to_none(quality_grade),
                "checklist_pass": bool(checklist_pass),
                # Priorizar el valor del campo input sobre el session_state
                "screenshot_url": empty_to_none(screenshot_url or st.session_state.get("screenshot_url", "")),

                # ✅ CRÍTICO
                "session_id": str(open_session_id),
            }

            result = _safe_repo_insert_trade(trade_repo, payload)
            inserted = _normalize_insert_result(result)

            trade_id = inserted.get("id", "—")

            # Limpiar el screenshot_url del session_state para que no se reutilice
            if "screenshot_url" in st.session_state:
                del st.session_state["screenshot_url"]

            st.cache_data.clear()

            # Mostrar mensaje de éxito
            st.success(f"Trade guardado ✅ id={trade_id} | PnL={pnl:.2f} USD | session_id={open_session_id}")

            # Forzar rerun para limpiar el formulario completamente
            st.rerun()
        except requests.HTTPError as e:
            detail = ""
            if e.response is not None:
                try:
                    detail = e.response.text
                except Exception:
                    detail = str(e)
            st.error(f"No se pudo guardar (HTTP): {detail}")

        except Exception as e:
            detail = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    detail = resp.text
                except Exception:
                    detail = str(e)
            else:
                detail = str(e)
            st.error(f"No se pudo guardar: {detail}")

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
