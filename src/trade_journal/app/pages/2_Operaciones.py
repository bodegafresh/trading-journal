from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import os
import uuid

import streamlit as st
import streamlit.components.v1 as components
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
# Sección de Evidencia (FUERA del form para subida automática)
# ---------------------------------------------------------
st.markdown("### 🗂️ Evidencia (opcional)")

uploaded_evidence = None
if EVIDENCE_STORAGE is not None:
    uploaded_evidence = st.file_uploader(
        "Subir screenshot (se sube automáticamente al seleccionar)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="evidence_file_uploader",
        help="Selecciona una imagen y se subirá automáticamente"
    )

    # SUBIDA AUTOMÁTICA INMEDIATA (fuera del form)
    if uploaded_evidence is not None:
        current_file_id = f"{uploaded_evidence.name}_{uploaded_evidence.size}"
        last_uploaded_id = st.session_state.get("last_uploaded_file_id", "")

        if current_file_id != last_uploaded_id:
            try:
                with st.spinner("Subiendo imagen..."):
                    trade_date_local = datetime.now(LOCAL_TZ).date()
                    current_asset = st.session_state.get("asset", ASSETS[0])

                    path = EVIDENCE_STORAGE.build_path(
                        trade_date=trade_date_local,
                        asset=current_asset,
                        filename=uploaded_evidence.name,
                    )

                    url = EVIDENCE_STORAGE.upload(
                        path=path,
                        content=uploaded_evidence.getvalue(),
                        overwrite=True,
                    )

                    st.session_state["screenshot_url"] = url
                    st.session_state["last_uploaded_file_id"] = current_file_id

                    st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.info(f"📎 Ya subida: {uploaded_evidence.name}")

    # Mostrar URL si está disponible
    if "screenshot_url" in st.session_state and st.session_state["screenshot_url"]:
        url_to_fill = st.session_state["screenshot_url"]
        st.success(f"🔗 URL guardada: {url_to_fill}")

        # Auto-llenar el campo del formulario SIEMPRE que haya URL
        # Ejecutar el JavaScript para llenar el campo
        auto_fill_js = f"""
            <script>
            setTimeout(function() {{
                const input = window.parent.document.querySelector('input[aria-label="Screenshot/Link evidencia"]');
                if (input) {{
                    input.value = "{url_to_fill}";
                    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}
            }}, 300);
            </script>
        """
        components.html(auto_fill_js, height=0)

    # Preview de la imagen
    if uploaded_evidence is not None:
        with st.expander("👁️ Ver preview", expanded=False):
            st.image(uploaded_evidence, width=300)
else:
    st.info("Subida deshabilitada: configura GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO, GITHUB_PAGES_BASE en .env")

# ---------------------------------------------------------
# Formulario
# ---------------------------------------------------------
with st.form("new_trade", clear_on_submit=True):
    st.markdown("### 📊 Datos de la Operación")
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
        checklist_pass = st.checkbox("Checklist PASS", value=True, key="checklist_pass")

    st.markdown("### 🧩 Contexto / Calidad (opcional)")
    s1, s2, s3 = st.columns([1.2, 1.2, 0.8])
    with s1:
        setup_tag = st.selectbox("Setup", options=SETUPS, index=0, key="setup_tag")
    with s2:
        market_regime = st.selectbox("Régimen de mercado", options=MARKET_REGIMES, index=0, key="market_regime")
    with s3:
        quality_grade = st.selectbox("Calidad", options=QUALITY_GRADES, index=0, key="quality_grade")

    # Campo de screenshot - se llenará automáticamente por JavaScript
    screenshot_url = st.text_input(
        "Screenshot/Link evidencia",
        key="screenshot_url_input",
        placeholder="Se completa automáticamente al subir imagen arriba ⬆️ o pega una URL manualmente",
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

            # Limpiar el screenshot_url y flags relacionados del session_state
            if "screenshot_url" in st.session_state:
                del st.session_state["screenshot_url"]
            if "last_uploaded_file_id" in st.session_state:
                del st.session_state["last_uploaded_file_id"]

            st.cache_data.clear()

            # Mostrar mensaje de éxito
            st.success(f"Trade guardado ✅ id={trade_id} | PnL={pnl:.2f} USD | session_id={open_session_id}")

            # JavaScript para limpiar el campo del formulario antes del rerun
            clear_field_js = """
                <script>
                setTimeout(function() {
                    const input = window.parent.document.querySelector('input[aria-label="Screenshot/Link evidencia"]');
                    if (input) {
                        input.value = "";
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        input.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }, 100);
                </script>
            """
            components.html(clear_field_js, height=0)

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
