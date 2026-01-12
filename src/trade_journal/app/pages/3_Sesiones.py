from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_sessions

st.set_page_config(page_title="Sesiones", page_icon="⏱️", layout="wide")
st.title("⏱️ Sesiones")

_, session_repo = get_repos()


# ----------------------------
# Helpers
# ----------------------------
def _parse_dt(dt_val: Any) -> datetime | None:
    """Convierte str/datetime a datetime tz-aware (UTC)."""
    if not dt_val:
        return None
    if isinstance(dt_val, datetime):
        return dt_val if dt_val.tzinfo else dt_val.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(dt_val).replace("Z", "+00:00"))
    except Exception:
        return None


def _is_nullish(v: Any) -> bool:
    return v is None or str(v).strip().lower() in ("", "null", "none")


def fetch_open_sessions(limit: int = 200) -> list[dict]:
    """
    Busca sesiones abiertas (end_time is null) desde BD.
    Ideal sería tener session_repo.list_open() con filter end_time=is.null,
    pero lo resolvemos aquí para no tocar repositorios.
    """
    rows = session_repo.list_recent(limit=limit) or []
    return [r for r in rows if _is_nullish(r.get("end_time"))]


def reconcile_active_session_state() -> tuple[str | None, datetime | None, str | None]:
    """
    Re-hidrata estado desde BD:
      - Si hay 1 sesión abierta -> setea session_state
      - Si hay 0 -> limpia session_state
      - Si hay >1 -> error (estado inválido)
    """
    try:
        open_rows = fetch_open_sessions(limit=300)

        if len(open_rows) == 0:
            st.session_state.active_session_id = None
            st.session_state.active_session_start = None
            return None, None, None

        if len(open_rows) > 1:
            # Estado inválido: más de una abierta
            st.session_state.active_session_id = None
            st.session_state.active_session_start = None
            return None, None, f"⚠️ Hay {len(open_rows)} sesiones abiertas en BD. Debe existir solo 1."

        row = open_rows[0]
        sid = row.get("id")
        stt = _parse_dt(row.get("start_time"))

        st.session_state.active_session_id = str(sid) if sid else None
        st.session_state.active_session_start = stt
        return st.session_state.active_session_id, st.session_state.active_session_start, None

    except Exception as e:
        # Evita pantallazo rojo
        st.session_state.active_session_id = None
        st.session_state.active_session_start = None
        return None, None, f"No pude leer sesiones desde BD: {e}"


# ----------------------------
# Init session_state (si faltan keys)
# ----------------------------
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "active_session_start" not in st.session_state:
    st.session_state.active_session_start = None


# ----------------------------
# Reconcile (siempre al cargar)
# ----------------------------
active_id, active_start, reconcile_err = reconcile_active_session_state()
if reconcile_err:
    st.error(reconcile_err)


# ----------------------------
# UI Controls
# ----------------------------
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.session_state.active_session_id is None:
        if st.button("▶️ Start sesión", use_container_width=True):
            try:
                now = datetime.now(timezone.utc)
                row = session_repo.start(start_time=now)
                # Re-hidrata desde BD para quedar consistente
                st.cache_data.clear()
                reconcile_active_session_state()
                st.success(f"Sesión iniciada ✅ id={row.get('id')}")
            except Exception as e:
                st.error(f"No pude iniciar sesión: {e}")
    else:
        st.info("Sesión activa ✅ (BD)")

with col2:
    if st.session_state.active_session_id is not None:
        if st.button("⏹️ Stop sesión", use_container_width=True):
            try:
                now = datetime.now(timezone.utc)

                # IMPORTANTE: usa start_time desde BD (rehidratado), no uno “guardado” en memoria vieja
                start = st.session_state.active_session_start
                if start is None:
                    # fallback: reconsultar por si acaso
                    open_rows = fetch_open_sessions(limit=300)
                    if len(open_rows) == 1:
                        start = _parse_dt(open_rows[0].get("start_time"))

                if start is None:
                    st.error("No pude determinar start_time de la sesión abierta (BD).")
                else:
                    duration_min = (now - start).total_seconds() / 60.0
                    row = session_repo.stop(
                        session_id=st.session_state.active_session_id,
                        end_time=now,
                        duration_min=float(duration_min),
                    )
                    st.cache_data.clear()
                    reconcile_active_session_state()
                    st.success(f"Sesión finalizada ✅ ({duration_min:.1f} min) id={row.get('id', '—')}")
            except Exception as e:
                st.error(f"No pude cerrar la sesión: {e}")

with col3:
    if st.session_state.active_session_id is not None and st.session_state.active_session_start is not None:
        elapsed = (datetime.now(timezone.utc) - st.session_state.active_session_start).total_seconds() / 60.0
        st.metric("Minutos (en curso)", f"{elapsed:.1f}")
    elif st.session_state.active_session_id is not None:
        st.caption("Sesión activa pero start_time no disponible (revisa BD).")


# ----------------------------
# Recent sessions table
# ----------------------------
st.divider()
st.subheader("Sesiones recientes")

try:
    df = get_recent_sessions(limit=500)
    if df.empty:
        st.write("—")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"No pude cargar sesiones recientes: {e}")
