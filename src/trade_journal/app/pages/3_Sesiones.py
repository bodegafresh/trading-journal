from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_sessions

st.set_page_config(page_title="Sesiones", page_icon="⏱️", layout="wide")

st.title("⏱️ Sesiones")

_, session_repo = get_repos()

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "active_session_start" not in st.session_state:
    st.session_state.active_session_start = None

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.session_state.active_session_id is None:
        if st.button("▶️ Start sesión", use_container_width=True):
            now = datetime.now(timezone.utc)
            row = session_repo.start(start_time=now)
            st.session_state.active_session_id = row["id"]
            st.session_state.active_session_start = now
            st.success("Sesión iniciada ✅")
            st.cache_data.clear()
    else:
        st.info("Sesión activa ✅")

with col2:
    if st.session_state.active_session_id is not None:
        if st.button("⏹️ Stop sesión", use_container_width=True):
            now = datetime.now(timezone.utc)
            start = st.session_state.active_session_start
            duration_min = (now - start).total_seconds() / 60.0
            row = session_repo.stop(
                session_id=st.session_state.active_session_id,
                end_time=now,
                duration_min=float(duration_min),
            )
            st.session_state.active_session_id = None
            st.session_state.active_session_start = None
            st.success(f"Sesión finalizada ✅ ({duration_min:.1f} min)")
            st.cache_data.clear()

with col3:
    if st.session_state.active_session_id is not None:
        start = st.session_state.active_session_start
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() / 60.0
        st.metric("Minutos (en curso)", f"{elapsed:.1f}")

st.divider()

st.subheader("Sesiones recientes")
df = get_recent_sessions()
if df.empty:
    st.write("—")
else:
    st.dataframe(df, use_container_width=True, hide_index=True)
