from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from trade_journal.data.supabase_client import load_supabase_from_env
from trade_journal.data.repositories import SessionRepository, TradeRepository

st.set_page_config(page_title="Trade Journal Pro", page_icon="📈", layout="wide")


def bootstrap():
    load_dotenv()
    sb = load_supabase_from_env()
    return TradeRepository(sb), SessionRepository(sb)


def main():
    st.title("📈 Trade Journal Pro")
    st.caption("Streamlit + Plotly + Supabase")

    try:
        trade_repo, session_repo = bootstrap()
    except Exception as e:
        st.error(f"No se pudo inicializar Supabase: {e}")
        st.info("Revisa tu .env con SUPABASE_URL y SUPABASE_KEY.")
        st.stop()

    st.success("Conectado a Supabase ✅")

    col1, col2 = st.columns(2)
    with col1:
        st.write("➡️ Usa el menú de la izquierda para navegar por las páginas.")
    with col2:
        st.write("Tip: en **Importar** puedes subir tu CSV del script antiguo.")


if __name__ == "__main__":
    main()
