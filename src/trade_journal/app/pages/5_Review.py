from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_sessions
from trade_journal.ai.checklist_a_plus import (
    BLOCK_KEYS,
    combine_scores,
    score_from_db_checklist_flag,
)
from trade_journal.ai.openai_review import TradeLite, build_session_payload, analyze_session_with_vision

st.set_page_config(page_title="Review", page_icon="🧠", layout="wide")
st.title("🧠 Review por sesión (A+ + pérdidas con evidencia)")

trade_repo, session_repo = get_repos()

# -----------------------------
# Helpers
# -----------------------------
def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _safe_str(x) -> str:
    return "" if x is None else str(x)

def _rows_to_tradelite(rows: List[dict]) -> List[TradeLite]:
    out: List[TradeLite] = []
    for r in rows or []:
        amount = _to_float(r.get("amount"), 0.0)
        pnl = _to_float(r.get("pnl"), 0.0)
        r_mult = 0.0
        if amount:
            r_mult = pnl / amount

        out.append(
            TradeLite(
                id=_safe_str(r.get("id")),
                trade_time=_safe_str(r.get("trade_time")),
                asset=_safe_str(r.get("asset")),
                timeframe=_safe_str(r.get("timeframe")),
                direction=_safe_str(r.get("direction")),
                outcome=_safe_str(r.get("outcome")),
                amount=amount,
                payout_pct=_to_float(r.get("payout_pct"), 0.0),
                pnl=pnl,
                r_mult=r_mult,
                emotion=_safe_str(r.get("emotion")),
                checklist_pass=r.get("checklist_pass"),
                screenshot_url=r.get("screenshot_url"),
                notes=r.get("notes"),
            )
        )
    return out

def _normalize_ai_reviews(ai: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Devuelve dict: trade_id -> {"blocks": {...}, ...}
    """
    out: Dict[str, Dict[str, Any]] = {}
    for item in (ai.get("loss_trade_reviews") or []):
        tid = str(item.get("trade_id") or "").strip()
        if not tid:
            continue
        blocks = item.get("blocks") or {}
        # normaliza claves faltantes
        norm_blocks = {k: blocks.get(k, "unclear") for k in BLOCK_KEYS}
        item["blocks"] = norm_blocks
        out[tid] = item
    return out

# -----------------------------
# UI: sesión
# -----------------------------
df_sessions = get_recent_sessions(limit=300)
if df_sessions.empty:
    st.info("No hay sesiones.")
    st.stop()

# nice labels
df_sessions = df_sessions.copy()
df_sessions["label"] = df_sessions.apply(
    lambda r: f"{r.get('start_time')} | id={r.get('id')}",
    axis=1,
)

sel = st.selectbox("Selecciona sesión", options=df_sessions["label"].tolist(), index=0)
sel_id = str(df_sessions.loc[df_sessions["label"] == sel, "id"].iloc[0])

c1, c2 = st.columns([1, 1])
with c1:
    use_ai = st.toggle("Usar IA con screenshots (solo pérdidas con screenshot_url)", value=True)
with c2:
    max_losses = st.number_input("Máx pérdidas con imagen a revisar", 1, 20, 10)

# -----------------------------
# Data por sesión
# -----------------------------
rows = trade_repo.list_by_session(sel_id, limit=10000) or []
if not rows:
    st.warning("Esta sesión no tiene trades.")
    st.stop()

trades = _rows_to_tradelite(rows)

# KPIs base
total = len(trades)
wins = sum(1 for t in trades if t.outcome == "WIN")
losses = sum(1 for t in trades if t.outcome == "LOSS")
ties = sum(1 for t in trades if t.outcome == "TIE")
pnl_total = sum(t.pnl for t in trades)
ev_r = (sum(t.r_mult for t in trades) / total) if total else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Trades", total)
k2.metric("W / L / T", f"{wins} / {losses} / {ties}")
k3.metric("PnL total", f"{pnl_total:.2f}")
k4.metric("EV (R)", f"{ev_r:.3f}")
k5.metric("Sesión", sel_id[:8] + "…")

st.divider()

# -----------------------------
# (2) Score A+ automático por trade
# -----------------------------
# default (solo DB flag)
score_map: Dict[str, Any] = {}
for t in trades:
    s = score_from_db_checklist_flag(t.checklist_pass)
    score_map[t.id] = s

ai_result: Optional[Dict[str, Any]] = None
ai_by_trade: Dict[str, Dict[str, Any]] = {}

if use_ai:
    # arma payload y llama IA (solo mira losses con imagen)
    session_meta = {"session_id": sel_id}
    payload = build_session_payload(
        session_meta=session_meta,
        trades=trades,
        max_losses_with_images=int(max_losses),
    )

    with st.spinner("Analizando pérdidas con screenshots (OpenAI)…"):
        ai_result = analyze_session_with_vision(session_payload=payload)

    ai_by_trade = _normalize_ai_reviews(ai_result)

    # reemplaza score para trades que vienen con bloques IA
    for tid, item in ai_by_trade.items():
        blocks = item.get("blocks") or {}
        # combina: preferir visión
        # (si no hay visión por alguna razón, cae al flag)
        for t in trades:
            if t.id == tid:
                score_map[tid] = combine_scores(db_flag=t.checklist_pass, vision_blocks=blocks)
                break

# -----------------------------
# Tabla con scores + foco en pérdidas
# -----------------------------
view_rows = []
for t in trades:
    s = score_map[t.id]
    view_rows.append(
        {
            "trade_time": t.trade_time,
            "asset": t.asset,
            "tf": t.timeframe,
            "dir": t.direction,
            "outcome": t.outcome,
            "pnl": t.pnl,
            "r_mult": round(t.r_mult, 3),
            "emotion": t.emotion,
            "checklist_pass_db": t.checklist_pass,
            "a_plus_score_0_5": s.score_0_5,
            "is_a_plus": s.is_a_plus,
            "has_screenshot": bool(t.screenshot_url),
            "trade_id": t.id,
        }
    )

df = pd.DataFrame(view_rows)
# orden por tiempo si viene ISO
try:
    df["_ts"] = pd.to_datetime(df["trade_time"], utc=True, errors="coerce")
    df = df.sort_values("_ts")
except Exception:
    pass

st.subheader("📌 Score A+ por trade (auto)")
st.dataframe(df.drop(columns=[c for c in ["_ts"] if c in df.columns]), use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Resumen A+ por sesión
# -----------------------------
a_plus_rate = float((df["is_a_plus"] == True).mean()) if len(df) else 0.0
avg_score = float(df["a_plus_score_0_5"].mean()) if len(df) else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("% A+ (binario)", f"{a_plus_rate*100:.1f}%")
c2.metric("Score medio (0–5)", f"{avg_score:.2f}")
c3.metric("Pérdidas con screenshot", int(((df["outcome"] == "LOSS") & (df["has_screenshot"] == True)).sum()))

# -----------------------------
# Profundización: pérdidas
# -----------------------------
st.subheader("🔍 Profundización en pérdidas")

df_loss = df[df["outcome"] == "LOSS"].copy()
if df_loss.empty:
    st.info("No hay pérdidas en esta sesión.")
    st.stop()

# ranking de peores trades por score
df_loss = df_loss.sort_values(["a_plus_score_0_5", "r_mult"], ascending=[True, True])

st.caption("Ordenado por menor score A+ y peor R-multiple.")
st.dataframe(df_loss, use_container_width=True, hide_index=True)

# -----------------------------
# Mostrar auditoría IA por trade LOSS (si existe)
# -----------------------------
if use_ai and ai_by_trade:
    st.divider()
    st.subheader("🧠 Auditoría IA por LOSS con evidencia")

    # pick trade id
    loss_ids = [r["trade_id"] for _, r in df_loss.iterrows() if r["trade_id"] in ai_by_trade]
    if not loss_ids:
        st.info("No hay pérdidas con screenshot_url válido para revisar.")
        st.stop()

    pick = st.selectbox("Selecciona trade LOSS con auditoría IA", options=loss_ids, index=0)
    item = ai_by_trade.get(pick, {})
    blocks = item.get("blocks") or {}

    st.markdown("### Checklist por bloques")
    cols = st.columns(5)
    for i, k in enumerate(BLOCK_KEYS):
        cols[i].metric(k, str(blocks.get(k, "unclear")))

    st.markdown("### Causa primaria")
    st.write(item.get("primary_cause", "—"))

    st.markdown("### Corrección (acción concreta)")
    st.write(item.get("correction", "—"))

    st.markdown("### Confianza")
    st.write(item.get("confidence_0_1", "—"))

    # Mostrar screenshot si existe
    # (buscamos en trades originales)
    t = next((x for x in trades if x.id == pick), None)
    if t and t.screenshot_url:
        st.markdown("### Screenshot")
        st.image(t.screenshot_url, use_container_width=True)

# -----------------------------
# Resumen IA de sesión (si existe)
# -----------------------------
if use_ai and ai_result:
    st.divider()
    st.subheader("🧾 Resumen IA de la sesión")
    st.markdown(ai_result.get("session_summary_md", "—"))
