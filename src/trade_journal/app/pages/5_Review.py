from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_sessions, get_review_repo
from trade_journal.ai.openai_review import (
    TradeLite,
    build_session_payload,
    analyze_session_with_vision,
    normalize_review,
)

LOCAL_TZ = pytz.timezone("America/Santiago")

st.set_page_config(page_title="Review (IA)", page_icon="🧠", layout="wide")
st.title("🧠 Review (IA) — por sesión")


# ----------------------------
# Helpers
# ----------------------------
def _calc_r_mult(pnl: float, amount: float) -> float:
    if amount == 0:
        return 0.0
    return float(pnl) / float(amount)


def _is_image_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower().split("?")[0]
    return u.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))


def _normalize_bool(x) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "t", "yes", "y"):
        return True
    if s in ("false", "0", "f", "no", "n"):
        return False
    return None


def _to_dt(x) -> datetime:
    """
    Normaliza a datetime.
    Acepta datetime / pandas.Timestamp / string ISO.
    """
    if x is None:
        return datetime.now()
    if isinstance(x, datetime):
        return x
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return datetime.now()


def _ensure_tz(dt: datetime, tz) -> datetime:
    """
    Devuelve datetime tz-aware en `tz`.
    - Si dt es naive, lo localiza.
    - Si dt es aware, lo convierte.
    """
    if dt.tzinfo is None:
        return tz.localize(dt)
    return dt.astimezone(tz)


def _weekly_window_start(now_local: Any) -> datetime:
    """
    Retorna inicio de semana (lunes 00:00) en la tz local.
    NO usa .to_pydatetime() (evita el error del screenshot).
    """
    dt = _ensure_tz(_to_dt(now_local), LOCAL_TZ)
    start = dt - timedelta(days=dt.weekday())
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    return start


def _score_proxy_from_user_checklist(trades_df: pd.DataFrame) -> pd.Series:
    """
    Proxy temporal (sin BD):
      - checklist_pass=True => score 5
      - checklist_pass=False => score 2
      - missing => 0
    """
    if "checklist_pass" not in trades_df.columns:
        return pd.Series([0] * len(trades_df), index=trades_df.index)

    def f(v):
        b = _normalize_bool(v)
        if b is True:
            return 5
        if b is False:
            return 2
        return 0

    return trades_df["checklist_pass"].map(f).astype(int)


# ----------------------------
# Load repos + sessions
# ----------------------------
trade_repo, session_repo = get_repos()
review_repo = get_review_repo()
df_sessions = get_recent_sessions(limit=500)

if df_sessions.empty:
    st.info("No hay sesiones aún.")
    st.stop()


def _label_session(r: pd.Series) -> str:
    sid = r.get("id", "—")
    stt = r.get("start_time", "—")
    end = r.get("end_time", "—")
    return f"{sid} | start={stt} | end={end}"


session_rows = df_sessions.to_dict("records")
labels = [_label_session(pd.Series(r)) for r in session_rows]
sel = st.selectbox("Selecciona sesión", options=list(range(len(labels))), format_func=lambda i: labels[i])

session = session_rows[int(sel)]
session_id = str(session.get("id"))

tabs = st.tabs(["📌 Sesión", "📊 Semanal (5 números)"])


# =========================================================
# TAB 1: SESIÓN
# =========================================================
with tabs[0]:
    rows = trade_repo.list_by_session(session_id=session_id, limit=10000) or []
    df = pd.DataFrame(rows)

    if df.empty:
        st.warning("Esta sesión no tiene trades asociados (session_id) en `trades`.")
        st.stop()

    # Normaliza trade_time a local sin tz
    if "trade_time" in df.columns:
        dt = pd.to_datetime(df["trade_time"], utc=True, errors="coerce", format="mixed")
        df["trade_time"] = dt.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    # compat
    if "checklist_pass" not in df.columns and "checklist_passed" in df.columns:
        df["checklist_pass"] = df["checklist_passed"]

    # r_mult
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce").fillna(0.0)
    df["pnl"] = pd.to_numeric(df.get("pnl"), errors="coerce").fillna(0.0)
    df["r_mult"] = (df["pnl"] / df["amount"].replace({0: np.nan})).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # KPIs sesión
    total = len(df)
    wins = int((df["outcome"] == "WIN").sum())
    losses = int((df["outcome"] == "LOSS").sum())
    ties = int((df["outcome"] == "TIE").sum())
    ev_r = float(df["r_mult"].mean()) if total else 0.0
    pnl_total = float(df["pnl"].sum()) if total else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trades", total)
    k2.metric("W / L / T", f"{wins} / {losses} / {ties}")
    k3.metric("EV (R)", f"{ev_r:.3f}")
    k4.metric("PnL total", f"{pnl_total:.2f}")

    st.divider()

    # Construye TradeLite list
    trades: List[TradeLite] = []
    for _, r in df.sort_values("trade_time").iterrows():
        trades.append(
            TradeLite(
                id=str(r.get("id")),
                trade_time=str(r.get("trade_time")),
                asset=str(r.get("asset", "")),
                timeframe=str(r.get("timeframe", "")),
                direction=str(r.get("direction", "")),
                outcome=str(r.get("outcome", "")),
                amount=float(r.get("amount", 0.0)),
                payout_pct=float(r.get("payout_pct", 0.0)),
                pnl=float(r.get("pnl", 0.0)),
                r_mult=float(r.get("r_mult", 0.0)),
                emotion=str(r.get("emotion", "")),
                checklist_pass=_normalize_bool(r.get("checklist_pass")),
                screenshot_url=(str(r.get("screenshot_url")).strip() if r.get("screenshot_url") else None),
                notes=(str(r.get("notes")).strip() if r.get("notes") else None),
            )
        )

    # ----------------------------
    # IA review
    # ----------------------------
    max_imgs = st.slider("Máx operaciones con imagen a auditar", 1, 30, 15)
    prioritize = st.checkbox("Priorizar pérdidas en el análisis", value=True,
                            help="Si está activado, analiza primero las pérdidas, luego ganancias y empates")
    use_cached = st.checkbox("Usar revisión guardada si existe", value=True)
    run = st.button("🧠 Analizar sesión con IA", type="primary", use_container_width=True)

    cache_key = f"review_ai::{session_id}"
    cached_db = review_repo.get_session_review(session_id)

    if run:
        with st.spinner("Analizando con IA..."):
            session_meta = dict(session or {})
            session_meta["scope"] = "session"
            payload = build_session_payload(
                session_meta=session_meta,
                trades=trades,
                max_trades_with_images=int(max_imgs),
                prioritize_losses=prioritize,
            )
            review = analyze_session_with_vision(session_payload=payload, max_output_tokens=2500)
            review_repo.upsert_session_review(session_id, payload, review)
            st.session_state[cache_key] = {"payload": payload, "review": review}
    elif use_cached and cached_db:
        st.session_state[cache_key] = {
            "payload": cached_db.get("payload"),
            "review": normalize_review(cached_db.get("review") or {}),
        }

    cached = st.session_state.get(cache_key)

    if cached:
        review = normalize_review(cached["review"] or {})

        # ---- Usage / tokens / meta ----
        meta = review.get("_meta") or {}
        usage = (meta.get("usage") or {})
        st.caption(
            "IA usage — "
            f"model={meta.get('model')} | prompt_version={meta.get('prompt_version')} | payload_hash={meta.get('payload_hash')} "
            f"| total_tokens={usage.get('total_tokens')} | prompt_tokens={usage.get('prompt_tokens')} | completion_tokens={usage.get('completion_tokens')} "
            f"| est_text_prompt_tokens={meta.get('estimated_text_prompt_tokens')}"
        )

        st.subheader("🧾 Resumen IA de la sesión")
        st.markdown(review.get("session_summary_md", "—"))

        st.subheader("✅ Hallazgos checklist A+")
        cf = review.get("checklist_findings") or {}
        st.write(cf.get("pass_vs_fail_interpretation", "—"))
        breaks = cf.get("most_common_breaks") or []
        if breaks:
            st.markdown("**Rupturas más comunes:**")
            for b in breaks:
                st.write(f"- {b}")

        # Análisis por tipo de outcome
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📉 Análisis de Pérdidas:**")
            st.write(cf.get("loss_analysis", "—"))
        with col2:
            st.markdown("**📈 Análisis de Ganancias:**")
            st.write(cf.get("win_analysis", "—"))
        with col3:
            st.markdown("**➡️ Análisis de Empates:**")
            st.write(cf.get("tie_analysis", "—"))

        st.subheader("🧱 Reglas accionables")
        rules = review.get("action_rules") or []
        if rules:
            for i, r in enumerate(rules, start=1):
                st.write(f"{i}. {r}")
        else:
            st.write("—")

        # ----------------------------
        # Auditoría por trade con evidencia (todas las operaciones)
        # ----------------------------
        st.divider()
        st.subheader("🧨 Auditoría por operación con evidencia")

        # Usar trade_reviews (nuevo) o loss_trade_reviews (compatibilidad)
        all_reviews = review.get("trade_reviews") or review.get("loss_trade_reviews") or []

        if not all_reviews:
            st.info("No hubo operaciones con imagen (o no se incluyeron por límite/URL no imagen).")
        else:
            # Estadísticas generales
            good_losses = sum(1 for x in all_reviews if x.get("trade_type") == "good_loss")
            bad_losses = sum(1 for x in all_reviews if x.get("trade_type") == "bad_loss")
            good_wins = sum(1 for x in all_reviews if x.get("trade_type") == "good_win")
            lucky_wins = sum(1 for x in all_reviews if x.get("trade_type") == "lucky_win")
            ties = sum(1 for x in all_reviews if x.get("trade_type") == "neutral_tie")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("✅ Good Loss", good_losses, help="Pérdidas válidas (setup A+ correcto)")
            c2.metric("❌ Bad Loss", bad_losses, help="Pérdidas por error de ejecución")
            c3.metric("💚 Good Win", good_wins, help="Ganancias válidas y replicables")
            c4.metric("🍀 Lucky Win", lucky_wins, help="Ganancias por suerte (no replicable)")
            c5.metric("➡️ Ties", ties, help="Empates")

            # Métricas derivadas
            total_losses = good_losses + bad_losses
            total_wins = good_wins + lucky_wins

            if total_losses > 0:
                bad_loss_pct = (bad_losses / total_losses) * 100
                st.warning(f"⚠️ {bad_loss_pct:.1f}% de las pérdidas fueron errores evitables (bad_loss)")

            if total_wins > 0:
                lucky_win_pct = (lucky_wins / total_wins) * 100
                if lucky_win_pct > 30:
                    st.warning(f"⚠️ {lucky_win_pct:.1f}% de las ganancias fueron por suerte (no replicables)")
                else:
                    st.success(f"✅ {100 - lucky_win_pct:.1f}% de las ganancias fueron por ejecución correcta")

            st.divider()

            # Filtros para visualización
            filter_outcome = st.multiselect(
                "Filtrar por resultado:",
                options=["LOSS", "WIN", "TIE"],
                default=["LOSS", "WIN", "TIE"]
            )

            filter_type = st.multiselect(
                "Filtrar por tipo:",
                options=["good_loss", "bad_loss", "good_win", "lucky_win", "neutral_tie"],
                default=["good_loss", "bad_loss", "good_win", "lucky_win", "neutral_tie"]
            )

            by_id = {str(t.id): t for t in trades}

            for item in all_reviews:
                outcome = str(item.get("outcome", "UNKNOWN")).upper()
                trade_type = item.get("trade_type", "unknown")

                # Aplicar filtros
                if outcome not in filter_outcome or trade_type not in filter_type:
                    continue

                tid = str(item.get("trade_id", "—"))
                score = item.get("ai_score_a_plus_0_5", "—")
                is_a = bool(item.get("ai_is_a_plus", False))
                valid = bool(item.get("ai_validity", False))
                conf = item.get("confidence_0_1", "—")

                # Emoji según el tipo
                emoji_map = {
                    "good_loss": "✅",
                    "bad_loss": "❌",
                    "good_win": "💚",
                    "lucky_win": "🍀",
                    "neutral_tie": "➡️"
                }
                emoji = emoji_map.get(trade_type, "❓")

                t = by_id.get(tid)
                title = f"{emoji} {outcome} | trade={tid} | score={score}/5 | A+={is_a} | validity={valid} | type={trade_type} | conf={conf}"

                # Expandir por defecto solo las bad_loss y lucky_win
                expand_default = trade_type in ["bad_loss", "lucky_win"]

                with st.expander(title, expanded=expand_default):
                    if t and t.screenshot_url and _is_image_url(t.screenshot_url):
                        st.image(t.screenshot_url, caption=f"trade_id={tid}", use_container_width=True)

                    st.markdown("**Checklist por bloques:**")
                    st.code(json.dumps(item.get("checklist", {}), ensure_ascii=False, indent=2))

                    fb = item.get("failed_blocks") or []
                    fr = item.get("failed_rules") or []
                    if fb:
                        st.markdown("**failed_blocks:** " + ", ".join(fb))
                    if fr:
                        st.markdown("**failed_rules:**")
                        for rr in fr:
                            st.write(f"- {rr}")

                    st.markdown(f"**Causa primaria:** {item.get('primary_cause','—')}")

                    # Mostrar campos específicos según el tipo
                    if trade_type in ["good_loss", "bad_loss"]:
                        st.markdown(f"**🔧 Qué resolver:** {item.get('what_to_fix', item.get('one_fix', '—'))}")
                        st.markdown(f"**📈 Qué mejorar:** {item.get('what_to_improve','—')}")
                        st.markdown(f"**💡 Lección clave:** {item.get('key_lesson','—')}")
                        st.markdown(f"**🔄 ¿Era evitable?:** {item.get('replicability','—')}")
                    elif trade_type in ["good_win", "lucky_win"]:
                        st.markdown(f"**🔄 ¿Es replicable?:** {item.get('replicability','—')}")
                        st.markdown(f"**💡 Lección clave:** {item.get('key_lesson','—')}")
                    else:  # neutral_tie
                        st.markdown(f"**📈 Qué mejorar:** {item.get('what_to_improve','—')}")
                        st.markdown(f"**💡 Lección clave:** {item.get('key_lesson','—')}")

        # ----------------------------
        # Tabla trades sesión
        # ----------------------------
        st.divider()
        st.subheader("📋 Trades de la sesión")
        show_cols = [c for c in df.columns if c not in ("session_id",)]
        st.dataframe(df[show_cols].sort_values("trade_time"), use_container_width=True, hide_index=True)

    else:
        st.caption("Tip: presiona **Analizar sesión con IA** para generar auditoría y clasificación A+ válido vs falso.")


# =========================================================
# TAB 2: SEMANAL (5 números)
# =========================================================
with tabs[1]:
    st.caption("Dashboard semanal mínimo (sin cambiar BD). Si no hay auditorías IA guardadas, usa proxy por checklist_pass.")

    week_start = _weekly_window_start(datetime.now(LOCAL_TZ))

    # sesiones semana
    ds = df_sessions.copy()
    ds["start_time_dt"] = pd.to_datetime(ds["start_time"], utc=True, errors="coerce").dt.tz_convert(LOCAL_TZ)
    ds_week = ds[ds["start_time_dt"] >= week_start]

    if ds_week.empty:
        st.info("No hay sesiones en la semana actual.")
        st.stop()

    # cargar trades de esas sesiones
    all_trades = []
    for sid in ds_week["id"].astype(str).tolist():
        all_trades.extend(trade_repo.list_by_session(sid, limit=5000) or [])

    dft = pd.DataFrame(all_trades)
    if dft.empty:
        st.info("No hay trades asociados a las sesiones de esta semana.")
        st.stop()

    if "checklist_pass" not in dft.columns and "checklist_passed" in dft.columns:
        dft["checklist_pass"] = dft["checklist_passed"]

    dft["amount"] = pd.to_numeric(dft.get("amount"), errors="coerce").fillna(0.0)
    dft["pnl"] = pd.to_numeric(dft.get("pnl"), errors="coerce").fillna(0.0)
    dft["r_mult"] = (dft["pnl"] / dft["amount"].replace({0: np.nan})).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # proxy score y proxy A+ (hasta persistir auditoría IA)
    dft["ai_score_proxy"] = _score_proxy_from_user_checklist(dft)
    dft["ai_is_a_plus_proxy"] = (dft["ai_score_proxy"] == 5)

    # 5 métricas
    score_avg = float(dft["ai_score_proxy"].mean()) if len(dft) else 0.0
    pct_a = float(dft["ai_is_a_plus_proxy"].mean() * 100.0) if len(dft) else 0.0
    ev_r_week = float(dft["r_mult"].mean()) if len(dft) else 0.0

    # sesiones disciplinadas (proxy): sesión disciplinada si >=70% checklist_pass True
    def sess_disc(sid: str) -> bool:
        s = dft[dft["session_id"].astype(str) == str(sid)]
        if s.empty:
            return False
        b = s["checklist_pass"].map(_normalize_bool)
        denom = max(1, int(b.notna().sum()))
        return float((b == True).sum()) / denom >= 0.70  # noqa: E712

    sess_ids = ds_week["id"].astype(str).tolist()
    disc_rate = sum(1 for sid in sess_ids if sess_disc(sid)) / max(1, len(sess_ids)) * 100.0

    # % pérdidas NO A+ (proxy): LOSS con score != 5
    losses = dft[dft["outcome"] == "LOSS"]
    if len(losses) == 0:
        pct_bad_losses = 0.0
    else:
        pct_bad_losses = float(((losses["ai_score_proxy"] != 5).sum() / len(losses)) * 100.0)

    a, b, c, d, e = st.columns(5)
    a.metric("Score A+ medio", f"{score_avg:.2f}")
    b.metric("% Trades A+", f"{pct_a:.1f}%")
    c.metric("EV (R)", f"{ev_r_week:.3f}")
    d.metric("% Sesiones disciplinadas", f"{disc_rate:.1f}%")
    e.metric("% pérdidas NO A+", f"{pct_bad_losses:.1f}%")

    st.divider()
    st.markdown("**Interpretación rápida:**")
    st.write("- Si % Trades A+ baja, el problema es disciplina/paciencia.")
    st.write("- Si EV < 0 con A+ alto, problema de payout/timing.")
    st.write("- Si % pérdidas NO A+ sube, estás regalando EV (errores de ejecución).")
