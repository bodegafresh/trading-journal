from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import streamlit as st

from trade_journal.app.utils import get_repos, get_recent_sessions, get_review_repo
from trade_journal.ai.openai_review import (
    TradeLite,
    analyze_session_with_vision,
    build_session_payload,
    normalize_review,
)

LOCAL_TZ = pytz.timezone("America/Santiago")

st.set_page_config(page_title="Review semanal (IA)", page_icon="🧠", layout="wide")
st.title("🧠 Review semanal (IA) — semana completa")


# ----------------------------
# Helpers
# ----------------------------
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


def _week_start(dt: Any) -> datetime:
    """
    Inicio de semana (domingo 00:00) en horario local (naive).
    """
    d = _to_dt(dt)
    if d.tzinfo is not None:
        d = d.astimezone(LOCAL_TZ).replace(tzinfo=None)
    # Python: weekday() -> lunes=0 ... domingo=6
    days_since_sunday = (d.weekday() + 1) % 7
    start = d - timedelta(days=days_since_sunday)
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


def _label_week(start: datetime) -> str:
    end = start + timedelta(days=7)
    return f"{start.date()} → {end.date()}"


def _fallback_weekly_summary(payload: Dict[str, Any]) -> str:
    kpis = payload.get("kpis") or {}
    trades = kpis.get("trades", 0)
    wins = kpis.get("wins", 0)
    losses = kpis.get("losses", 0)
    ties = kpis.get("ties", 0)
    ev_r = kpis.get("ev_r", 0.0)
    pnl_total = kpis.get("pnl_total", 0.0)
    return (
        "### Resumen de la semana (auto)\n"
        f"- Trades: {trades} | W/L/T: {wins}/{losses}/{ties}\n"
        f"- EV (R): {ev_r:.3f} | PnL: {pnl_total:.2f}\n"
        "- Nota: resumen generado automáticamente por falta de contenido en la respuesta IA."
    )


def _collect_session_trade_reviews(session_ids: List[str]) -> List[Dict[str, Any]]:
    reviews = review_repo.list_session_reviews(session_ids)
    collected: List[Dict[str, Any]] = []
    for r in reviews:
        rev = r.get("review") or {}
        t_reviews = rev.get("trade_reviews") or rev.get("loss_trade_reviews") or []
        collected.extend(t_reviews)
    return collected


# ----------------------------
# Load repos + sessions
# ----------------------------
trade_repo, session_repo = get_repos()
review_repo = get_review_repo()
df_sessions = get_recent_sessions(limit=800)

if df_sessions.empty:
    st.info("No hay sesiones aún.")
    st.stop()

ds = df_sessions.copy()
ds["start_time_dt"] = pd.to_datetime(ds["start_time"], errors="coerce")
ds["week_start"] = ds["start_time_dt"].apply(_week_start)

week_starts = sorted(ds["week_start"].dropna().unique(), reverse=True)
if not week_starts:
    st.info("No hay sesiones con fecha válida para agrupar por semana.")
    st.stop()

week_idx = st.selectbox(
    "Selecciona semana",
    options=list(range(len(week_starts))),
    format_func=lambda i: _label_week(week_starts[i]),
)

week_start = week_starts[int(week_idx)]
week_end = week_start + timedelta(days=7)

ds_week = ds[(ds["start_time_dt"] >= week_start) & (ds["start_time_dt"] < week_end)]
if ds_week.empty:
    st.info("No hay sesiones en la semana seleccionada.")
    st.stop()


# ----------------------------
# Cargar trades de la semana
# ----------------------------
all_trades = []
for sid in ds_week["id"].astype(str).tolist():
    all_trades.extend(trade_repo.list_by_session(sid, limit=10000) or [])

dft = pd.DataFrame(all_trades)
if dft.empty:
    st.info("No hay trades asociados a las sesiones de esta semana.")
    st.stop()

if "trade_time" in dft.columns:
    dt = pd.to_datetime(dft["trade_time"], utc=True, errors="coerce", format="mixed")
    dft["trade_time"] = dt.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

if "checklist_pass" not in dft.columns and "checklist_passed" in dft.columns:
    dft["checklist_pass"] = dft["checklist_passed"]

dft["amount"] = pd.to_numeric(dft.get("amount"), errors="coerce").fillna(0.0)
dft["pnl"] = pd.to_numeric(dft.get("pnl"), errors="coerce").fillna(0.0)
dft["r_mult"] = (
    dft["pnl"] / dft["amount"].replace({0: np.nan})
).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# KPIs por activo (top 8 por volumen)
asset_stats = []
if "asset" in dft.columns:
    grp = dft.groupby("asset", dropna=False)
    for asset, g in grp:
        trades_n = len(g)
        wins_n = int((g["outcome"] == "WIN").sum())
        losses_n = int((g["outcome"] == "LOSS").sum())
        ties_n = int((g["outcome"] == "TIE").sum())
        ev_r_a = float(g["r_mult"].mean()) if trades_n else 0.0
        pnl_a = float(g["pnl"].sum()) if trades_n else 0.0
        wr_no_ties = float(wins_n / max(1, wins_n + losses_n))
        asset_stats.append(
            {
                "asset": str(asset),
                "trades": trades_n,
                "wins": wins_n,
                "losses": losses_n,
                "ties": ties_n,
                "wr_no_ties": wr_no_ties,
                "ev_r": ev_r_a,
                "pnl_total": pnl_a,
            }
        )
    asset_stats = sorted(asset_stats, key=lambda x: x["trades"], reverse=True)[:8]

# KPIs por hora (UTC localizada)
hour_stats = []
if "trade_time" in dft.columns:
    hours = pd.to_datetime(dft["trade_time"], errors="coerce").dt.hour
    dft["_hour"] = hours
    for hour, g in dft.groupby("_hour"):
        if pd.isna(hour):
            continue
        trades_n = len(g)
        wins_n = int((g["outcome"] == "WIN").sum())
        losses_n = int((g["outcome"] == "LOSS").sum())
        ties_n = int((g["outcome"] == "TIE").sum())
        ev_r_h = float(g["r_mult"].mean()) if trades_n else 0.0
        pnl_h = float(g["pnl"].sum()) if trades_n else 0.0
        hour_stats.append(
            {
                "hour": int(hour),
                "trades": trades_n,
                "wins": wins_n,
                "losses": losses_n,
                "ties": ties_n,
                "ev_r": ev_r_h,
                "pnl_total": pnl_h,
            }
        )
    hour_stats = sorted(hour_stats, key=lambda x: x["trades"], reverse=True)[:8]

total = len(dft)
wins = int((dft["outcome"] == "WIN").sum())
losses = int((dft["outcome"] == "LOSS").sum())
ties = int((dft["outcome"] == "TIE").sum())
ev_r = float(dft["r_mult"].mean()) if total else 0.0
pnl_total = float(dft["pnl"].sum()) if total else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Sesiones", len(ds_week))
k2.metric("Trades", total)
k3.metric("W / L / T", f"{wins} / {losses} / {ties}")
k4.metric("EV (R)", f"{ev_r:.3f}")
k5.metric("PnL total", f"{pnl_total:.2f}")

st.divider()


# ----------------------------
# Construye TradeLite list
# ----------------------------
trades: List[TradeLite] = []
for _, r in dft.sort_values("trade_time").iterrows():
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
# IA review semanal
# ----------------------------
max_imgs = st.slider("Máx operaciones con imagen a auditar", 1, 40, 20)
prioritize = st.checkbox(
    "Priorizar pérdidas en el análisis",
    value=True,
    help="Si está activado, analiza primero las pérdidas, luego ganancias y empates",
)
use_cached = st.checkbox("Usar revisión guardada si existe", value=True)
run = st.button("🧠 Analizar semana con IA", type="primary", use_container_width=True)

week_start_str = week_start.date().isoformat()
week_end_str = week_end.date().isoformat()
cache_key = f"review_ai_week::{week_start_str}"
cached_db = review_repo.get_weekly_review(week_start_str)
if run:
    with st.spinner("Analizando semana con IA..."):
        week_meta: Dict[str, Any] = {
            "scope": "weekly",
            "week_start": week_start_str,
            "week_end": week_end_str,
            "sessions_count": len(ds_week),
            "session_ids": [str(s) for s in ds_week["id"].tolist()],
        }
        payload = build_session_payload(
            session_meta=week_meta,
            trades=trades,
            max_trades_with_images=int(max_imgs),
            prioritize_losses=prioritize,
        )
        if asset_stats:
            payload["kpis_by_asset"] = asset_stats
        if hour_stats:
            payload["kpis_by_hour"] = hour_stats
        session_ids = [str(s) for s in ds_week["id"].tolist()]
        prev_reviews = review_repo.list_session_reviews(session_ids)
        if prev_reviews:
            payload["session_reviews"] = [
                {
                    "session_id": r.get("session_id"),
                    "created_at": r.get("created_at"),
                    "session_summary_md": (r.get("review") or {}).get("session_summary_md"),
                    "checklist_findings": (r.get("review") or {}).get("checklist_findings"),
                    "action_rules": (r.get("review") or {}).get("action_rules"),
                }
                for r in prev_reviews
            ]
            # Resumen agregado de trade_reviews (si existen)
            trade_type_counts: Dict[str, int] = {}
            failed_blocks_counts: Dict[str, int] = {}
            failed_rules_counts: Dict[str, int] = {}
            total_trade_reviews = 0
            for r in prev_reviews:
                rev = r.get("review") or {}
                t_reviews = rev.get("trade_reviews") or rev.get("loss_trade_reviews") or []
                for tr in t_reviews:
                    total_trade_reviews += 1
                    tt = str(tr.get("trade_type", "unknown"))
                    trade_type_counts[tt] = trade_type_counts.get(tt, 0) + 1
                    for fb in tr.get("failed_blocks") or []:
                        failed_blocks_counts[fb] = failed_blocks_counts.get(fb, 0) + 1
                    for fr in tr.get("failed_rules") or []:
                        failed_rules_counts[fr] = failed_rules_counts.get(fr, 0) + 1
            if total_trade_reviews > 0:
                payload["session_trade_reviews_summary"] = {
                    "total_trade_reviews": total_trade_reviews,
                    "trade_type_counts": trade_type_counts,
                    "failed_blocks_top": sorted(
                        failed_blocks_counts.items(), key=lambda x: x[1], reverse=True
                    )[:5],
                    "failed_rules_top": sorted(
                        failed_rules_counts.items(), key=lambda x: x[1], reverse=True
                    )[:7],
                }
        prev_week_start = (week_start.date() - timedelta(days=7)).isoformat()
        prev_week = review_repo.get_weekly_review(prev_week_start)
        if prev_week:
            prev_review = prev_week.get("review") or {}
            payload["previous_week_review"] = {
                "week_start": prev_week.get("week_start"),
                "week_end": prev_week.get("week_end"),
                "session_summary_md": prev_review.get("session_summary_md"),
                "checklist_findings": prev_review.get("checklist_findings"),
                "action_rules": prev_review.get("action_rules"),
            }
        review = analyze_session_with_vision(session_payload=payload, max_output_tokens=3000)
        review_repo.upsert_weekly_review(week_start_str, week_end_str, payload, review)
        st.session_state[cache_key] = {"payload": payload, "review": review}
elif use_cached and cached_db:
    cached_review = normalize_review(cached_db.get("review") or {})
    # Si la review guardada está vacía, forzar nuevo análisis
    if cached_review.get("session_summary_md") or cached_review.get("checklist_findings"):
        st.session_state[cache_key] = {
            "payload": cached_db.get("payload"),
            "review": cached_review,
        }

cached = st.session_state.get(cache_key)

if cached:
    payload = cached.get("payload") or {}
    review = normalize_review(cached["review"] or {})

    meta = review.get("_meta") or {}
    usage = (meta.get("usage") or {})
    st.caption(
        "IA usage — "
        f"model={meta.get('model')} | prompt_version={meta.get('prompt_version')} | payload_hash={meta.get('payload_hash')} "
        f"| total_tokens={usage.get('total_tokens')} | prompt_tokens={usage.get('prompt_tokens')} | completion_tokens={usage.get('completion_tokens')} "
        f"| est_text_prompt_tokens={meta.get('estimated_text_prompt_tokens')}"
    )

    st.subheader("🧾 Resumen IA de la semana")
    summary = review.get("session_summary_md", "—")
    if str(summary).strip() in ("", "—"):
        summary = _fallback_weekly_summary(payload)
    st.markdown(summary)

    st.subheader("✅ Hallazgos checklist A+")
    cf = review.get("checklist_findings") or {}
    st.write(cf.get("pass_vs_fail_interpretation", "—"))
    breaks = cf.get("most_common_breaks") or []
    if breaks:
        st.markdown("**Rupturas más comunes:**")
        for b in breaks:
            st.write(f"- {b}")

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

    st.divider()
    st.subheader("🧨 Auditoría por operación con evidencia")

    all_reviews = review.get("trade_reviews") or review.get("loss_trade_reviews") or []
    if not all_reviews:
        session_ids = payload.get("session_meta", {}).get("session_ids") or ds_week["id"].astype(str).tolist()
        all_reviews = _collect_session_trade_reviews(session_ids)

    if not all_reviews:
        st.info("No hubo operaciones con imagen (o no se incluyeron por límite/URL no imagen).")
    else:
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

        filter_outcome = st.multiselect(
            "Filtrar por resultado:",
            options=["LOSS", "WIN", "TIE"],
            default=["LOSS", "WIN", "TIE"],
        )

        filter_type = st.multiselect(
            "Filtrar por tipo:",
            options=["good_loss", "bad_loss", "good_win", "lucky_win", "neutral_tie"],
            default=["good_loss", "bad_loss", "good_win", "lucky_win", "neutral_tie"],
        )

        by_id = {str(t.id): t for t in trades}

        for item in all_reviews:
            outcome = str(item.get("outcome", "UNKNOWN")).upper()
            trade_type = item.get("trade_type", "unknown")

            if outcome not in filter_outcome or trade_type not in filter_type:
                continue

            tid = str(item.get("trade_id", "—"))
            score = item.get("ai_score_a_plus_0_5", "—")
            is_a = bool(item.get("ai_is_a_plus", False))
            valid = bool(item.get("ai_validity", False))
            conf = item.get("confidence_0_1", "—")

            emoji_map = {
                "good_loss": "✅",
                "bad_loss": "❌",
                "good_win": "💚",
                "lucky_win": "🍀",
                "neutral_tie": "➡️",
            }
            emoji = emoji_map.get(trade_type, "❓")

            t = by_id.get(tid)
            title = (
                f"{emoji} {outcome} | trade={tid} | score={score}/5 | A+={is_a} | "
                f"validity={valid} | type={trade_type} | conf={conf}"
            )

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

                if trade_type in ["good_loss", "bad_loss"]:
                    st.markdown(f"**🔧 Qué resolver:** {item.get('what_to_fix', item.get('one_fix', '—'))}")
                    st.markdown(f"**📈 Qué mejorar:** {item.get('what_to_improve','—')}")
                    st.markdown(f"**💡 Lección clave:** {item.get('key_lesson','—')}")
                    st.markdown(f"**🔄 ¿Era evitable?:** {item.get('replicability','—')}")
                elif trade_type in ["good_win", "lucky_win"]:
                    st.markdown(f"**🔄 ¿Es replicable?:** {item.get('replicability','—')}")
                    st.markdown(f"**💡 Lección clave:** {item.get('key_lesson','—')}")
                else:
                    st.markdown(f"**📈 Qué mejorar:** {item.get('what_to_improve','—')}")
                    st.markdown(f"**💡 Lección clave:** {item.get('key_lesson','—')}")

    st.divider()
    st.subheader("📋 Trades de la semana")
    show_cols = [c for c in dft.columns if c not in ("session_id",)]
    st.dataframe(dft[show_cols].sort_values("trade_time"), use_container_width=True, hide_index=True)
else:
    st.caption("Tip: presiona **Analizar semana con IA** para generar auditoría y clasificación A+.")
