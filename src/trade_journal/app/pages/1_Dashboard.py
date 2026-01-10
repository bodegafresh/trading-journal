from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import pytz

from trade_journal.app.utils import get_repos

LOCAL_TZ = pytz.timezone("America/Santiago")

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

trade_repo, _ = get_repos()


# ----------------------------
# Helpers
# ----------------------------
def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Evita KeyErrors si el repo/BD no trae columnas nuevas."""
    if df is None or df.empty:
        return df

    defaults = {
        "setup_tag": None,
        "market_regime": None,
        "quality_grade": None,
        "checklist_passed": True,  # default lógico
        "session_id": None,
        "screenshot_url": None,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # normaliza strings
    for col in ("setup_tag", "market_regime", "quality_grade"):
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").astype(str)

    # bool checklist
    if "checklist_passed" in df.columns:
        # acepta True/False, 1/0, "true"/"false", None
        df["checklist_passed"] = df["checklist_passed"].map(
            lambda x: True if str(x).strip().lower() in ("true", "1", "t", "yes") else (False if str(x).strip().lower() in ("false", "0", "f", "no") else bool(x))
        )

    return df


def _rows_to_df(rows: Any) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if df.empty:
        return df

    # --- trade_time (UTC -> America/Santiago) ---
    if "trade_time" in df.columns:
        dt = pd.to_datetime(df["trade_time"], errors="coerce", utc=True, format="mixed")
        df["trade_time"] = dt.dt.tz_convert(LOCAL_TZ)

    # --- trade_date (de la BD) ---
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date

    # --- trade_date_local (derivado de trade_time local; fallback a trade_date) ---
    if "trade_time" in df.columns:
        df["trade_date_local"] = df["trade_time"].dt.date
    if "trade_date_local" not in df.columns:
        df["trade_date_local"] = df["trade_date"] if "trade_date" in df.columns else pd.NaT

    # numeric fields
    for col in ("pnl", "amount", "payout_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # strings
    for col in ("outcome", "timeframe", "emotion", "asset"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # R multiple
    if "pnl" in df.columns and "amount" in df.columns:
        denom = df["amount"].replace({0: np.nan})
        df["r_mult"] = (df["pnl"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        df["r_mult"] = 0.0

    df = _ensure_columns(df)
    return df


@st.cache_data(ttl=30)
def get_trades_day(selected_day: date) -> pd.DataFrame:
    rows = trade_repo.list_recent(limit=50000)
    df = _rows_to_df(rows)
    if df.empty:
        return df
    return df[df["trade_date_local"] == selected_day]


@st.cache_data(ttl=30)
def get_trades_all(limit: int = 50000) -> pd.DataFrame:
    rows = trade_repo.list_recent(limit=limit)
    return _rows_to_df(rows)


# ----------------------------
# Metrics
# ----------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "ties": 0,
            "wr_with_ties": 0.0, "wr_no_ties": 0.0, "tie_rate": 0.0,
            "pnl_total": 0.0, "pnl_avg": 0.0, "stake_avg": 0.0, "ev_r": 0.0,
        }

    wins = int((df["outcome"] == "WIN").sum())
    losses = int((df["outcome"] == "LOSS").sum())
    ties = int((df["outcome"] == "TIE").sum())
    trades = int(len(df))

    pnl_total = float(df["pnl"].sum()) if "pnl" in df.columns else 0.0
    pnl_avg = float(pnl_total / trades) if trades else 0.0
    stake_avg = float(df["amount"].mean()) if "amount" in df.columns and trades else 0.0

    wr_with_ties = (wins / trades) if trades else 0.0
    decided = wins + losses
    wr_no_ties = (wins / decided) if decided else 0.0
    tie_rate = (ties / trades) if trades else 0.0

    ev_r = float(df["r_mult"].mean()) if "r_mult" in df.columns and trades else 0.0

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "wr_with_ties": wr_with_ties,
        "wr_no_ties": wr_no_ties,
        "tie_rate": tie_rate,
        "pnl_total": pnl_total,
        "pnl_avg": pnl_avg,
        "stake_avg": stake_avg,
        "ev_r": ev_r,
    }


def equity_and_drawdown(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "trade_time" not in df.columns:
        return {"curve": pd.DataFrame(), "max_dd": 0.0, "avg_dd": 0.0, "max_dd_duration_trades": 0}

    d = df.sort_values("trade_time").copy()
    d["equity_r"] = d["r_mult"].cumsum()
    d["peak"] = d["equity_r"].cummax()
    d["dd"] = d["equity_r"] - d["peak"]

    max_dd = float(d["dd"].min()) if len(d) else 0.0
    avg_dd = float(d["dd"][d["dd"] < 0].mean()) if (d["dd"] < 0).any() else 0.0

    in_dd = d["dd"] < 0
    dd_id = (in_dd != in_dd.shift(1)).cumsum()
    max_dur = 0
    if in_dd.any():
        for _, seg in d[in_dd].groupby(dd_id):
            max_dur = max(max_dur, int(len(seg)))

    curve = d[["trade_time", "r_mult", "equity_r", "dd"]].copy()
    return {"curve": curve, "max_dd": max_dd, "avg_dd": avg_dd, "max_dd_duration_trades": max_dur}


def group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()

    g = df.copy()
    g["is_win"] = (g["outcome"] == "WIN").astype(int)
    g["is_loss"] = (g["outcome"] == "LOSS").astype(int)
    g["is_tie"] = (g["outcome"] == "TIE").astype(int)

    out = (
        g.groupby(group_col, as_index=False)
        .agg(
            trades=("outcome", "count"),
            wins=("is_win", "sum"),
            losses=("is_loss", "sum"),
            ties=("is_tie", "sum"),
            pnl_total=("pnl", "sum"),
            stake_avg=("amount", "mean"),
            ev_r=("r_mult", "mean"),
        )
    )

    out["wr_with_ties"] = np.where(out["trades"] > 0, out["wins"] / out["trades"], 0.0)
    decided = out["wins"] + out["losses"]
    out["wr_no_ties"] = np.where(decided > 0, out["wins"] / decided, 0.0)
    out["tie_rate"] = np.where(out["trades"] > 0, out["ties"] / out["trades"], 0.0)

    return out.sort_values(["ev_r", "wr_no_ties", "trades"], ascending=[False, False, False])


def hour_block_stats(df: pd.DataFrame, block_hours: int = 1) -> pd.DataFrame:
    if df is None or df.empty or "trade_time" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d["hour"] = d["trade_time"].dt.hour
    d["hour_block"] = (d["hour"] // block_hours) * block_hours
    d["hour_label"] = d["hour_block"].astype(int).map(
        lambda h: f"{h:02d}:00–{(h+block_hours-1)%24:02d}:59"
    )
    return group_stats(d, "hour_label")


def streak_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty or "outcome" not in df.columns or "trade_time" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    d = df.sort_values("trade_time").copy()
    seq = d["outcome"].tolist()

    win_streaks, loss_streaks = [], []
    cur_type, cur_len = None, 0

    def flush():
        nonlocal cur_type, cur_len
        if cur_type == "WIN" and cur_len > 0:
            win_streaks.append(cur_len)
        elif cur_type == "LOSS" and cur_len > 0:
            loss_streaks.append(cur_len)
        cur_type, cur_len = None, 0

    for o in seq:
        if o not in ("WIN", "LOSS"):
            flush()
            continue
        if cur_type is None:
            cur_type, cur_len = o, 1
        elif o == cur_type:
            cur_len += 1
        else:
            flush()
            cur_type, cur_len = o, 1

    flush()

    def dist(streaks: list[int], label: str) -> pd.DataFrame:
        if not streaks:
            return pd.DataFrame(columns=["length", "count", "type"])
        s = pd.Series(streaks)
        out = s.value_counts().sort_index().reset_index()
        out.columns = ["length", "count"]
        out["type"] = label
        return out

    return dist(win_streaks, "WIN"), dist(loss_streaks, "LOSS")


# ----------------------------
# UI: filters
# ----------------------------
c_left, c_right = st.columns([1, 2])
with c_left:
    selected_day: date = st.date_input("Día", value=date.today())
with c_right:
    st.caption("Los datos se leen desde Supabase (tabla `trades`).")

with st.sidebar:
    st.markdown("### Acumulado")
    all_limit = st.number_input("Máx trades a cargar", min_value=1000, max_value=200000, value=50000, step=5000)
    hour_block = st.selectbox("Bloque horario", options=[1, 2], index=0)
    st.caption("Horarios calculados en zona local: America/Santiago")


# ----------------------------
# Data
# ----------------------------
df_day = get_trades_day(selected_day)
df_all = get_trades_all(limit=int(all_limit))

k_day = compute_kpis(df_day)
k_all = compute_kpis(df_all)


# ----------------------------
# KPIs Day
# ----------------------------
st.subheader("📅 Resumen del día")

d1, d2, d3, d4, d5, d6 = st.columns(6)
d1.metric("Trades", k_day["trades"])
d2.metric("W / L / T", f"{k_day['wins']} / {k_day['losses']} / {k_day['ties']}")
d3.metric("WinRate (incl. TIE)", f"{k_day['wr_with_ties']*100:.2f}%")
d4.metric("WinRate (sin TIE)", f"{k_day['wr_no_ties']*100:.2f}%")
d5.metric("PnL", f"{k_day['pnl_total']:.2f}")
d6.metric("EV (R)", f"{k_day['ev_r']:.3f}")


# ----------------------------
# KPIs Total
# ----------------------------
st.subheader("📈 Acumulado")

a1, a2, a3, a4, a5, a6 = st.columns(6)
a1.metric("Trades", k_all["trades"])
a2.metric("W / L / T", f"{k_all['wins']} / {k_all['losses']} / {k_all['ties']}")
a3.metric("WinRate (incl. TIE)", f"{k_all['wr_with_ties']*100:.2f}%")
a4.metric("WinRate (sin TIE)", f"{k_all['wr_no_ties']*100:.2f}%")
a5.metric("PnL total", f"{k_all['pnl_total']:.2f}")
a6.metric("PnL promedio/trade", f"{k_all['pnl_avg']:.2f}")


# ----------------------------
# ✅ NUEVO: Estrategia / Calidad
# ----------------------------
st.divider()
st.subheader("🧩 Estrategia / Calidad")

tabs = st.tabs(["Setup", "Régimen de mercado", "Calidad", "Checklist"])

with tabs[0]:
    by_setup = group_stats(df_all[df_all["setup_tag"].astype(str).str.strip() != ""], "setup_tag")
    if by_setup.empty:
        st.info("Aún no hay datos de setup_tag (o el repo no trae la columna).")
    else:
        view = by_setup.copy()
        view["wr_with_ties"] = (view["wr_with_ties"] * 100).round(2)
        view["wr_no_ties"] = (view["wr_no_ties"] * 100).round(2)
        view["tie_rate"] = (view["tie_rate"] * 100).round(2)
        view["ev_r"] = view["ev_r"].round(3)
        view["pnl_total"] = view["pnl_total"].round(2)
        view["stake_avg"] = view["stake_avg"].round(2)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(by_setup.sort_values("ev_r", ascending=False), x="setup_tag", y="ev_r", hover_data=["trades"]), use_container_width=True)

with tabs[1]:
    by_reg = group_stats(df_all[df_all["market_regime"].astype(str).str.strip() != ""], "market_regime")
    if by_reg.empty:
        st.info("Aún no hay datos de market_regime (o el repo no trae la columna).")
    else:
        view = by_reg.copy()
        view["wr_with_ties"] = (view["wr_with_ties"] * 100).round(2)
        view["wr_no_ties"] = (view["wr_no_ties"] * 100).round(2)
        view["tie_rate"] = (view["tie_rate"] * 100).round(2)
        view["ev_r"] = view["ev_r"].round(3)
        view["pnl_total"] = view["pnl_total"].round(2)
        view["stake_avg"] = view["stake_avg"].round(2)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(by_reg.sort_values("ev_r", ascending=False), x="market_regime", y="ev_r", hover_data=["trades"]), use_container_width=True)

with tabs[2]:
    by_q = group_stats(df_all[df_all["quality_grade"].astype(str).str.strip() != ""], "quality_grade")
    if by_q.empty:
        st.info("Aún no hay datos de quality_grade (o el repo no trae la columna).")
    else:
        view = by_q.copy()
        view["wr_with_ties"] = (view["wr_with_ties"] * 100).round(2)
        view["wr_no_ties"] = (view["wr_no_ties"] * 100).round(2)
        view["tie_rate"] = (view["tie_rate"] * 100).round(2)
        view["ev_r"] = view["ev_r"].round(3)
        view["pnl_total"] = view["pnl_total"].round(2)
        view["stake_avg"] = view["stake_avg"].round(2)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(by_q.sort_values("ev_r", ascending=False), x="quality_grade", y="ev_r", hover_data=["trades"]), use_container_width=True)

with tabs[3]:
    dchk = df_all.copy()
    if "checklist_passed" not in dchk.columns:
        st.info("No viene checklist_passed desde el repo/BD.")
    else:
        dchk["checklist"] = np.where(dchk["checklist_passed"], "PASS", "FAIL")
        by_chk = group_stats(dchk, "checklist")
        view = by_chk.copy()
        view["wr_with_ties"] = (view["wr_with_ties"] * 100).round(2)
        view["wr_no_ties"] = (view["wr_no_ties"] * 100).round(2)
        view["tie_rate"] = (view["tie_rate"] * 100).round(2)
        view["ev_r"] = view["ev_r"].round(3)
        view["pnl_total"] = view["pnl_total"].round(2)
        view["stake_avg"] = view["stake_avg"].round(2)
        st.dataframe(view, use_container_width=True, hide_index=True)


# ----------------------------
# Equity in R + Drawdown
# ----------------------------
st.subheader("📉 Equity en R & Drawdown")

dd = equity_and_drawdown(df_all)
curve = dd["curve"]

e1, e2, e3 = st.columns(3)
e1.metric("Max drawdown (R)", f"{dd['max_dd']:.3f}")
e2.metric("Avg drawdown (R)", f"{dd['avg_dd']:.3f}")
e3.metric("Duración máx DD (trades)", f"{dd['max_dd_duration_trades']}")

g1, g2 = st.columns(2)
with g1:
    st.markdown("### Curva de equity (R)")
    if curve.empty:
        st.info("Sin datos históricos.")
    else:
        st.plotly_chart(px.line(curve, x="trade_time", y="equity_r"), use_container_width=True)

with g2:
    st.markdown("### Drawdown (R)")
    if curve.empty:
        st.info("Sin datos históricos.")
    else:
        st.plotly_chart(px.area(curve, x="trade_time", y="dd"), use_container_width=True)


# ----------------------------
# EV / WinRate by hour blocks
# ----------------------------
st.subheader("🕒 EV / WinRate por horario")

by_hour = hour_block_stats(df_all, block_hours=int(hour_block))
if by_hour.empty:
    st.info("Sin datos suficientes para agrupar por horario.")
else:
    view = by_hour.copy()
    view["wr_with_ties"] = (view["wr_with_ties"] * 100).round(2)
    view["wr_no_ties"] = (view["wr_no_ties"] * 100).round(2)
    view["tie_rate"] = (view["tie_rate"] * 100).round(2)
    view["ev_r"] = view["ev_r"].round(3)
    view["pnl_total"] = view["pnl_total"].round(2)
    view["stake_avg"] = view["stake_avg"].round(2)

    st.dataframe(
        view.rename(
            columns={
                "hour_label": "bloque",
                "wr_with_ties": "wr_incl_tie_%",
                "wr_no_ties": "wr_sin_tie_%",
                "tie_rate": "tie_%",
                "ev_r": "ev_R",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(
        px.bar(by_hour.sort_values("ev_r", ascending=False), x="hour_label", y="ev_r", hover_data=["trades"]),
        use_container_width=True,
    )


# ----------------------------
# EV / WinRate by timeframe
# ----------------------------
st.subheader("⏱️ EV / WinRate por timeframe")

by_tf = group_stats(df_all, "timeframe")
if by_tf.empty:
    st.info("Sin datos para agrupar por timeframe.")
else:
    view = by_tf.copy()
    view["wr_with_ties"] = (view["wr_with_ties"] * 100).round(2)
    view["wr_no_ties"] = (view["wr_no_ties"] * 100).round(2)
    view["tie_rate"] = (view["tie_rate"] * 100).round(2)
    view["ev_r"] = view["ev_r"].round(3)
    view["pnl_total"] = view["pnl_total"].round(2)
    view["stake_avg"] = view["stake_avg"].round(2)

    st.dataframe(
        view.rename(
            columns={
                "wr_with_ties": "wr_incl_tie_%",
                "wr_no_ties": "wr_sin_tie_%",
                "tie_rate": "tie_%",
                "ev_r": "ev_R",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------
# EV / WinRate by emotion
# ----------------------------
st.subheader("🧠 EV / WinRate por emoción")

by_em = group_stats(df_all, "emotion")
if by_em.empty:
    st.info("Sin datos para agrupar por emoción.")
else:
    view = by_em.copy()
    view["wr_with_ties"] = (view["wr_with_ties"] * 100).round(2)
    view["wr_no_ties"] = (view["wr_no_ties"] * 100).round(2)
    view["tie_rate"] = (view["tie_rate"] * 100).round(2)
    view["ev_r"] = view["ev_r"].round(3)
    view["pnl_total"] = view["pnl_total"].round(2)
    view["stake_avg"] = view["stake_avg"].round(2)

    st.dataframe(
        view.rename(
            columns={
                "wr_with_ties": "wr_incl_tie_%",
                "wr_no_ties": "wr_sin_tie_%",
                "tie_rate": "tie_%",
                "ev_r": "ev_R",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(px.bar(by_em.sort_values("ev_r", ascending=False), x="emotion", y="ev_r", hover_data=["trades"]), use_container_width=True)


# ----------------------------
# Streaks
# ----------------------------
st.subheader("🔁 Secuencias (rachas)")

win_dist, loss_dist = streak_distribution(df_all)
s1, s2 = st.columns(2)

with s1:
    st.markdown("### Rachas de WIN (consecutivos)")
    if win_dist.empty:
        st.info("Sin rachas WIN para mostrar (o no hay datos).")
    else:
        st.plotly_chart(px.bar(win_dist, x="length", y="count"), use_container_width=True)
        st.dataframe(win_dist.sort_values("length"), use_container_width=True, hide_index=True)

with s2:
    st.markdown("### Rachas de LOSS (consecutivos)")
    if loss_dist.empty:
        st.info("Sin rachas LOSS para mostrar (o no hay datos).")
    else:
        st.plotly_chart(px.bar(loss_dist, x="length", y="count"), use_container_width=True)
        st.dataframe(loss_dist.sort_values("length"), use_container_width=True, hide_index=True)


# ----------------------------
# Trades day table
# ----------------------------
st.divider()
st.subheader("🧾 Trades del día")

if df_day.empty:
    st.write("—")
else:
    st.dataframe(df_day.sort_values("trade_time", ascending=False), use_container_width=True, hide_index=True)


# ----------------------------
# Summary
# ----------------------------
st.divider()
st.subheader("📌 Métricas adicionales útiles para discusión")

st.markdown(
    f"""
### Win rate: dos definiciones (ambas útiles)
- **Win rate (incluye ties en el total):** wins / (wins+losses+ties)
- **Win rate “decided only” (excluye ties):** wins / (wins+losses)

En tu histórico:
- Win rate (incl. ties): **{k_all["wr_with_ties"]*100:.2f}%**
- Win rate (decided): **{k_all["wr_no_ties"]*100:.2f}%**

### Tasa de tie
\\[
P(tie) = {k_all["tie_rate"]*100:.2f}\\%
\\]

### EV en unidades monetarias (referencial)
\\[
E[PnL] = E[R \\cdot Amount]
\\]

En tu histórico:
- Stake promedio: **{k_all["stake_avg"]:.2f}**
- EV en R (promedio): **{k_all["ev_r"]:.3f}**
- PnL promedio por trade: **{k_all["pnl_avg"]:.2f}**
- PnL total: **{k_all["pnl_total"]:.2f}**
"""
)
