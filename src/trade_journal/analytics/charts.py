from __future__ import annotations

import pandas as pd
import plotly.express as px


def equity_curve(trades: pd.DataFrame):
    if trades.empty:
        return None
    df = trades.sort_values("trade_time").copy()
    df["cum_pnl"] = df["pnl"].astype(float).cumsum()
    fig = px.line(df, x="trade_time", y="cum_pnl", markers=True, title="PnL acumulado")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def pnl_by_asset(trades: pd.DataFrame):
    if trades.empty:
        return None
    df = trades.copy()
    df["pnl"] = df["pnl"].astype(float)
    agg = df.groupby("asset", as_index=False)["pnl"].sum().sort_values("pnl", ascending=False)
    fig = px.bar(agg, x="asset", y="pnl", title="PnL por activo")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    return fig
