from __future__ import annotations

import pandas as pd


def compute_kpis(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "winrate": 0.0,
            "pnl": 0.0,
            "avg_pnl": 0.0,
        }

    outcome = trades["outcome"].astype(str).str.upper()
    wins = int((outcome == "WIN").sum())
    losses = int((outcome == "LOSS").sum())
    ties = int((outcome == "TIE").sum())
    total = int(len(trades))
    winrate = (wins / max(1, (wins + losses))) * 100.0  # ties fuera del denominador
    pnl = float(trades["pnl"].astype(float).sum())
    avg_pnl = pnl / total

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "winrate": winrate,
        "pnl": pnl,
        "avg_pnl": avg_pnl,
    }
