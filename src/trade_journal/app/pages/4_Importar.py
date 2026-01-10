from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from trade_journal.app.utils import get_repos
from trade_journal.domain.models import Direction, Outcome, TradeCreate

st.set_page_config(page_title="Importar", page_icon="📥", layout="wide")

st.title("📥 Importar CSV legado")

st.markdown(
    "Sube el CSV del script antiguo con columnas: "
    "`datetime,date,asset,timeframe,amount,direction,outcome,payout_pct,pnl,emotion,notes`"
)

trade_repo, _ = get_repos()

file = st.file_uploader("CSV", type=["csv"])
if not file:
    st.stop()

df = pd.read_csv(file)
st.write("Vista previa:")
st.dataframe(df.head(20), use_container_width=True)

required = ["datetime","asset","timeframe","amount","direction","outcome","payout_pct","pnl"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Faltan columnas requeridas: {missing}")
    st.stop()

def norm_dir(x: str) -> Direction:
    x = str(x).strip().upper()
    if x in ("UP","CALL","BUY","↑"):
        return Direction.UP
    if x in ("DOWN","PUT","SELL","↓"):
        return Direction.DOWN
    return Direction.UP

def norm_outcome(x: str) -> Outcome:
    x = str(x).strip().upper()
    if x in ("WIN","W","✅"):
        return Outcome.WIN
    if x in ("LOSS","L","❌"):
        return Outcome.LOSS
    if x in ("TIE","DRAW","0"):
        return Outcome.TIE
    return Outcome.LOSS

if st.button("Importar a Supabase"):
    trades = []
    for _, r in df.iterrows():
        # datetime en CSV puede venir como string; intentamos parsear
        dt_raw = r.get("datetime")
        try:
            trade_time = pd.to_datetime(dt_raw).to_pydatetime()
        except Exception:
            trade_time = datetime.utcnow()

        trade = TradeCreate(
            trade_time=trade_time,
            asset=str(r.get("asset","")).strip(),
            timeframe=str(r.get("timeframe","")).strip(),
            amount=float(r.get("amount", 0) or 0),
            direction=norm_dir(r.get("direction")),
            outcome=norm_outcome(r.get("outcome")),
            payout_pct=float(r.get("payout_pct", 0) or 0),
            pnl=float(r.get("pnl", 0) or 0),
            emotion=(str(r.get("emotion","")).strip() or None),
            notes=(str(r.get("notes","")).strip() or None),
        )
        trades.append(trade)

    try:
        # inserción por lotes
        batch = 500
        inserted = 0
        for i in range(0, len(trades), batch):
            chunk = trades[i:i+batch]
            payload = []
            for t in chunk:
                d = t.model_dump()
                d["trade_time"] = t.trade_time.isoformat()
                payload.append(d)
            trade_repo.sb.insert("trades", payload)
            inserted += len(chunk)

        st.success(f"Importados {inserted} trades ✅")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Error importando: {e}")
