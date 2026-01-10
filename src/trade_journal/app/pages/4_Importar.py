from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
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

required = ["datetime", "asset", "timeframe", "amount", "direction", "outcome", "payout_pct", "pnl"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Faltan columnas requeridas: {missing}")
    st.stop()


ALLOWED_EMOTIONS = {"Neutral","Confiado","Enfocado","Ansioso","Impulsivo","Cansado","Frustrado"}

def norm_dir(x: str) -> Direction:
    x = str(x).strip().upper()
    if x in ("UP", "CALL", "BUY", "↑"):
        return Direction.UP
    if x in ("DOWN", "PUT", "SELL", "↓"):
        return Direction.DOWN
    return Direction.UP

def norm_outcome(x: str) -> Outcome:
    x = str(x).strip().upper()
    if x in ("WIN", "W", "✅"):
        return Outcome.WIN
    if x in ("LOSS", "L", "❌"):
        return Outcome.LOSS
    if x in ("TIE", "DRAW", "0"):
        return Outcome.TIE
    return Outcome.LOSS

def norm_float(v, default=0.0) -> float:
    if v is None:
        return float(default)
    if isinstance(v, str):
        v = v.replace(",", ".").strip()
        if v == "":
            return float(default)
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return float(default)
        return x
    except Exception:
        return float(default)

def norm_emotion(v) -> str:
    s = str(v).strip() if v is not None else ""
    if not s:
        return "Neutral"
    # normaliza capitalización simple
    s = s[0].upper() + s[1:].lower()
    return s if s in ALLOWED_EMOTIONS else "Neutral"

def parse_trade_time(dt_raw) -> datetime:
    # Fuerza UTC siempre
    ts = pd.to_datetime(dt_raw, errors="coerce", utc=True)
    if pd.isna(ts):
        return datetime.now(timezone.utc)
    return ts.to_pydatetime()

if st.button("Importar a Supabase"):
    payload = []
    errors = []

    for idx, r in df.iterrows():
        try:
            trade_time = parse_trade_time(r.get("datetime"))

            trade = TradeCreate(
                trade_time=trade_time,
                asset=str(r.get("asset","")).strip(),
                timeframe=str(r.get("timeframe","")).strip(),
                amount=norm_float(r.get("amount", 0), 0.0),
                direction=norm_dir(r.get("direction")),
                outcome=norm_outcome(r.get("outcome")),
                payout_pct=norm_float(r.get("payout_pct", 0), 0.0),
                pnl=norm_float(r.get("pnl", 0), 0.0),
                emotion=norm_emotion(r.get("emotion")),
                notes=(str(r.get("notes","")).strip() or None),
            )

            # CLAVE: serialización JSON para Enums + datetime
            d = trade.model_dump(mode="json")

            # Asegura strings exactos para checks (por si tu config de pydantic cambia)
            d["direction"] = trade.direction.value
            d["outcome"] = trade.outcome.value

            # Asegura ISO 8601 con tz
            if isinstance(trade.trade_time, datetime):
                # trade.model_dump(mode="json") normalmente ya lo hace,
                # pero lo dejamos blindado:
                if trade.trade_time.tzinfo is None:
                    tt = trade.trade_time.replace(tzinfo=timezone.utc)
                else:
                    tt = trade.trade_time.astimezone(timezone.utc)
                d["trade_time"] = tt.isoformat()

            payload.append(d)

        except Exception as e:
            errors.append((idx, str(e)))

    if errors:
        st.error("Errores construyendo payload (primeros 10):")
        st.write(errors[:10])
        st.stop()

    try:
        batch = 500
        inserted = 0
        for i in range(0, len(payload), batch):
            chunk = payload[i:i+batch]
            # IMPORTANTE: dependiendo de tu wrapper, puede ser:
            # trade_repo.sb.table("trades").insert(chunk).execute()
            # o trade_repo.sb.insert("trades", chunk)
            trade_repo.sb.insert("trades", chunk)
            inserted += len(chunk)

        st.success(f"Importados {inserted} trades ✅")
        st.cache_data.clear()

    except Exception as e:
        st.error(f"Error importando: {e}")
        st.info("Tip: imprime el payload del primer item para ver tipos exactos (direction/outcome/emotion).")
        st.write(payload[0] if payload else {})
