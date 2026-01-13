from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from trade_journal.data.supabase_client import SupabaseClient
from trade_journal.domain.models import SessionCreate, TradeCreate


class TradeRepository:
    def __init__(self, sb: SupabaseClient):
        self.sb = sb

    def list_recent(self, limit: int = 500) -> List[dict]:
        return self.sb.select("trades", order="trade_time.desc", limit=limit)

    def list_by_date(self, day: str) -> List[dict]:
        return self.sb.select("trades", filters={"trade_date": f"eq.{day}"}, order="trade_time.asc")

    def list_by_session(self, session_id: str, limit: int = 5000) -> List[dict]:
        # requiere columna session_id en trades (ya la tienes)
        return self.sb.select(
            "trades",
            filters={"session_id": f"eq.{session_id}"},
            order="trade_time.asc",
            limit=limit,
        )

    def create(self, trade: TradeCreate) -> dict:
        payload = trade.model_dump()
        payload["trade_time"] = trade.trade_time.isoformat()
        return self.sb.insert("trades", [payload])[0]


class SessionRepository:
    def __init__(self, sb: SupabaseClient):
        self.sb = sb

    def list_recent(self, limit: int = 200) -> List[dict]:
        return self.sb.select("sessions", order="start_time.desc", limit=limit)

    def get(self, session_id: str) -> Optional[dict]:
        rows = self.sb.select("sessions", filters={"id": f"eq.{session_id}"}, limit=1)
        if rows:
            return rows[0]
        return None

    def start(self, start_time: datetime, notes: Optional[str] = None) -> dict:
        payload = SessionCreate(start_time=start_time, notes=notes).model_dump()
        payload["start_time"] = start_time.isoformat()
        return self.sb.insert("sessions", [payload])[0]

    def stop(self, session_id: str, end_time: datetime, duration_min: float) -> dict:
        patch = {"end_time": end_time.isoformat(), "duration_min": duration_min}
        return self.sb.patch("sessions", {"id": f"eq.{session_id}"}, patch)[0]
