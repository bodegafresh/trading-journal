from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from trade_journal.data.supabase_client import SupabaseClient
from trade_journal.domain.models import SessionCreate, TradeCreate


class TradeRepository:
    def __init__(self, sb: SupabaseClient):
        self.sb = sb

    def list_recent(self, limit: int = 500) -> List[dict]:
        return self.sb.select("trades", order="trade_time.desc", limit=limit)

    def list_by_date(self, day: str) -> List[dict]:
        return self.sb.select(
            "trades",
            filters={"trade_date": f"eq.{day}"},
            order="trade_time.asc",
        )

    def create(self, trade: Union[TradeCreate, Dict[str, Any]]) -> dict:
        """
        Permite:
          - TradeCreate (tu modelo)
          - dict con nombres EXACTOS de columnas (recomendado para evitar nulls)
        """
        if isinstance(trade, dict):
            payload = dict(trade)
            # normaliza datetime si viene como datetime
            if isinstance(payload.get("trade_time"), datetime):
                payload["trade_time"] = payload["trade_time"].isoformat()
        else:
            payload = trade.model_dump()
            payload["trade_time"] = trade.trade_time.isoformat()

        inserted = self.sb.insert("trades", [payload])
        return inserted[0] if inserted else {}

    def list_by_session(self, session_id: str, limit: int = 5000) -> List[dict]:
        return self.sb.select(
            "trades",
            filters={"session_id": f"eq.{session_id}"},
            order="trade_time.asc",
            limit=limit,
        )



class SessionRepository:
    def __init__(self, sb: SupabaseClient):
        self.sb = sb

    def list_recent(self, limit: int = 200) -> List[dict]:
        return self.sb.select("sessions", order="start_time.desc", limit=limit)

    def list_open(self, limit: int = 50) -> List[dict]:
        # end_time is null => sesión abierta
        return self.sb.select(
            "sessions",
            filters={"end_time": "is.null"},
            order="start_time.desc",
            limit=limit,
        )

    def start(self, start_time: datetime, notes: Optional[str] = None) -> dict:
        payload = SessionCreate(start_time=start_time, notes=notes).model_dump()
        payload["start_time"] = start_time.isoformat()
        return self.sb.insert("sessions", [payload])[0]

    def stop(self, session_id: str, end_time: datetime, duration_min: float) -> dict:
        patch = {
            "end_time": end_time.isoformat(),
            "duration_min": duration_min,
        }
        return self.sb.patch("sessions", {"id": f"eq.{session_id}"}, patch)[0]
