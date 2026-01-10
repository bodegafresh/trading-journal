from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class Outcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    TIE = "TIE"


class TradeCreate(BaseModel):
    trade_time: datetime = Field(default_factory=datetime.utcnow)
    asset: str
    timeframe: str
    amount: float
    direction: Direction
    outcome: Outcome
    payout_pct: float = 0.0
    pnl: float = 0.0
    emotion: Optional[str] = None
    notes: Optional[str] = None


class Trade(TradeCreate):
    id: str
    created_at: datetime


class SessionCreate(BaseModel):
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_min: Optional[float] = None
    notes: Optional[str] = None


class Session(SessionCreate):
    id: str
    created_at: datetime
