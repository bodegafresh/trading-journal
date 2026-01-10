from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str
    schema: str = "public"


class SupabaseClient:
    """Cliente mínimo para PostgREST de Supabase (HTTP).

    Evitamos drivers SQL para que:
    - funcione igual en Streamlit
    - sea consistente con Apps Script (también HTTP)
    """

    def __init__(self, cfg: SupabaseConfig):
        self.cfg = cfg
        self.base = cfg.url.rstrip("/") + "/rest/v1"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "apikey": cfg.key,
                "Authorization": f"Bearer {cfg.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Profile": cfg.schema,
            }
        )

    def _url(self, table: str) -> str:
        return f"{self.base}/{table}"

    def select(
        self,
        table: str,
        *,
        select: str = "*",
        filters: Optional[Dict[str, str]] = None,
        order: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, str] = {"select": select}
        if filters:
            params.update(filters)
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(limit)

        r = self.session.get(self._url(table), params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def insert(self, table: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # return=representation devuelve los registros insertados
        headers = {"Prefer": "return=representation"}
        r = self.session.post(self._url(table), json=rows, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def patch(
        self, table: str, filters: Dict[str, str], patch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        headers = {"Prefer": "return=representation"}
        r = self.session.patch(self._url(table), params=filters, json=patch, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()


def load_supabase_from_env() -> SupabaseClient:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("Falta SUPABASE_URL o SUPABASE_KEY en el entorno (.env).")
    return SupabaseClient(SupabaseConfig(url=url, key=key))
