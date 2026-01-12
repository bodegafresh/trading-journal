from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str
    schema: str = "public"
    # timeouts HTTP: (connect, read)
    timeout: Tuple[float, float] = (5.0, 30.0)
    # pool/retry
    pool_connections: int = 20
    pool_maxsize: int = 20
    retries_total: int = 6
    backoff_factor: float = 0.5


def _build_retry(cfg: SupabaseConfig) -> Retry:
    return Retry(
        total=cfg.retries_total,
        connect=cfg.retries_total,
        read=cfg.retries_total,
        status=cfg.retries_total,
        backoff_factor=cfg.backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST", "PATCH", "DELETE"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )


class SupabaseClient:
    """Cliente mínimo para PostgREST de Supabase (HTTP)."""

    def __init__(self, cfg: SupabaseConfig):
        self.cfg = cfg
        self.base = cfg.url.rstrip("/") + "/rest/v1"

        # ✅ Session + retry + pool (evita RemoteDisconnected y reduce overhead)
        self.session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=_build_retry(cfg),
            pool_connections=cfg.pool_connections,
            pool_maxsize=cfg.pool_maxsize,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update(
            {
                "apikey": cfg.key,
                "Authorization": f"Bearer {cfg.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Profile": cfg.schema,
                # opcional, pero ayuda a mantener conexiones
                "Connection": "keep-alive",
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

        r = self.session.get(self._url(table), params=params, timeout=self.cfg.timeout)
        r.raise_for_status()
        data = r.json()
        # PostgREST siempre debería devolver lista, pero lo aseguramos
        return data if isinstance(data, list) else [data]

    def insert(self, table: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        headers = {"Prefer": "return=representation"}
        r = self.session.post(
            self._url(table), json=rows, headers=headers, timeout=self.cfg.timeout
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else [data]

    def patch(
        self, table: str, filters: Dict[str, str], patch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        headers = {"Prefer": "return=representation"}
        r = self.session.patch(
            self._url(table),
            params=filters,
            json=patch,
            headers=headers,
            timeout=self.cfg.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else [data]


def load_supabase_from_env() -> SupabaseClient:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("Falta SUPABASE_URL o SUPABASE_KEY en el entorno (.env).")
    return SupabaseClient(SupabaseConfig(url=url, key=key))
