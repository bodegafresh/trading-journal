from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date


class EvidenceStorage(ABC):
    @abstractmethod
    def build_path(
        self,
        *,
        trade_date: date,
        asset: str,
        filename: str,
    ) -> str:
        pass

    @abstractmethod
    def upload(
        self,
        *,
        path: str,
        content: bytes,
        overwrite: bool = False,
    ) -> str:
        """
        Sube el archivo y devuelve la URL pública/final.
        """
        pass
