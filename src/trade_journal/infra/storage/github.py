from __future__ import annotations

import base64
import os
import re
import requests
from datetime import date

from .base import EvidenceStorage


def _sanitize(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9._-]+", "", text)


class GitHubPagesStorage(EvidenceStorage):
    def __init__(
        self,
        *,
        owner: str,
        repo: str,
        token: str,
        pages_base_url: str,
    ):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.pages_base_url = pages_base_url.rstrip("/")

    def build_path(
        self,
        *,
        trade_date: date,
        asset: str,
        filename: str,
    ) -> str:
        yyyy = f"{trade_date.year:04d}"
        mm = f"{trade_date.month:02d}"
        asset_clean = _sanitize(asset.replace("/", "_"))
        fname = _sanitize(filename)

        return f"imagenes/{yyyy}/{mm}/{asset_clean}/{fname}"

    def _get_sha_if_exists(self, path: str) -> str | None:
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{path}"
        r = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("sha")
        if r.status_code == 404:
            return None
        raise RuntimeError(f"GitHub GET error {r.status_code}: {r.text}")

    def upload(
        self,
        *,
        path: str,
        content: bytes,
        overwrite: bool = False,
    ) -> str:
        sha = self._get_sha_if_exists(path) if overwrite else None

        payload = {
            "message": f"Upload evidence {path}",
            "content": base64.b64encode(content).decode("utf-8"),
        }
        if sha:
            payload["sha"] = sha

        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{path}"
        r = requests.put(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Accept": "application/vnd.github+json",
            },
            timeout=60,
        )

        if not r.ok:
            raise RuntimeError(f"GitHub PUT error {r.status_code}: {r.text}")

        return f"{self.pages_base_url}/{path}"
