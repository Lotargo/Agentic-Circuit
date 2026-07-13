"""langsearch web-search tool, used ONLY by the synthesis agent."""

from __future__ import annotations

import os

import httpx

LANGSEARCH_API_KEY = os.environ.get("LANGSEARCH_API_KEY", "")
LANGSEARCH_URL = "https://api.langsearch.com/v1/web-search"


class WebSearchTool:
    """Thin wrapper over langsearch for synthesis-time web lookup + rerank."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = api_key if api_key is not None else LANGSEARCH_API_KEY
        self._client = httpx.AsyncClient(timeout=timeout)

    async def search(self, query: str, top_k: int = 5) -> list[str]:
        if not self.api_key:
            return []
        resp = await self._client.get(
            LANGSEARCH_URL,
            params={"query": query, "count": top_k},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", data.get("data", []))
        return [r.get("content") or r.get("snippet") or "" for r in results][:top_k]

    async def aclose(self) -> None:
        await self._client.aclose()
