"""ColBERT late-interaction rerank client (Answer.AI ColBERT small v1 sidecar)."""

from __future__ import annotations

import os

import httpx

RERANK_SIDECAR_URL = os.environ.get("RERANK_SIDECAR_URL", "http://localhost:8898")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "answerdotai/colbert-small-v1")


class RerankClient:
    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 30.0):
        self.url = (url or RERANK_SIDECAR_URL).rstrip("/")
        self.model = model or RERANK_MODEL
        self._client = httpx.AsyncClient(timeout=timeout)

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[tuple[str, float]]:
        payload = {"query": query, "documents": documents, "model": self.model}
        if top_n:
            payload["top_n"] = top_n
        resp = await self._client.post(f"{self.url}/rerank", json=payload)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", data) if isinstance(data, dict) else data
        out: list[tuple[str, float]] = []
        for r in results:
            if isinstance(r, dict):
                out.append((r.get("document", r.get("text", "")), float(r.get("score", 0.0))))
            else:  # bare list of scores aligned with input order
                out.append(("", float(r)))
        if not any(text for text, _ in out) and documents:
            out = list(zip(documents, [s for _, s in out])) if out else [(d, 0.0) for d in documents]
        return out

    async def aclose(self) -> None:
        await self._client.aclose()
