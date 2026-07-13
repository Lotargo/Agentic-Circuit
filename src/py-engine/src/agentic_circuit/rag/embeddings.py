"""Embedding client for the TEI/vLLM sidecar (E5 small dense embeddings)."""

from __future__ import annotations

import os

import httpx

EMBEDDING_SIDECAR_URL = os.environ.get("EMBEDDING_SIDECAR_URL", "http://localhost:8899")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")


class EmbeddingClient:
    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 30.0):
        self.url = (url or EMBEDDING_SIDECAR_URL).rstrip("/")
        self.model = model or EMBEDDING_MODEL
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """TEI /embed endpoint: {"inputs": [...]} -> {"embeddings": [[...]]}."""
        resp = await self._client.post(
            f"{self.url}/embed",
            json={"inputs": texts, "model": self.model},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]
        return data  # some TEI versions return a bare list

    async def aclose(self) -> None:
        await self._client.aclose()
