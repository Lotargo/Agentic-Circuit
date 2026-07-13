"""Embedding client for a Text Embeddings Inference sidecar."""

from __future__ import annotations

import os
from typing import Literal

import httpx

EMBEDDING_SIDECAR_URL = os.environ.get("EMBEDDING_SIDECAR_URL", "http://localhost:8899")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

InputType = Literal["query", "passage"]


class EmbeddingClient:
    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.url = (url or EMBEDDING_SIDECAR_URL).rstrip("/")
        # A TEI process serves one model selected at startup. The value is used
        # here to apply model-family-specific input formatting.
        self.model = model or EMBEDDING_MODEL
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    def _prepare(self, texts: list[str], input_type: InputType) -> list[str]:
        if "e5" not in self.model.lower():
            return texts
        prefix = "query: " if input_type == "query" else "passage: "
        return [text if text.startswith(prefix) else f"{prefix}{text}" for text in texts]

    async def embed(
        self,
        texts: list[str],
        *,
        input_type: InputType = "passage",
    ) -> list[list[float]]:
        if not texts:
            return []

        response = await self._client.post(
            f"{self.url}/embed",
            json={"inputs": self._prepare(texts, input_type)},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings") if isinstance(data, dict) else data
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError("embedding sidecar returned an invalid batch")
        return embeddings

    async def aclose(self) -> None:
        await self._client.aclose()
