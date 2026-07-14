"""HTTP client for a Text Embeddings Inference cross-encoder reranker."""

from __future__ import annotations

import os
from collections import defaultdict, deque

import httpx

RERANK_SIDECAR_URL = os.environ.get("RERANK_SIDECAR_URL", "http://localhost:8898")
RERANK_MODEL = os.environ.get(
    "RERANK_MODEL",
    "Alibaba-NLP/gte-multilingual-reranker-base",
)


class RerankClient:
    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.url = (url or RERANK_SIDECAR_URL).rstrip("/")
        self.model = model or RERANK_MODEL
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def rerank_indices(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        response = await self._client.post(
            f"{self.url}/rerank",
            json={
                "query": query,
                "texts": documents,
                "return_text": True,
                "raw_scores": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(results, list):
            raise ValueError("rerank sidecar returned an unsupported response")

        by_text: dict[str, deque[int]] = defaultdict(deque)
        for index, document in enumerate(documents):
            by_text[document].append(index)

        ranked: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("rerank sidecar returned a non-object result")
            score = float(item.get("score", 0.0))
            index = item.get("index")
            if not isinstance(index, int):
                text = item.get("text") or item.get("document")
                if not isinstance(text, str) or not by_text[text]:
                    raise ValueError("rerank result has neither index nor known text")
                index = by_text[text].popleft()
            if 0 <= index < len(documents):
                ranked.append((index, score))

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n] if top_n is not None else ranked

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
        ranked = await self.rerank_indices(query, documents, top_n)
        return [(documents[index], score) for index, score in ranked]

    async def aclose(self) -> None:
        await self._client.aclose()
