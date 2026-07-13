"""HTTP client for a Text Embeddings Inference cross-encoder reranker.

TEI's ``/rerank`` endpoint accepts ``query`` + ``texts``. It is not a ColBERT
late-interaction endpoint: the sidecar must serve a supported sequence
classification reranker such as ``Alibaba-NLP/gte-multilingual-reranker-base``.
"""

from __future__ import annotations

import os

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
        # Kept for diagnostics/config parity. A TEI container serves one model,
        # selected at process startup rather than per request.
        self.model = model or RERANK_MODEL
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
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

        ranked: list[tuple[str, float]] = []
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("rerank sidecar returned a non-object result")

            score = float(item.get("score", 0.0))
            text = item.get("text") or item.get("document")
            if not isinstance(text, str):
                index = item.get("index")
                if isinstance(index, int) and 0 <= index < len(documents):
                    text = documents[index]
                else:
                    raise ValueError("rerank result has neither text nor a valid index")
            ranked.append((text, score))

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n] if top_n is not None else ranked

    async def aclose(self) -> None:
        await self._client.aclose()
