"""Qdrant-backed vector memory with dense + BM25 retrieval and reranking.

Each circuit has an isolated collection. Text payloads are restored from
Qdrant at startup so both dense retrieval and the local BM25 component survive
Python process restarts.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from .bm25 import BM25Index
from .embeddings import EmbeddingClient
from .rerank import RerankClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6633")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
VECTOR_SIZE = int(os.environ.get("EMBEDDING_VECTOR_SIZE", "384"))


class VectorMemory:
    def __init__(
        self,
        collection: str,
        embedding_client: EmbeddingClient,
        rerank_client: Optional[RerankClient] = None,
        qdrant_url: str | None = None,
        vector_size: int = VECTOR_SIZE,
    ):
        self.collection = collection
        self._embed = embedding_client
        self._rerank = rerank_client
        self._vector_size = vector_size
        self._bm25 = BM25Index()
        self._texts: dict[str, str] = {}
        self._vecs: dict[str, list[float]] = {}
        self._qclient = None
        self._qdrant_url = qdrant_url or QDRANT_URL

    def _ensure_qdrant(self):
        if self._qclient is not None:
            return self._qclient
        try:
            from qdrant_client import AsyncQdrantClient

            self._qclient = AsyncQdrantClient(
                url=self._qdrant_url,
                api_key=QDRANT_API_KEY or None,
            )
        except Exception:
            self._qclient = None
        return self._qclient

    def _remember_text(self, doc_id: str, text: str) -> None:
        self._texts[doc_id] = text
        self._bm25.add(doc_id, text)

    async def _hydrate_payloads(self, client) -> None:
        """Restore all text payloads needed by local BM25 after a restart."""
        offset = None
        while True:
            points, offset = await client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                text = payload.get("text") if isinstance(payload, dict) else None
                if isinstance(text, str):
                    self._remember_text(str(point.id), text)
            if offset is None:
                break

    async def ensure_collection(self) -> None:
        client = self._ensure_qdrant()
        if client is None:
            return
        try:
            from qdrant_client.models import Distance, VectorParams

            if not await client.collection_exists(self.collection):
                await client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
            await self._hydrate_payloads(client)
        except Exception:
            self._qclient = None

    async def upsert(self, text: str, doc_id: Optional[str] = None) -> str:
        doc_id = doc_id or str(uuid.uuid4())
        vector = (
            await self._embed.embed([text], input_type="passage")
        )[0]
        if len(vector) != self._vector_size:
            raise ValueError(
                f"embedding dimension {len(vector)} does not match "
                f"EMBEDDING_VECTOR_SIZE={self._vector_size}"
            )

        self._remember_text(doc_id, text)
        self._vecs[doc_id] = vector

        client = self._ensure_qdrant()
        if client is not None:
            try:
                from qdrant_client.models import PointStruct

                await client.upsert(
                    collection_name=self.collection,
                    points=[
                        PointStruct(
                            id=doc_id,
                            vector=vector,
                            payload={"text": text},
                        )
                    ],
                )
            except Exception:
                self._qclient = None
        return doc_id

    async def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        client = self._ensure_qdrant()
        query_vector = (
            await self._embed.embed([query], input_type="query")
        )[0]
        if len(query_vector) != self._vector_size:
            raise ValueError(
                f"embedding dimension {len(query_vector)} does not match "
                f"EMBEDDING_VECTOR_SIZE={self._vector_size}"
            )

        if client is not None:
            try:
                hits = await client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                results: list[tuple[str, float]] = []
                for hit in hits:
                    doc_id = str(hit.id)
                    payload = hit.payload or {}
                    text = payload.get("text") if isinstance(payload, dict) else None
                    if isinstance(text, str):
                        self._remember_text(doc_id, text)
                    results.append((doc_id, float(hit.score)))
                return results
            except Exception:
                self._qclient = None

        results = [
            (doc_id, _cosine(query_vector, vector))
            for doc_id, vector in self._vecs.items()
        ]
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> list[str]:
        dense = await self._dense_search(query, top_k * 2)
        lexical = self._bm25.search(query, top_k * 2)

        fused: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(dense):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank + 1)
        for rank, (doc_id, _) in enumerate(lexical):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rank + 1)

        ordered_ids = sorted(fused, key=fused.__getitem__, reverse=True)
        documents = [
            self._texts[doc_id]
            for doc_id in ordered_ids[: top_k * 2]
            if doc_id in self._texts
        ]

        if use_rerank and self._rerank and documents:
            try:
                ranked = await self._rerank.rerank(
                    query,
                    documents,
                    top_n=top_k,
                )
                return [document for document, _ in ranked]
            except Exception:
                pass
        return documents[:top_k]

    async def aclose(self) -> None:
        client = self._qclient
        self._qclient = None
        if client is not None:
            close = getattr(client, "close", None)
            if close is not None:
                result = close()
                if hasattr(result, "__await__"):
                    await result


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(left * right for left, right in zip(a, b))
    norm_a = sum(value * value for value in a) ** 0.5
    norm_b = sum(value * value for value in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class NullMemory:
    """Memory that does nothing (used when RAG is disabled)."""

    async def ensure_collection(self) -> None: ...

    async def upsert(self, text: str, doc_id: Optional[str] = None) -> str:
        return doc_id or ""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> list[str]:
        return []

    async def aclose(self) -> None: ...
