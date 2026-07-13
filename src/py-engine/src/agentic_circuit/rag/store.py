"""Qdrant-backed vector memory with hybrid (dense + BM25 + ColBERT rerank) retrieval.

Per-circuit isolation: each circuit gets its own collection name. The synthesis
agent is given one retriever over all circuit collections.
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
VECTOR_SIZE = 384  # intfloat/multilingual-e5-small


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

    # -- connection -----------------------------------------------------
    def _ensure_qdrant(self):
        if self._qclient is not None:
            return self._qclient
        try:
            from qdrant_client import AsyncQdrantClient

            self._qclient = AsyncQdrantClient(url=self._qdrant_url, api_key=QDRANT_API_KEY or None)
        except Exception:
            self._qclient = None
        return self._qclient

    async def ensure_collection(self) -> None:
        q = self._ensure_qdrant()
        if q is None:
            return
        try:
            from qdrant_client.models import Distance, VectorParams

            if not await q.collection_exists(self.collection):
                await q.create_collection(
                    self.collection,
                    vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
                )
        except Exception:
            self._qclient = None

    # -- write ----------------------------------------------------------
    async def upsert(self, text: str, doc_id: Optional[str] = None) -> str:
        doc_id = doc_id or str(uuid.uuid4())
        self._texts[doc_id] = text
        self._bm25.add(doc_id, text)
        vec = (await self._embed.embed([text]))[0]
        self._vecs[doc_id] = vec
        q = self._ensure_qdrant()
        if q is not None:
            try:
                from qdrant_client.models import PointStruct

                await q.upsert(self.collection, points=[PointStruct(id=doc_id, vector=vec, payload={"text": text})])
            except Exception:
                self._qclient = None
        return doc_id

    # -- read -----------------------------------------------------------
    async def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q = self._ensure_qdrant()
        qvec = (await self._embed.embed([query]))[0]
        if q is not None:
            try:
                hits = await q.search(self.collection, query_vector=qvec, limit=top_k)
                return [(h.id, float(h.score)) for h in hits]
            except Exception:
                self._qclient = None
        # in-memory fallback (cosine)
        out = []
        for did, vec in self._vecs.items():
            out.append((did, _cosine(qvec, vec)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]

    async def retrieve(self, query: str, top_k: int = 5, use_rerank: bool = True) -> list[str]:
        dense = await self._dense_search(query, top_k * 2)
        lexical = self._bm25.search(query, top_k * 2)
        # reciprocal rank fusion
        fused: dict[str, float] = {}
        for rank, (did, _) in enumerate(dense):
            fused[did] = fused.get(did, 0.0) + 1.0 / (rank + 1)
        for rank, (did, _) in enumerate(lexical):
            fused[did] = fused.get(did, 0.0) + 1.0 / (rank + 1)
        ordered = sorted(fused.keys(), key=lambda d: fused[d], reverse=True)
        docs = [self._texts[d] for d in ordered[: top_k * 2]]
        if use_rerank and self._rerank and docs:
            try:
                ranked = await self._rerank.rerank(query, docs, top_n=top_k)
                # rerank returns (doc_text, score) when aligned
                if ranked and ranked[0][0]:
                    return [d for d, _ in ranked]
                # else scores aligned to docs by index
                order = sorted(range(len(docs)), key=lambda i: ranked[i][1], reverse=True)
                return [docs[i] for i in order[:top_k]]
            except Exception:
                pass
        return docs[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class NullMemory:
    """Memory that does nothing (used for circuits/tools with rag disabled)."""

    async def ensure_collection(self) -> None: ...

    async def upsert(self, text: str, doc_id: Optional[str] = None) -> str:
        return doc_id or ""

    async def retrieve(self, query: str, top_k: int = 5, use_rerank: bool = True) -> list[str]:
        return []
