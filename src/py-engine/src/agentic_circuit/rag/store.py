"""Scoped Qdrant memory with dense + BM25 retrieval and final reranking.

Persistent retrieval is disabled when no stable user scope is available. Every
stored point carries provenance, and legacy unscoped points are intentionally
ignored to prevent cross-user context leakage.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional

from .bm25 import BM25Index
from .embeddings import EmbeddingClient
from .rerank import RerankClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6633")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
VECTOR_SIZE = int(os.environ.get("EMBEDDING_VECTOR_SIZE", "384"))
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
DENSE_WEIGHT = float(os.environ.get("RAG_DENSE_WEIGHT", "0.6"))
LEXICAL_WEIGHT = float(os.environ.get("RAG_LEXICAL_WEIGHT", "0.4"))
MIN_DENSE_SCORE = float(os.environ.get("RAG_MIN_DENSE_SCORE", "0.0"))


@dataclass(frozen=True)
class MemoryHit:
    doc_id: str
    text: str
    score: float
    collection: str
    scope: str
    kind: str = "memory"
    source: str = "unknown"
    query: str = ""
    prism: str = "neutral"
    created_at: str = ""

    def prompt_text(self) -> str:
        metadata = [f"source={self.source}", f"kind={self.kind}"]
        if self.created_at:
            metadata.append(f"created_at={self.created_at}")
        if self.prism:
            metadata.append(f"prism={self.prism}")
        lines = [f"[{', '.join(metadata)}]"]
        if self.query:
            lines.append(f"Прошлый запрос: {self.query}")
        lines.append(f"Прошлый вывод: {self.text}")
        return "\n".join(lines)

    def rerank_text(self) -> str:
        return f"Запрос: {self.query}\nВывод: {self.text}" if self.query else self.text


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
        self._records: dict[str, dict[str, MemoryHit]] = {}
        self._vectors: dict[str, dict[str, list[float]]] = {}
        self._bm25: dict[str, BM25Index] = {}
        self._hydrated_scopes: set[str] = set()
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
                timeout=10.0,
            )
        except Exception:
            self._qclient = None
        return self._qclient

    @staticmethod
    def _scope_filter(scope: str):
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="scope",
                    match=models.MatchValue(value=scope),
                )
            ]
        )

    def _remember(self, hit: MemoryHit, vector: list[float] | None = None) -> None:
        records = self._records.setdefault(hit.scope, {})
        records[hit.doc_id] = hit
        self._bm25.setdefault(hit.scope, BM25Index()).add(
            hit.doc_id,
            hit.rerank_text(),
        )
        if vector is not None:
            self._vectors.setdefault(hit.scope, {})[hit.doc_id] = vector

    @staticmethod
    def _point_to_hit(collection: str, point, scope: str, score: float = 0.0) -> MemoryHit | None:
        payload = point.payload or {}
        if not isinstance(payload, dict):
            return None
        text = payload.get("text")
        point_scope = payload.get("scope")
        if not isinstance(text, str) or point_scope != scope:
            return None
        return MemoryHit(
            doc_id=str(point.id),
            text=text,
            score=score,
            collection=collection,
            scope=scope,
            kind=str(payload.get("kind") or "memory"),
            source=str(payload.get("source") or collection),
            query=str(payload.get("query") or ""),
            prism=str(payload.get("prism") or "neutral"),
            created_at=str(payload.get("created_at") or ""),
        )

    async def ensure_collection(self) -> None:
        client = self._ensure_qdrant()
        if client is None:
            return
        try:
            from qdrant_client import models

            if not await client.collection_exists(self.collection):
                await client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self._vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            try:
                await client.create_payload_index(
                    collection_name=self.collection,
                    field_name="scope",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Existing indexes and older Qdrant versions are both harmless.
                pass
        except Exception:
            self._qclient = None

    async def _ensure_scope_loaded(self, scope: str) -> None:
        if not scope or scope in self._hydrated_scopes:
            return
        client = self._ensure_qdrant()
        if client is None:
            return
        loaded: list[MemoryHit] = []
        try:
            offset = None
            while True:
                points, offset = await client.scroll(
                    collection_name=self.collection,
                    scroll_filter=self._scope_filter(scope),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    hit = self._point_to_hit(self.collection, point, scope)
                    if hit is not None:
                        loaded.append(hit)
                if offset is None:
                    break
        except Exception:
            self._qclient = None
            return

        records = self._records.setdefault(scope, {})
        records.update({hit.doc_id: hit for hit in loaded})
        self._bm25.setdefault(scope, BM25Index()).add_many(
            (hit.doc_id, hit.rerank_text()) for hit in loaded
        )
        self._hydrated_scopes.add(scope)

    async def upsert(
        self,
        text: str,
        *,
        scope: str | None,
        kind: str = "memory",
        source: str | None = None,
        query: str = "",
        prism: str = "neutral",
        doc_id: Optional[str] = None,
    ) -> str:
        text = text.strip()
        if not scope or not text:
            return ""

        source = source or self.collection
        fingerprint = "\x1f".join(
            [self.collection, scope, kind, source, query, prism, text]
        )
        doc_id = doc_id or str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint))
        embedding_text = f"Запрос: {query}\nВывод: {text}" if query else text
        vector = (
            await self._embed.embed([embedding_text], input_type="passage")
        )[0]
        if len(vector) != self._vector_size:
            raise ValueError(
                f"embedding dimension {len(vector)} does not match "
                f"EMBEDDING_VECTOR_SIZE={self._vector_size}"
            )

        hit = MemoryHit(
            doc_id=doc_id,
            text=text,
            score=0.0,
            collection=self.collection,
            scope=scope,
            kind=kind,
            source=source,
            query=query,
            prism=prism,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._remember(hit, vector)

        client = self._ensure_qdrant()
        if client is not None:
            try:
                from qdrant_client import models

                await client.upsert(
                    collection_name=self.collection,
                    points=[
                        models.PointStruct(
                            id=doc_id,
                            vector=vector,
                            payload={
                                "text": hit.text,
                                "scope": hit.scope,
                                "kind": hit.kind,
                                "source": hit.source,
                                "query": hit.query,
                                "prism": hit.prism,
                                "created_at": hit.created_at,
                            },
                        )
                    ],
                )
            except Exception:
                self._qclient = None
        return doc_id

    async def _dense_search(
        self,
        query: str,
        scope: str,
        top_k: int,
    ) -> list[MemoryHit]:
        query_vector = (
            await self._embed.embed([query], input_type="query")
        )[0]
        if len(query_vector) != self._vector_size:
            raise ValueError(
                f"embedding dimension {len(query_vector)} does not match "
                f"EMBEDDING_VECTOR_SIZE={self._vector_size}"
            )

        client = self._ensure_qdrant()
        if client is not None:
            try:
                result = await client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    query_filter=self._scope_filter(scope),
                    limit=top_k,
                    with_payload=True,
                )
                hits: list[MemoryHit] = []
                for point in result.points:
                    score = float(point.score or 0.0)
                    if score < MIN_DENSE_SCORE:
                        continue
                    hit = self._point_to_hit(self.collection, point, scope, score)
                    if hit is not None:
                        self._remember(hit)
                        hits.append(hit)
                return hits
            except Exception:
                self._qclient = None

        local = []
        for doc_id, vector in self._vectors.get(scope, {}).items():
            score = _cosine(query_vector, vector)
            if score >= MIN_DENSE_SCORE and doc_id in self._records.get(scope, {}):
                local.append(replace(self._records[scope][doc_id], score=score))
        local.sort(key=lambda item: item.score, reverse=True)
        return local[:top_k]

    async def retrieve(
        self,
        query: str,
        *,
        scope: str | None,
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> list[MemoryHit]:
        if not scope or not query.strip() or top_k <= 0:
            return []
        await self._ensure_scope_loaded(scope)

        candidate_limit = max(top_k * 3, top_k)
        dense = await self._dense_search(query, scope, candidate_limit)
        lexical = self._bm25.get(scope, BM25Index()).search(
            query,
            candidate_limit,
        )

        fused: dict[str, float] = {}
        for rank, hit in enumerate(dense, 1):
            fused[hit.doc_id] = fused.get(hit.doc_id, 0.0) + (
                DENSE_WEIGHT / (RRF_K + rank)
            )
        for rank, (doc_id, _) in enumerate(lexical, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + (
                LEXICAL_WEIGHT / (RRF_K + rank)
            )

        records = self._records.get(scope, {})
        candidates = [
            replace(records[doc_id], score=score)
            for doc_id, score in sorted(
                fused.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if doc_id in records
        ]
        candidates = candidates[:candidate_limit]

        if use_rerank and self._rerank and candidates:
            try:
                ranked = await self._rerank.rerank_indices(
                    query,
                    [hit.rerank_text() for hit in candidates],
                    top_n=top_k,
                )
                return [
                    replace(candidates[index], score=score)
                    for index, score in ranked
                    if 0 <= index < len(candidates)
                ]
            except Exception:
                pass
        return candidates[:top_k]

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
    async def ensure_collection(self) -> None: ...

    async def upsert(self, text: str, *, scope: str | None = None, **_metadata) -> str:
        return ""

    async def retrieve(
        self,
        query: str,
        *,
        scope: str | None = None,
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> list[MemoryHit]:
        return []

    async def aclose(self) -> None: ...
