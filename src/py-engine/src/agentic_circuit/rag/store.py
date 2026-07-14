"""Scoped Qdrant memory with hybrid retrieval and lifecycle-aware ranking.

Persistent retrieval is disabled when no stable user scope is available. Every
new point carries a logical memory type, project/conversation namespaces,
confidence, importance, expiry and supersession metadata. Legacy unscoped points
are intentionally ignored to prevent cross-user context leakage.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

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
QDRANT_RETRY_SECONDS = float(os.environ.get("QDRANT_RETRY_SECONDS", "30"))
MAX_MEMORY_CHARS = int(os.environ.get("RAG_MAX_MEMORY_CHARS", "6000"))
MAX_QUERY_CHARS = int(os.environ.get("RAG_MAX_QUERY_CHARS", "2000"))
MAX_SCOPE_RECORDS = int(os.environ.get("RAG_MAX_SCOPE_RECORDS", "2000"))

_SOURCE_QUALITY = {
    "user_explicit": 1.0,
    "user_correction": 1.0,
    "project_decision": 0.95,
    "assistant_verified": 0.72,
    "synthesis": 0.62,
}
_HALF_LIFE_DAYS = {
    "temporary_context": 7.0,
    "project_state": 30.0,
    "assistant_conclusion": 45.0,
    "relationship_context": 180.0,
    "project_decision": 365.0,
    "user_preference": 730.0,
    "negative_preference": 730.0,
    "user_fact": 1460.0,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


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
    updated_at: str = ""
    memory_type: str = "assistant_conclusion"
    canonical_key: str = ""
    project_id: str = ""
    conversation_id: str = ""
    confidence: float = 0.7
    importance: float = 0.5
    source_quality: float = 0.7
    status: str = "active"
    expires_at: str = ""
    superseded_by: str = ""

    def prompt_text(self) -> str:
        metadata = [
            f"source={self.source}",
            f"type={self.memory_type}",
            f"confidence={self.confidence:.2f}",
            f"importance={self.importance:.2f}",
        ]
        if self.created_at:
            metadata.append(f"created_at={self.created_at}")
        if self.project_id:
            metadata.append("project=scoped")
        lines = [f"[{', '.join(metadata)}]"]
        if self.query:
            lines.append(f"Прошлый контекст: {self.query}")
        lines.append(f"Память: {self.text}")
        return "\n".join(lines)

    def rerank_text(self) -> str:
        prefix = f"Тип памяти: {self.memory_type}. "
        if self.query:
            return f"{prefix}Контекст: {self.query}\nПамять: {self.text}"
        return f"{prefix}{self.text}"

    def is_active(self, now: datetime | None = None) -> bool:
        if self.status != "active":
            return False
        expires = _parse_datetime(self.expires_at)
        return not expires or expires > (now or _utcnow())


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
        self._qdrant_retry_after = 0.0

    def _mark_qdrant_failure(self) -> None:
        self._qclient = None
        self._qdrant_retry_after = time.monotonic() + QDRANT_RETRY_SECONDS

    def _ensure_qdrant(self):
        if self._qclient is not None:
            return self._qclient
        if time.monotonic() < self._qdrant_retry_after:
            return None
        try:
            from qdrant_client import AsyncQdrantClient

            self._qclient = AsyncQdrantClient(
                url=self._qdrant_url,
                api_key=QDRANT_API_KEY or None,
                timeout=10.0,
            )
        except Exception:
            self._mark_qdrant_failure()
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
        index = self._bm25.setdefault(hit.scope, BM25Index())
        if hit.is_active():
            index.add(hit.doc_id, hit.rerank_text())
        else:
            index.remove(hit.doc_id)
        if vector is not None:
            self._vectors.setdefault(hit.scope, {})[hit.doc_id] = vector

    @staticmethod
    def _point_to_hit(
        collection: str,
        point,
        scope: str,
        score: float = 0.0,
    ) -> MemoryHit | None:
        payload = point.payload or {}
        if not isinstance(payload, dict):
            return None
        text = payload.get("text")
        point_scope = payload.get("scope")
        if not isinstance(text, str) or point_scope != scope:
            return None
        source = str(payload.get("source") or collection)
        return MemoryHit(
            doc_id=str(point.id),
            text=text[:MAX_MEMORY_CHARS],
            score=score,
            collection=collection,
            scope=scope,
            kind=str(payload.get("kind") or payload.get("memory_type") or "memory"),
            source=source,
            query=str(payload.get("query") or "")[:MAX_QUERY_CHARS],
            prism=str(payload.get("prism") or "neutral"),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or ""),
            memory_type=str(
                payload.get("memory_type")
                or ("assistant_conclusion" if source in {"synthesis", collection} else "memory")
            ),
            canonical_key=str(payload.get("canonical_key") or ""),
            project_id=str(payload.get("project_id") or ""),
            conversation_id=str(payload.get("conversation_id") or ""),
            confidence=float(payload.get("confidence") or 0.7),
            importance=float(payload.get("importance") or 0.5),
            source_quality=float(
                payload.get("source_quality") or _SOURCE_QUALITY.get(source, 0.7)
            ),
            status=str(payload.get("status") or "active"),
            expires_at=str(payload.get("expires_at") or ""),
            superseded_by=str(payload.get("superseded_by") or ""),
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
            for field_name in (
                "scope",
                "project_id",
                "conversation_id",
                "memory_type",
                "canonical_key",
                "status",
            ):
                try:
                    await client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
        except Exception:
            self._mark_qdrant_failure()

    async def _ensure_scope_loaded(self, scope: str) -> None:
        if not scope or scope in self._hydrated_scopes:
            return
        client = self._ensure_qdrant()
        if client is None:
            return
        loaded: list[MemoryHit] = []
        try:
            offset = None
            while len(loaded) < MAX_SCOPE_RECORDS:
                points, offset = await client.scroll(
                    collection_name=self.collection,
                    scroll_filter=self._scope_filter(scope),
                    limit=min(256, MAX_SCOPE_RECORDS - len(loaded)),
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
            self._mark_qdrant_failure()
            return

        for hit in loaded:
            self._remember(hit)
        self._hydrated_scopes.add(scope)

    async def _set_status(
        self,
        scope: str,
        doc_ids: Iterable[str],
        *,
        status: str,
        superseded_by: str = "",
    ) -> None:
        ids = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id]
        if not ids:
            return
        now = _utcnow().isoformat()
        records = self._records.setdefault(scope, {})
        for doc_id in ids:
            hit = records.get(doc_id)
            if hit is not None:
                self._remember(
                    replace(
                        hit,
                        status=status,
                        superseded_by=superseded_by,
                        updated_at=now,
                    )
                )
        client = self._ensure_qdrant()
        if client is not None:
            try:
                await client.set_payload(
                    collection_name=self.collection,
                    points=ids,
                    payload={
                        "status": status,
                        "superseded_by": superseded_by,
                        "updated_at": now,
                    },
                )
            except Exception:
                self._mark_qdrant_failure()

    async def supersede(self, scope: str, doc_ids: Iterable[str], new_doc_id: str) -> None:
        await self._set_status(
            scope,
            doc_ids,
            status="superseded",
            superseded_by=new_doc_id,
        )

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
        memory_type: str | None = None,
        canonical_key: str = "",
        project_id: str = "",
        conversation_id: str = "",
        confidence: float = 0.7,
        importance: float = 0.5,
        source_quality: float | None = None,
        ttl_days: int | None = None,
        supersede_existing: bool = True,
    ) -> str:
        text = text.strip()[:MAX_MEMORY_CHARS]
        query = query.strip()[:MAX_QUERY_CHARS]
        if not scope or not text:
            return ""

        source = source or self.collection
        memory_type = memory_type or kind or "memory"
        canonical_key = canonical_key.strip().lower()[:160]
        confidence = min(1.0, max(0.0, float(confidence)))
        importance = min(1.0, max(0.0, float(importance)))
        source_quality = min(
            1.0,
            max(0.0, float(source_quality or _SOURCE_QUALITY.get(source, 0.7))),
        )
        fingerprint = "\x1f".join(
            [
                self.collection,
                scope,
                project_id,
                memory_type,
                canonical_key,
                text,
            ]
        )
        doc_id = doc_id or str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint))
        now = _utcnow()
        expires_at = (
            (now + timedelta(days=ttl_days)).isoformat() if ttl_days else ""
        )

        await self._ensure_scope_loaded(scope)
        existing = self._records.get(scope, {})
        same_key = [
            hit.doc_id
            for hit in existing.values()
            if hit.doc_id != doc_id
            and hit.is_active(now)
            and canonical_key
            and hit.canonical_key == canonical_key
            and hit.project_id == project_id
        ]

        previous = existing.get(doc_id)
        created_at = previous.created_at if previous and previous.created_at else now.isoformat()
        hit = MemoryHit(
            doc_id=doc_id,
            text=text,
            score=0.0,
            collection=self.collection,
            scope=scope,
            kind=memory_type,
            source=source,
            query=query,
            prism=prism,
            created_at=created_at,
            updated_at=now.isoformat(),
            memory_type=memory_type,
            canonical_key=canonical_key,
            project_id=project_id,
            conversation_id=conversation_id,
            confidence=confidence,
            importance=importance,
            source_quality=source_quality,
            status="active",
            expires_at=expires_at,
        )
        embedding_text = hit.rerank_text()
        try:
            vector = (
                await self._embed.embed([embedding_text], input_type="passage")
            )[0]
            if len(vector) != self._vector_size:
                raise ValueError(
                    f"embedding dimension {len(vector)} does not match "
                    f"EMBEDDING_VECTOR_SIZE={self._vector_size}"
                )
        except Exception:
            self._remember(hit)
            if supersede_existing and same_key:
                await self.supersede(scope, same_key, doc_id)
            raise

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
                                "updated_at": hit.updated_at,
                                "memory_type": hit.memory_type,
                                "canonical_key": hit.canonical_key,
                                "project_id": hit.project_id,
                                "conversation_id": hit.conversation_id,
                                "confidence": hit.confidence,
                                "importance": hit.importance,
                                "source_quality": hit.source_quality,
                                "status": hit.status,
                                "expires_at": hit.expires_at,
                                "superseded_by": "",
                            },
                        )
                    ],
                )
            except Exception:
                self._mark_qdrant_failure()
        if supersede_existing and same_key:
            await self.supersede(scope, same_key, doc_id)
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
                    hit = self._point_to_hit(
                        self.collection,
                        point,
                        scope,
                        score,
                    )
                    if hit is not None:
                        self._remember(hit)
                        hits.append(hit)
                return hits
            except Exception:
                self._mark_qdrant_failure()

        local = []
        for doc_id, vector in self._vectors.get(scope, {}).items():
            score = _cosine(query_vector, vector)
            if score >= MIN_DENSE_SCORE and doc_id in self._records.get(scope, {}):
                local.append(replace(self._records[scope][doc_id], score=score))
        local.sort(key=lambda item: item.score, reverse=True)
        return local[:top_k]

    @staticmethod
    def _eligible(
        hit: MemoryHit,
        *,
        project_id: str,
        conversation_id: str,
        memory_types: set[str] | None,
        now: datetime,
    ) -> bool:
        if not hit.is_active(now):
            return False
        if memory_types and hit.memory_type not in memory_types:
            return False
        if project_id:
            if hit.project_id and hit.project_id != project_id:
                return False
        elif hit.project_id:
            return False
        if (
            hit.memory_type == "temporary_context"
            and hit.conversation_id
            and hit.conversation_id != conversation_id
        ):
            return False
        return True

    @staticmethod
    def _policy_factor(
        hit: MemoryHit,
        *,
        project_id: str,
        conversation_id: str,
        now: datetime,
    ) -> float:
        created = _parse_datetime(hit.updated_at or hit.created_at)
        if created:
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            half_life = _HALF_LIFE_DAYS.get(hit.memory_type, 180.0)
            freshness = 0.5 + 0.5 * (2 ** (-age_days / max(half_life, 1.0)))
        else:
            freshness = 0.7
        if project_id:
            project_match = 1.0 if hit.project_id == project_id else 0.72
        else:
            project_match = 1.0
        conversation_match = (
            1.0
            if not hit.conversation_id or hit.conversation_id == conversation_id
            else 0.88
        )
        confidence = 0.5 + 0.5 * hit.confidence
        importance = 0.5 + 0.5 * hit.importance
        return (
            confidence
            * importance
            * freshness
            * max(0.2, hit.source_quality)
            * project_match
            * conversation_match
        )

    async def retrieve(
        self,
        query: str,
        *,
        scope: str | None,
        project_id: str = "",
        conversation_id: str = "",
        memory_types: Iterable[str] | None = None,
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> list[MemoryHit]:
        query = query.strip()[:MAX_QUERY_CHARS]
        if not scope or not query or top_k <= 0:
            return []
        await self._ensure_scope_loaded(scope)

        candidate_limit = max(top_k * 4, 12)
        try:
            dense = await self._dense_search(query, scope, candidate_limit)
        except Exception:
            dense = []
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

        now = _utcnow()
        allowed_types = set(memory_types) if memory_types else None
        records = self._records.get(scope, {})
        candidates = [
            replace(records[doc_id], score=score)
            for doc_id, score in sorted(
                fused.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if doc_id in records
            and self._eligible(
                records[doc_id],
                project_id=project_id,
                conversation_id=conversation_id,
                memory_types=allowed_types,
                now=now,
            )
        ][:candidate_limit]

        if use_rerank and self._rerank and candidates:
            try:
                ranked = await self._rerank.rerank_indices(
                    query,
                    [hit.rerank_text() for hit in candidates],
                    top_n=candidate_limit,
                )
                candidates = [
                    replace(candidates[index], score=max(0.0, float(score)))
                    for index, score in ranked
                    if 0 <= index < len(candidates)
                ]
            except Exception:
                pass

        rescored = [
            replace(
                hit,
                score=hit.score
                * self._policy_factor(
                    hit,
                    project_id=project_id,
                    conversation_id=conversation_id,
                    now=now,
                ),
            )
            for hit in candidates
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]

    async def aclose(self) -> None:
        client = self._qclient
        self._qclient = None
        self._qdrant_retry_after = 0.0
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
        project_id: str = "",
        conversation_id: str = "",
        memory_types: Iterable[str] | None = None,
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> list[MemoryHit]:
        return []

    async def aclose(self) -> None: ...
