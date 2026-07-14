from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agentic_circuit.rag.store import VectorMemory


class FakeEmbeddingClient:
    def __init__(self):
        self.input_types: list[str] = []

    async def embed(self, texts, *, input_type="passage"):
        self.input_types.append(input_type)
        return [[1.0, 0.0] for _ in texts]


class FailingQueryEmbeddingClient(FakeEmbeddingClient):
    async def embed(self, texts, *, input_type="passage"):
        self.input_types.append(input_type)
        if input_type == "query":
            raise RuntimeError("embedding sidecar unavailable")
        return [[1.0, 0.0] for _ in texts]


class FakeQdrantClient:
    def __init__(self, points=None, collection="memory"):
        self.points = list(points or [])
        self.collection = collection
        self.last_scope = None

    @staticmethod
    def _scope(filter_value):
        return filter_value.must[0].match.value

    async def scroll(
        self,
        *,
        collection_name,
        scroll_filter,
        limit,
        offset,
        with_payload,
        with_vectors,
    ):
        assert collection_name == self.collection
        assert with_payload is True
        assert with_vectors is False
        scope = self._scope(scroll_filter)
        self.last_scope = scope
        return (
            [point for point in self.points if point.payload.get("scope") == scope][:limit],
            None,
        )

    async def query_points(
        self,
        *,
        collection_name,
        query,
        query_filter,
        limit,
        with_payload,
    ):
        assert collection_name == self.collection
        assert with_payload is True
        scope = self._scope(query_filter)
        self.last_scope = scope
        points = [
            SimpleNamespace(
                id=point.id,
                score=getattr(point, "score", 0.99),
                payload=point.payload,
            )
            for point in self.points
            if point.payload.get("scope") == scope
        ]
        return SimpleNamespace(points=points[:limit])

    async def upsert(self, *, collection_name, points):
        assert collection_name == self.collection
        for point in points:
            self.points = [existing for existing in self.points if existing.id != point.id]
            self.points.append(point)

    async def set_payload(self, *, collection_name, points, payload):
        assert collection_name == self.collection
        ids = {str(value) for value in points}
        for point in self.points:
            if str(point.id) in ids:
                point.payload.update(payload)


def point(
    doc_id: str,
    scope: str | None,
    text: str,
    *,
    project_id: str = "",
    conversation_id: str = "",
    memory_type: str = "user_preference",
    canonical_key: str = "user.preference.example",
    status: str = "active",
    confidence: float = 0.9,
    importance: float = 0.7,
    created_at: str = "2026-07-14T00:00:00+00:00",
    expires_at: str = "",
):
    payload = {
        "text": text,
        "kind": memory_type,
        "memory_type": memory_type,
        "canonical_key": canonical_key,
        "source": "user_explicit",
        "source_quality": 1.0,
        "query": "старый запрос",
        "prism": "neutral",
        "project_id": project_id,
        "conversation_id": conversation_id,
        "confidence": confidence,
        "importance": importance,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "expires_at": expires_at,
    }
    if scope is not None:
        payload["scope"] = scope
    return SimpleNamespace(id=doc_id, payload=payload, score=0.99)


async def test_retrieve_is_filtered_by_user_and_project_scope():
    embeddings = FakeEmbeddingClient()
    qdrant = FakeQdrantClient(
        [
            point("project-a", "scope-a", "память проекта A", project_id="project-a"),
            point("global", "scope-a", "глобальное предпочтение"),
            point("project-b", "scope-a", "память проекта B", project_id="project-b"),
            point("other-user", "scope-b", "чужая память", project_id="project-a"),
            point("legacy", None, "старая глобальная память"),
        ]
    )
    memory = VectorMemory("memory", embeddings, vector_size=2)
    memory._qclient = qdrant

    result = await memory.retrieve(
        "память предпочтение",
        scope="scope-a",
        project_id="project-a",
        use_rerank=False,
        top_k=10,
    )

    assert {hit.doc_id for hit in result} == {"project-a", "global"}
    assert qdrant.last_scope == "scope-a"
    assert embeddings.input_types == ["query"]


async def test_query_without_project_does_not_leak_project_specific_memory():
    memory = VectorMemory("memory", FakeEmbeddingClient(), vector_size=2)
    memory._qclient = FakeQdrantClient(
        [
            point("global", "scope-a", "общее предпочтение"),
            point("project", "scope-a", "секрет проекта", project_id="project-a"),
        ]
    )
    result = await memory.retrieve(
        "предпочтение секрет",
        scope="scope-a",
        use_rerank=False,
        top_k=10,
    )
    assert [hit.doc_id for hit in result] == ["global"]


async def test_missing_scope_disables_persistent_retrieval():
    embeddings = FakeEmbeddingClient()
    memory = VectorMemory("memory", embeddings, vector_size=2)
    memory._qclient = FakeQdrantClient([point("a", "scope-a", "секрет")])

    result = await memory.retrieve("секрет", scope=None, use_rerank=False)

    assert result == []
    assert embeddings.input_types == []


async def test_scope_hydration_builds_bm25_and_ignores_zero_match_documents():
    memory = VectorMemory("memory", FakeEmbeddingClient(), vector_size=2)
    memory._qclient = FakeQdrantClient(
        [
            point("rare", "scope-a", "старое редкое слово"),
            point("other", "scope-a", "полностью другой документ"),
        ]
    )

    result = await memory.retrieve(
        "редкое",
        scope="scope-a",
        use_rerank=False,
    )

    assert result[0].doc_id == "rare"
    assert memory._bm25["scope-a"].search("несуществующийтермин", 10) == []


async def test_embedding_failure_still_returns_hydrated_bm25_result():
    embeddings = FailingQueryEmbeddingClient()
    memory = VectorMemory("memory", embeddings, vector_size=2)
    memory._qclient = FakeQdrantClient(
        [point("rare", "scope-a", "старое редкое слово")]
    )

    result = await memory.retrieve(
        "редкое",
        scope="scope-a",
        use_rerank=False,
    )

    assert [hit.doc_id for hit in result] == ["rare"]
    assert embeddings.input_types == ["query"]


async def test_upsert_is_deterministic_and_stores_policy_metadata():
    embeddings = FakeEmbeddingClient()
    qdrant = FakeQdrantClient()
    memory = VectorMemory("memory", embeddings, vector_size=2)
    memory._qclient = qdrant

    kwargs = dict(
        scope="scope-a",
        memory_type="user_preference",
        canonical_key="user.preference.hr.dash_style",
        source="user_explicit",
        query="Что использовать?",
        project_id="project-a",
        conversation_id="conversation-a",
        confidence=0.97,
        importance=0.8,
    )
    first = await memory.upsert("Не использовать длинные тире", **kwargs)
    second = await memory.upsert("Не использовать длинные тире", **kwargs)

    assert first == second
    assert len(qdrant.points) == 1
    payload = qdrant.points[0].payload
    assert payload["scope"] == "scope-a"
    assert payload["memory_type"] == "user_preference"
    assert payload["canonical_key"] == "user.preference.hr.dash_style"
    assert payload["project_id"] == "project-a"
    assert payload["confidence"] == 0.97
    assert payload["status"] == "active"


async def test_new_value_with_same_canonical_key_supersedes_old_value():
    embeddings = FakeEmbeddingClient()
    qdrant = FakeQdrantClient()
    memory = VectorMemory("memory", embeddings, vector_size=2)
    memory._qclient = qdrant

    common = dict(
        scope="scope-a",
        memory_type="project_decision",
        canonical_key="project.database.choice",
        source="project_decision",
        project_id="project-a",
        confidence=0.95,
        importance=0.9,
    )
    old_id = await memory.upsert("Использовать Supabase", **common)
    new_id = await memory.upsert("Использовать Neon", **common)

    assert old_id != new_id
    payloads = {str(item.id): item.payload for item in qdrant.points}
    assert payloads[old_id]["status"] == "superseded"
    assert payloads[old_id]["superseded_by"] == new_id
    assert payloads[new_id]["status"] == "active"
    result = await memory.retrieve(
        "какую базу использовать",
        scope="scope-a",
        project_id="project-a",
        use_rerank=False,
        top_k=10,
    )
    assert [hit.doc_id for hit in result] == [new_id]


async def test_temporary_context_is_limited_to_its_conversation():
    memory = VectorMemory("memory", FakeEmbeddingClient(), vector_size=2)
    memory._qclient = FakeQdrantClient(
        [
            point(
                "temporary",
                "scope-a",
                "временный код 123",
                memory_type="temporary_context",
                conversation_id="conversation-a",
            )
        ]
    )
    other = await memory.retrieve(
        "временный код",
        scope="scope-a",
        conversation_id="conversation-b",
        use_rerank=False,
    )
    same = await memory.retrieve(
        "временный код",
        scope="scope-a",
        conversation_id="conversation-a",
        use_rerank=False,
    )
    assert other == []
    assert [hit.doc_id for hit in same] == ["temporary"]


async def test_expired_and_superseded_memories_are_not_returned():
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    memory = VectorMemory("memory", FakeEmbeddingClient(), vector_size=2)
    memory._qclient = FakeQdrantClient(
        [
            point("expired", "scope-a", "старое", expires_at=yesterday),
            point("superseded", "scope-a", "заменённое", status="superseded"),
            point("active", "scope-a", "активное"),
        ]
    )
    result = await memory.retrieve(
        "старое заменённое активное",
        scope="scope-a",
        use_rerank=False,
        top_k=10,
    )
    assert [hit.doc_id for hit in result] == ["active"]


async def test_policy_ranking_prefers_fresh_important_explicit_memory():
    now = datetime.now(timezone.utc)
    memory = VectorMemory("memory", FakeEmbeddingClient(), vector_size=2)
    memory._qclient = FakeQdrantClient(
        [
            point(
                "weak-old",
                "scope-a",
                "выбор базы",
                memory_type="assistant_conclusion",
                confidence=0.5,
                importance=0.2,
                created_at=(now - timedelta(days=180)).isoformat(),
            ),
            point(
                "strong-new",
                "scope-a",
                "выбор базы",
                memory_type="project_decision",
                confidence=0.99,
                importance=0.95,
                created_at=now.isoformat(),
            ),
        ]
    )
    result = await memory.retrieve(
        "выбор базы",
        scope="scope-a",
        use_rerank=False,
        top_k=2,
    )
    assert [hit.doc_id for hit in result] == ["strong-new", "weak-old"]
