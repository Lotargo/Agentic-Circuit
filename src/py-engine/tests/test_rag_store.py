from types import SimpleNamespace

from agentic_circuit.rag.store import VectorMemory


class FakeEmbeddingClient:
    def __init__(self):
        self.input_types: list[str] = []

    async def embed(self, texts, *, input_type="passage"):
        self.input_types.append(input_type)
        return [[1.0, 0.0] for _ in texts]


class FakeQdrantClient:
    def __init__(self, points=None):
        self.points = list(points or [])
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
        assert collection_name == "creative"
        assert with_payload is True
        assert with_vectors is False
        scope = self._scope(scroll_filter)
        self.last_scope = scope
        return (
            [point for point in self.points if point.payload.get("scope") == scope],
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
        assert collection_name == "creative"
        assert with_payload is True
        scope = self._scope(query_filter)
        self.last_scope = scope
        points = [
            SimpleNamespace(
                id=point.id,
                score=0.99,
                payload=point.payload,
            )
            for point in self.points
            if point.payload.get("scope") == scope
        ]
        return SimpleNamespace(points=points[:limit])

    async def upsert(self, *, collection_name, points):
        assert collection_name == "creative"
        for point in points:
            self.points = [existing for existing in self.points if existing.id != point.id]
            self.points.append(point)


def point(doc_id: str, scope: str | None, text: str):
    payload = {
        "text": text,
        "kind": "refined_perspective",
        "source": "creative",
        "query": "старый запрос",
        "prism": "neutral",
        "created_at": "2026-07-14T00:00:00+00:00",
    }
    if scope is not None:
        payload["scope"] = scope
    return SimpleNamespace(id=doc_id, payload=payload)


async def test_retrieve_is_filtered_by_user_scope_and_keeps_provenance():
    embeddings = FakeEmbeddingClient()
    qdrant = FakeQdrantClient(
        [
            point("user-a", "scope-a", "память пользователя A"),
            point("user-b", "scope-b", "память пользователя B"),
            point("legacy", None, "старая глобальная память"),
        ]
    )
    memory = VectorMemory("creative", embeddings, vector_size=2)
    memory._qclient = qdrant

    result = await memory.retrieve(
        "похожий запрос",
        scope="scope-a",
        use_rerank=False,
    )

    assert [hit.text for hit in result] == ["память пользователя A"]
    assert result[0].source == "creative"
    assert result[0].kind == "refined_perspective"
    assert qdrant.last_scope == "scope-a"
    assert embeddings.input_types == ["query"]


async def test_missing_scope_disables_persistent_retrieval():
    embeddings = FakeEmbeddingClient()
    memory = VectorMemory("creative", embeddings, vector_size=2)
    memory._qclient = FakeQdrantClient([point("a", "scope-a", "секрет")])

    result = await memory.retrieve("секрет", scope=None, use_rerank=False)

    assert result == []
    assert embeddings.input_types == []


async def test_scope_hydration_builds_bm25_and_ignores_zero_match_documents():
    memory = VectorMemory("creative", FakeEmbeddingClient(), vector_size=2)
    qdrant = FakeQdrantClient(
        [
            point("rare", "scope-a", "старое редкое слово"),
            point("other", "scope-a", "полностью другой документ"),
        ]
    )
    memory._qclient = qdrant

    result = await memory.retrieve(
        "редкое",
        scope="scope-a",
        use_rerank=False,
    )

    assert result[0].doc_id == "rare"
    lexical = memory._bm25["scope-a"].search("несуществующийтермин", 10)
    assert lexical == []


async def test_upsert_is_deterministic_and_scoped():
    embeddings = FakeEmbeddingClient()
    qdrant = FakeQdrantClient()
    memory = VectorMemory("creative", embeddings, vector_size=2)
    memory._qclient = qdrant

    first = await memory.upsert(
        "проверенная мысль",
        scope="scope-a",
        kind="refined_perspective",
        source="creative",
        query="вопрос",
        prism="joy",
    )
    second = await memory.upsert(
        "проверенная мысль",
        scope="scope-a",
        kind="refined_perspective",
        source="creative",
        query="вопрос",
        prism="joy",
    )

    assert first == second
    assert len(qdrant.points) == 1
    assert qdrant.points[0].payload["scope"] == "scope-a"
    assert qdrant.points[0].payload["kind"] == "refined_perspective"
    assert embeddings.input_types == ["passage", "passage"]
