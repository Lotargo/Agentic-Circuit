from types import SimpleNamespace

from agentic_circuit.rag.store import VectorMemory


class FakeEmbeddingClient:
    def __init__(self):
        self.input_types: list[str] = []

    async def embed(self, texts, *, input_type="passage"):
        self.input_types.append(input_type)
        return [[1.0, 0.0] for _ in texts]


class FakeQdrantClient:
    async def search(
        self,
        *,
        collection_name,
        query_vector,
        limit,
        with_payload,
    ):
        assert collection_name == "creative"
        assert with_payload is True
        return [
            SimpleNamespace(
                id="persisted-id",
                score=0.99,
                payload={"text": "мысль из прошлого запуска"},
            )
        ]


async def test_retrieve_hydrates_text_from_qdrant_payload_after_restart():
    embeddings = FakeEmbeddingClient()
    memory = VectorMemory(
        collection="creative",
        embedding_client=embeddings,
        vector_size=2,
    )
    memory._qclient = FakeQdrantClient()

    result = await memory.retrieve("похожий запрос", use_rerank=False)

    assert result == ["мысль из прошлого запуска"]
    assert embeddings.input_types == ["query"]


class HydrationQdrantClient:
    async def collection_exists(self, _collection):
        return True

    async def scroll(
        self,
        *,
        collection_name,
        limit,
        offset,
        with_payload,
        with_vectors,
    ):
        assert collection_name == "creative"
        assert with_payload is True
        assert with_vectors is False
        return (
            [
                SimpleNamespace(
                    id="old-id",
                    payload={"text": "старое редкое слово"},
                )
            ],
            None,
        )


async def test_ensure_collection_rebuilds_bm25_from_qdrant_payloads():
    memory = VectorMemory(
        collection="creative",
        embedding_client=FakeEmbeddingClient(),
        vector_size=2,
    )
    memory._qclient = HydrationQdrantClient()

    await memory.ensure_collection()

    assert memory._bm25.search("редкое", 1) == [("old-id", memory._bm25.search("редкое", 1)[0][1])]
    assert memory._texts["old-id"] == "старое редкое слово"
