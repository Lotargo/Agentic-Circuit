from types import SimpleNamespace

from agentic_circuit.rag.store import VectorMemory


class FakeEmbeddingClient:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeQdrantClient:
    async def search(self, collection, query_vector, limit, with_payload):
        assert collection == "creative"
        assert with_payload is True
        return [
            SimpleNamespace(
                id="persisted-id",
                score=0.99,
                payload={"text": "мысль из прошлого запуска"},
            )
        ]


async def test_retrieve_hydrates_text_from_qdrant_payload_after_restart():
    memory = VectorMemory(
        collection="creative",
        embedding_client=FakeEmbeddingClient(),
        vector_size=2,
    )
    memory._qclient = FakeQdrantClient()

    result = await memory.retrieve("похожий запрос", use_rerank=False)

    assert result == ["мысль из прошлого запуска"]
