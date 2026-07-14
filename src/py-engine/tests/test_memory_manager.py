from agentic_circuit.config import CircuitConfig
from agentic_circuit.memory import MemoryManager
from agentic_circuit.providers import LLMResult
from agentic_circuit.rag import MemoryHit


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def acomplete(self, messages, model_cfg, tools=None):
        self.calls.append(messages)
        content = self.responses.pop(0)
        return LLMResult(content=content, model=model_cfg.model)


class Registry:
    def __init__(self, client):
        self.client = client

    def get(self, _name):
        return self.client


def hit(doc_id: str, text: str) -> MemoryHit:
    return MemoryHit(
        doc_id=doc_id,
        text=text,
        score=0.9,
        collection="memory",
        scope="user:test",
        memory_type="user_preference",
        canonical_key=f"user.preference.{doc_id}",
        source="user_explicit",
    )


async def test_extract_accepts_durable_memories_and_applies_default_ttl():
    client = QueueClient(
        [
            """```json
            {
              "memories": [
                {
                  "should_store": true,
                  "memory_type": "project_decision",
                  "canonical_key": "project.database.choice",
                  "content": "Для проекта выбран Neon",
                  "source": "project_decision",
                  "confidence": 0.96,
                  "importance": 0.9,
                  "ttl_days": null
                },
                {
                  "should_store": true,
                  "memory_type": "assistant_conclusion",
                  "canonical_key": "project.database.risk",
                  "content": "Возможен риск миграции",
                  "source": "assistant_verified",
                  "confidence": 0.8,
                  "importance": 0.5,
                  "ttl_days": null
                },
                {
                  "should_store": true,
                  "memory_type": "temporary_context",
                  "canonical_key": "conversation.random",
                  "content": "Случайная мелочь",
                  "source": "user_explicit",
                  "confidence": 0.2,
                  "importance": 0.1,
                  "ttl_days": 2
                }
              ]
            }
            ```"""
        ]
    )
    manager = MemoryManager(CircuitConfig.from_disk().memory, Registry(client))
    memories = await manager.extract(
        [{"role": "user", "content": "Для проекта выбираем Neon"}],
        "Neon подходит",
        project_id="project:test",
    )

    assert [item.canonical_key for item in memories] == [
        "project.database.choice",
        "project.database.risk",
    ]
    assert memories[0].ttl_days is None
    assert memories[1].ttl_days == 30
    assert "MEMORY_EXTRACT" in client.calls[0][0]["content"]


async def test_extract_returns_empty_on_invalid_or_nonconforming_json():
    client = QueueClient(["не json"])
    manager = MemoryManager(CircuitConfig.from_disk().memory, Registry(client))
    assert await manager.extract(
        [{"role": "user", "content": "Привет"}],
        "Привет",
    ) == []


async def test_selector_uses_selected_ids_and_excludes_outdated_ids():
    client = QueueClient(
        ['{"selected_ids":["a","b"],"outdated_ids":["b"]}']
    )
    manager = MemoryManager(CircuitConfig.from_disk().memory, Registry(client))
    selected = await manager.select(
        "что я предпочитаю",
        [hit("a", "короткие ответы"), hit("b", "длинные ответы")],
        project_id="project:test",
    )
    assert [item.doc_id for item in selected] == ["a"]
    assert "MEMORY_SELECT" in client.calls[0][0]["content"]


async def test_selector_falls_back_to_bounded_ranked_candidates():
    client = QueueClient(["сломанный ответ"])
    manager = MemoryManager(CircuitConfig.from_disk().memory, Registry(client))
    candidates = [hit(str(index), f"memory {index}") for index in range(10)]
    selected = await manager.select("query", candidates, top_k=3)
    assert [item.doc_id for item in selected] == ["0", "1", "2"]
