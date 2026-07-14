from pathlib import Path

from agentic_circuit.rag.evaluation import evaluate_retriever, load_cases
from agentic_circuit.rag.store import MemoryHit, VectorMemory


class LexicalOnlyEmbeddings:
    async def embed(self, texts, *, input_type="passage"):
        if input_type == "query":
            raise RuntimeError("dense retrieval disabled for deterministic eval")
        return [[1.0, 0.0] for _ in texts]


def memory_hit(
    doc_id: str,
    text: str,
    *,
    scope: str = "user:test",
    project_id: str = "",
    conversation_id: str = "",
    memory_type: str = "user_fact",
    status: str = "active",
) -> MemoryHit:
    return MemoryHit(
        doc_id=doc_id,
        text=text,
        score=0.0,
        collection="memory",
        scope=scope,
        source="user_explicit",
        memory_type=memory_type,
        kind=memory_type,
        canonical_key=f"eval.{doc_id}",
        project_id=project_id,
        conversation_id=conversation_id,
        confidence=0.95,
        importance=0.8,
        source_quality=1.0,
        status=status,
    )


async def test_deterministic_rag_eval_meets_quality_and_isolation_thresholds():
    memory = VectorMemory("memory", LexicalOnlyEmbeddings(), vector_size=2)
    records = [
        memory_hit("name", "Пользователя зовут Олег"),
        memory_hit("other-user", "Другого пользователя зовут Иван", scope="user:other"),
        memory_hit(
            "neon",
            "Для проекта выбрана база данных Neon",
            project_id="project:a",
            memory_type="project_decision",
        ),
        memory_hit(
            "supabase-old",
            "Для проекта выбрана база данных Supabase",
            project_id="project:a",
            memory_type="project_decision",
            status="superseded",
        ),
        memory_hit(
            "project-b-db",
            "Для другого проекта выбрана база данных MongoDB",
            project_id="project:b",
            memory_type="project_decision",
        ),
        memory_hit(
            "no-long-dashes",
            "В текстах для HR не использовать длинные тире",
            memory_type="negative_preference",
        ),
        memory_hit(
            "temporary-code",
            "Временный код 4821",
            conversation_id="conversation:a",
            memory_type="temporary_context",
        ),
    ]
    for record in records:
        memory._remember(record)
    memory._hydrated_scopes.update({"user:test", "user:other"})

    cases = load_cases(Path(__file__).parent / "fixtures" / "rag_eval.json")
    report = await evaluate_retriever(memory, cases)

    assert report.recall_at_k == 1.0
    assert report.precision_at_k == 1.0
    assert report.mrr == 1.0
    assert report.forbidden_case_rate == 0.0
