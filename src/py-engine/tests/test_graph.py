"""Integration tests for the LangGraph circuit on mocked providers."""

import pytest

from agentic_circuit.config import CircuitConfig
from agentic_circuit.graph import EngineContext, build_graph
from agentic_circuit.memory import MemoryCandidate
from agentic_circuit.providers import LLMResult
from agentic_circuit.rag import MemoryHit

ROUTE = {"value": "fast"}
CONVERSATION = [
    {"role": "user", "content": "Меня зовут Олег"},
    {"role": "assistant", "content": "Запомнила"},
    {"role": "user", "content": "Как меня зовут?"},
]


def current_function(system_prompt: str) -> str:
    section = system_prompt.split("## Текущая функция мышления", 1)[-1]
    section = section.split("## Правила синтеза", 1)[0]
    section = section.split("## Активная эмоциональная призма", 1)[0]
    return section


class RecordingClient:
    def __init__(self):
        self.calls: list[tuple[list[dict], object]] = []

    async def acomplete(self, messages, model_cfg, tools=None):
        self.calls.append((messages, model_cfg))
        function = current_function(messages[0]["content"])
        if "функцию выбора глубины" in function:
            return LLMResult(content=ROUTE["value"], model=model_cfg.model)
        if "единственный внешний ответ" in function:
            return LLMResult(content="SYNTH", model=model_cfg.model)
        role = (
            "creative"
            if "креативн" in function
            else "pragmatic"
            if "прагматичн" in function
            else "effective"
            if "эффективн" in function
            else "unknown"
        )
        all_content = "\n".join(message["content"] for message in messages)
        if "Черновая мысль текущего хода" in all_content:
            return LLMResult(content=f"P2::{role}", model=model_cfg.model)
        return LLMResult(content=f"P1::{role}", model=model_cfg.model)


class FakeRegistry:
    def __init__(self, client):
        self._client = client

    def get(self, _name):
        return self._client

    async def aclose(self): ...


class RecordingMemory:
    def __init__(self):
        self.retrievals = []
        self.upserts = []

    async def ensure_collection(self): ...

    async def retrieve(self, query, *, scope, project_id="", conversation_id="", top_k=5, use_rerank=True):
        self.retrievals.append(
            {
                "query": query,
                "scope": scope,
                "project_id": project_id,
                "conversation_id": conversation_id,
            }
        )
        return []

    async def upsert(self, text, *, scope, **metadata):
        self.upserts.append({"text": text, "scope": scope, **metadata})
        return "saved"

    async def aclose(self): ...


class FakeComposite(RecordingMemory):
    def __init__(self):
        super().__init__()
        self.hit = MemoryHit(
            doc_id="preference-1",
            text="Олег предпочитает короткие сообщения",
            score=0.9,
            collection="memory",
            scope="user:test",
            memory_type="user_preference",
            canonical_key="user.preference.message_length",
            source="user_explicit",
            project_id="project:test",
        )

    async def retrieve(self, query, *, scope, project_id="", conversation_id="", top_k=5, use_rerank=True):
        await super().retrieve(
            query,
            scope=scope,
            project_id=project_id,
            conversation_id=conversation_id,
            top_k=top_k,
            use_rerank=use_rerank,
        )
        return [self.hit]


class FakeMemoryManager:
    def __init__(self):
        self.selections = []
        self.extractions = []

    async def select(self, query, candidates, *, project_id="", top_k=6):
        self.selections.append((query, candidates, project_id, top_k))
        return candidates[:1]

    async def extract(self, conversation, answer, *, project_id="", existing=None):
        self.extractions.append((conversation, answer, project_id, existing))
        return [
            MemoryCandidate(
                memory_type="user_fact",
                canonical_key="user.identity.name",
                content="Пользователя зовут Олег",
                source="user_explicit",
                confidence=0.99,
                importance=0.8,
            )
        ]


@pytest.fixture
def ctx():
    cfg = CircuitConfig.from_disk()
    client = RecordingClient()
    legacy = {circuit: RecordingMemory() for circuit in cfg.circuit_collections}
    structured = RecordingMemory()
    synthesis_memory = FakeComposite()
    memory_manager = FakeMemoryManager()
    context = EngineContext(
        config=cfg,
        clients=FakeRegistry(client),
        embeddings=None,
        rerank=None,
        web=None,
        memories=legacy,
        structured_memory=structured,
        synthesis_memory=synthesis_memory,
        memory_manager=memory_manager,
    )
    context._client = client
    context._structured_memory = structured
    context._synthesis_memory = synthesis_memory
    context._memory_manager = memory_manager
    return context


@pytest.fixture(autouse=True)
def reset():
    ROUTE["value"] = "fast"
    yield


def initial_state():
    return {
        "user_input": CONVERSATION[-1]["content"],
        "conversation": CONVERSATION,
        "prism": "joy",
        "memory_scope": "user:test",
        "workspace_id": "workspace:test",
        "project_id": "project:test",
        "conversation_id": "conversation:test",
    }


async def test_fast_path_selects_recall_and_stores_only_gate_output(ctx):
    result = await build_graph(ctx).ainvoke(initial_state())
    assert result["route"] == "fast"
    assert result["synthesis_output"] == "SYNTH"
    assert result.get("circuit_phase1", {}) == {}
    assert len(ctx._memory_manager.selections) == 1
    assert ctx._structured_memory.upserts == [
        {
            "text": "Пользователя зовут Олег",
            "scope": "user:test",
            "kind": "user_fact",
            "memory_type": "user_fact",
            "canonical_key": "user.identity.name",
            "source": "user_explicit",
            "query": "Как меня зовут?",
            "prism": "joy",
            "project_id": "project:test",
            "conversation_id": "conversation:test",
            "confidence": 0.99,
            "importance": 0.8,
            "ttl_days": None,
        }
    ]


async def test_slow_path_runs_all_circuits_without_persisting_internal_thoughts(ctx):
    ROUTE["value"] = "slow"
    result = await build_graph(ctx).ainvoke(initial_state())
    assert set(result["circuit_phase1"]) == {"creative", "pragmatic", "effective"}
    assert set(result["circuit_phase2"]) == {"creative", "pragmatic", "effective"}
    assert all(not memory.upserts for memory in ctx.memories.values())
    assert len(ctx._structured_memory.upserts) == 1


async def test_selected_memory_reaches_every_circuit_with_project_scope(ctx):
    ROUTE["value"] = "slow"
    await build_graph(ctx).ainvoke(initial_state())
    circuit_calls = [
        messages
        for messages, _ in ctx._client.calls
        if "креативн" in current_function(messages[0]["content"])
    ]
    assert circuit_calls
    combined = "\n".join(message["content"] for message in circuit_calls[0])
    assert "Олег предпочитает короткие сообщения" in combined
    recall = ctx._synthesis_memory.retrievals[0]
    assert recall["scope"] == "user:test"
    assert recall["project_id"] == "project:test"
    assert recall["conversation_id"] == "conversation:test"


async def test_circuit_isolation_phase2_sees_only_own_phase1(ctx):
    ROUTE["value"] = "slow"
    await build_graph(ctx).ainvoke(initial_state())
    phase2_creative = None
    for messages, _ in ctx._client.calls:
        content = "\n".join(message["content"] for message in messages)
        function = current_function(messages[0]["content"])
        if "креативн" in function and "P1::creative" in content:
            phase2_creative = content
    assert phase2_creative is not None
    assert "P1::pragmatic" not in phase2_creative
    assert "P1::effective" not in phase2_creative
