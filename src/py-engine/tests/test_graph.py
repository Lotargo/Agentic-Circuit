"""Integration tests for the LangGraph circuit on mocked providers."""

import pytest

from agentic_circuit.config import CircuitConfig
from agentic_circuit.graph import EngineContext, build_graph
from agentic_circuit.providers import LLMResult

ROUTE = {"value": "fast"}
CONVERSATION = [
    {"role": "user", "content": "Меня зовут Олег"},
    {"role": "assistant", "content": "Запомнила"},
    {"role": "user", "content": "Как меня зовут?"},
]


class RecordingClient:
    def __init__(self):
        self.calls: list[tuple[list[dict], object]] = []

    async def acomplete(self, messages, model_cfg, tools=None):
        self.calls.append((messages, model_cfg))
        system = messages[0]["content"]
        if "функцию выбора глубины" in system:
            return LLMResult(content=ROUTE["value"], model=model_cfg.model)
        if "единственный внешний ответ" in system:
            return LLMResult(content="SYNTH", model=model_cfg.model)
        role = (
            "creative"
            if "креативную перспективу" in system
            else "pragmatic"
            if "прагматичную перспективу" in system
            else "effective"
            if "эффективную перспективу" in system
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

    async def retrieve(self, query, *, scope, top_k=5, use_rerank=True):
        self.retrievals.append({"query": query, "scope": scope})
        return []

    async def upsert(self, text, *, scope, **metadata):
        self.upserts.append({"text": text, "scope": scope, **metadata})
        return "saved"

    async def aclose(self): ...


class FakeComposite:
    def __init__(self):
        self.retrievals = []

    async def ensure_collection(self): ...

    async def retrieve(self, query, *, scope, top_k=5, use_rerank=True):
        self.retrievals.append({"query": query, "scope": scope})
        return []


@pytest.fixture
def ctx():
    cfg = CircuitConfig.from_disk()
    client = RecordingClient()
    memories = {circuit: RecordingMemory() for circuit in cfg.circuit_collections}
    conversation_memory = RecordingMemory()
    synthesis_memory = FakeComposite()
    context = EngineContext(
        config=cfg,
        clients=FakeRegistry(client),
        embeddings=None,
        rerank=None,
        web=None,
        memories=memories,
        conversation_memory=conversation_memory,
        synthesis_memory=synthesis_memory,
    )
    context._client = client
    context._conversation_memory = conversation_memory
    context._synthesis_memory = synthesis_memory
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
    }


async def test_fast_path_goes_straight_to_synthesis_and_saves_final_answer(ctx):
    result = await build_graph(ctx).ainvoke(initial_state())
    assert result["route"] == "fast"
    assert result["synthesis_output"] == "SYNTH"
    assert result.get("circuit_phase1", {}) == {}
    assert ctx._conversation_memory.upserts == [
        {
            "text": "SYNTH",
            "scope": "user:test",
            "kind": "assistant_answer",
            "source": "synthesis",
            "query": "Как меня зовут?",
            "prism": "joy",
        }
    ]


async def test_slow_path_runs_all_circuits_and_saves_only_refined_outputs(ctx):
    ROUTE["value"] = "slow"
    result = await build_graph(ctx).ainvoke(initial_state())
    assert set(result["circuit_phase1"]) == {"creative", "pragmatic", "effective"}
    assert set(result["circuit_phase2"]) == {"creative", "pragmatic", "effective"}
    for circuit, memory in ctx.memories.items():
        assert len(memory.upserts) == 1
        saved = memory.upserts[0]
        assert saved["text"] == f"P2::{circuit}"
        assert saved["kind"] == "refined_perspective"
        assert saved["scope"] == "user:test"


async def test_history_prism_and_scope_reach_circuit(ctx):
    ROUTE["value"] = "slow"
    await build_graph(ctx).ainvoke(initial_state())
    circuit_calls = [
        messages
        for messages, _ in ctx._client.calls
        if "креативную перспективу" in messages[0]["content"]
    ]
    assert circuit_calls
    combined = "\n".join(message["content"] for message in circuit_calls[0])
    assert "Меня зовут Олег" in combined
    assert "Активная эмоциональная призма: joy" in circuit_calls[0][0]["content"]
    assert ctx.memories["creative"].retrievals[0]["scope"] == "user:test"


async def test_circuit_isolation_phase2_sees_only_own_phase1(ctx):
    ROUTE["value"] = "slow"
    await build_graph(ctx).ainvoke(initial_state())
    phase2_creative = None
    for messages, _ in ctx._client.calls:
        content = "\n".join(message["content"] for message in messages)
        if "креативную перспективу" in messages[0]["content"] and "P1::creative" in content:
            phase2_creative = content
    assert phase2_creative is not None
    assert "P1::pragmatic" not in phase2_creative
    assert "P1::effective" not in phase2_creative
