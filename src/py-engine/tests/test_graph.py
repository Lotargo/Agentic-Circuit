"""Integration tests for the full LangGraph circuit on mocked providers."""

import pytest

from agentic_circuit.config import CircuitConfig
from agentic_circuit.graph import EngineContext, build_graph
from agentic_circuit.providers import LLMResult
from agentic_circuit.rag import NullMemory

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
        if "маршрутизатор" in system:
            return LLMResult(content=ROUTE["value"], model=model_cfg.model)
        if "собираешь" in system:
            return LLMResult(content="SYNTH", model=model_cfg.model)
        role = (
            "creative" if "креативная" in system else
            "pragmatic" if "прагматичная" in system else
            "effective" if "эффективная" in system else "unknown"
        )
        all_content = "\n".join(message["content"] for message in messages)
        if "сырой ответ текущего хода" in all_content.lower():
            return LLMResult(content=f"P2::{role}", model=model_cfg.model)
        return LLMResult(content=f"P1::{role}", model=model_cfg.model)


class FakeRegistry:
    def __init__(self, client):
        self._client = client

    def get(self, _name):
        return self._client

    async def aclose(self): ...


class FakeComposite:
    async def ensure_collection(self): ...
    async def retrieve(self, query, top_k=5, use_rerank=True): return []
    async def upsert(self, text, doc_id=None): return ""


@pytest.fixture
def ctx():
    cfg = CircuitConfig.from_disk()
    client = RecordingClient()
    circuits = sorted({agent.circuit for agent in cfg.agents.values() if agent.circuit})
    context = EngineContext(
        config=cfg,
        clients=FakeRegistry(client),
        embeddings=None,
        rerank=None,
        web=None,
        memories={circuit: NullMemory() for circuit in circuits},
        synthesis_memory=FakeComposite(),
    )
    context._client = client
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
    }


async def test_fast_path_goes_straight_to_synthesis(ctx):
    result = await build_graph(ctx).ainvoke(initial_state())
    assert result["route"] == "fast"
    assert result["synthesis_output"] == "SYNTH"
    assert result.get("circuit_phase1", {}) == {}


async def test_slow_path_runs_all_circuits_in_parallel(ctx):
    ROUTE["value"] = "slow"
    result = await build_graph(ctx).ainvoke(initial_state())
    assert set(result["circuit_phase1"]) == {"creative", "pragmatic", "effective"}
    assert set(result["circuit_phase2"]) == {"creative", "pragmatic", "effective"}
    assert result["synthesis_output"] == "SYNTH"


async def test_history_and_prism_reach_circuit_prompts(ctx):
    ROUTE["value"] = "slow"
    await build_graph(ctx).ainvoke(initial_state())
    circuit_calls = [messages for messages, _ in ctx._client.calls if "креативная" in messages[0]["content"]]
    assert circuit_calls
    combined = "\n".join(message["content"] for message in circuit_calls[0])
    assert "Меня зовут Олег" in combined
    assert "Активная призма настроения: joy" in circuit_calls[0][0]["content"]


async def test_circuit_isolation_phase2_sees_only_own_phase1(ctx):
    ROUTE["value"] = "slow"
    await build_graph(ctx).ainvoke(initial_state())
    phase2_creative = None
    for messages, _ in ctx._client.calls:
        content = "\n".join(message["content"] for message in messages)
        if "креативная" in messages[0]["content"] and "P1::creative" in content:
            phase2_creative = content
    assert phase2_creative is not None
    assert "P1::pragmatic" not in phase2_creative
    assert "P1::effective" not in phase2_creative
