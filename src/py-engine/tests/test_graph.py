"""Integration test: full LangGraph circuit on mocked providers.

Verifies fast/slow branches, per-circuit isolation (a circuit's phase-2 never
sees another circuit's phase-1), and aggregation into synthesis.
"""

import pytest

from agentic_circuit.config import CircuitConfig
from agentic_circuit.graph import EngineContext, build_graph
from agentic_circuit.providers import LLMResult
from agentic_circuit.rag import NullMemory


ROUTE = {"value": "fast"}


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
            "creative"
            if "креативная" in system
            else "pragmatic"
            if "прагматичная" in system
            else "effective"
            if "эффективная" in system
            else "unknown"
        )
        if "критик" in system or "критика самой себя" in system:
            return LLMResult(content=f"P2::{role}", model=model_cfg.model)
        return LLMResult(content=f"P1::{role}", model=model_cfg.model)


class FakeRegistry:
    def __init__(self, client):
        self._client = client

    def get(self, name):
        return self._client


class FakeComposite:
    async def ensure_collection(self): ...
    async def retrieve(self, query, top_k=5, use_rerank=True):
        return []
    async def upsert(self, text, doc_id=None):
        return ""


@pytest.fixture
def ctx():
    cfg = CircuitConfig.from_disk()
    client = RecordingClient()
    circuits = sorted({a.circuit for a in cfg.agents.values() if a.circuit})
    memories = {c: NullMemory() for c in circuits}
    context = EngineContext(
        config=cfg,
        clients=FakeRegistry(client),
        embeddings=None,
        rerank=None,
        web=None,
        memories=memories,
        synthesis_memory=FakeComposite(),
    )
    context._client = client
    return context


@pytest.fixture(autouse=True)
def reset():
    ROUTE["value"] = "fast"
    yield


async def test_fast_path_goes_straight_to_synthesis(ctx):
    ROUTE["value"] = "fast"
    graph = build_graph(ctx)
    result = await graph.ainvoke({"user_input": "Привет"})
    assert result["route"] == "fast"
    assert result["synthesis_output"] == "SYNTH"
    assert result.get("circuit_phase1", {}) == {}


async def test_slow_path_runs_all_circuits_in_parallel(ctx):
    ROUTE["value"] = "slow"
    graph = build_graph(ctx)
    result = await graph.ainvoke({"user_input": "Спланируй отпуск"})
    assert result["route"] == "slow"
    assert set(result["circuit_phase1"].keys()) == {"creative", "pragmatic", "effective"}
    assert set(result["circuit_phase2"].keys()) == {"creative", "pragmatic", "effective"}
    assert "P1::creative" in result["circuit_phase1"]["creative"]
    assert result["synthesis_output"] == "SYNTH"


async def test_circuit_isolation_phase2_sees_only_own_phase1(ctx):
    ROUTE["value"] = "slow"
    graph = build_graph(ctx)
    await graph.ainvoke({"user_input": "Идея для подарка"})
    client = ctx._client
    # find creative phase-2 call
    phase2_creative = None
    for messages, _ in client.calls:
        sys_text = messages[0]["content"]
        if "креативная" in sys_text and ("критик" in sys_text or "критика самой себя" in sys_text):
            phase2_creative = messages[1]["content"]
    assert phase2_creative is not None
    # creative's phase-2 must contain its own phase-1 but NOT other circuits'
    assert "P1::creative" in phase2_creative
    assert "P1::pragmatic" not in phase2_creative
    assert "P1::effective" not in phase2_creative
