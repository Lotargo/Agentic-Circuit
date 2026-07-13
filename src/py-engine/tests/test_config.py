"""Tests: config parsing + manifest loading + system prompt assembly."""

from agentic_circuit.config import (
    CircuitConfig,
    load_agent_manifests,
    load_meta_instruction,
)
from agentic_circuit.graph.prompts import assemble_system_prompt


def test_load_all_agents_and_providers():
    cfg = CircuitConfig.from_disk()
    assert "opencode-zen" in cfg.providers.providers
    names = set(cfg.agents.keys())
    assert {
        "router",
        "creative-1",
        "creative-2",
        "pragmatic-1",
        "pragmatic-2",
        "effective-1",
        "effective-2",
        "synthesis",
    } == names


def test_router_and_synthesis_roles():
    cfg = CircuitConfig.from_disk()
    assert cfg.router.is_router
    assert cfg.synthesis.is_synthesis
    assert cfg.synthesis.meta_instruction is not None
    assert cfg.synthesis.manifests == []


def test_circuit_agents_have_manifests():
    cfg = CircuitConfig.from_disk()
    for agent in cfg.circuit_agents:
        assert agent.manifests, f"{agent.name} should list manifests"
        texts = load_agent_manifests(agent)
        assert len(texts) == len(agent.manifests)
        # manifests must be individual per agent — file paths include agent name
        for m in agent.manifests:
            assert f"{agent.name}/{m}"  # path is namespaced per agent


def test_assemble_system_prompt_circuit_includes_manifests():
    cfg = CircuitConfig.from_disk()
    prompt = assemble_system_prompt(cfg.agents["creative-1"])
    assert "Ты — Лиза" in prompt
    assert "joy" in prompt.lower() or "радость" in prompt.lower()


def test_assemble_system_prompt_synthesis_uses_meta_not_manifests():
    cfg = CircuitConfig.from_disk()
    prompt = assemble_system_prompt(cfg.synthesis)
    assert "синтез" in prompt.lower() or "собираешь" in prompt.lower()
    meta = load_meta_instruction(cfg.synthesis)
    assert meta and meta[:20] in prompt
