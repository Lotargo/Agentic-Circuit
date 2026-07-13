"""Tests for config parsing, manifest selection, and system prompts."""

from agentic_circuit.config import (
    CircuitConfig,
    load_meta_instruction,
    resolve_prism_manifest,
)
from agentic_circuit.graph.prompts import assemble_system_prompt


def test_load_all_agents_and_providers():
    cfg = CircuitConfig.from_disk()
    assert "opencode-zen" in cfg.providers.providers
    assert {
        "router",
        "creative-1",
        "creative-2",
        "pragmatic-1",
        "pragmatic-2",
        "effective-1",
        "effective-2",
        "synthesis",
    } == set(cfg.agents)


def test_router_and_synthesis_roles():
    cfg = CircuitConfig.from_disk()
    assert cfg.router.is_router
    assert cfg.synthesis.is_synthesis
    assert cfg.synthesis.meta_instruction is not None
    assert cfg.synthesis.manifests == []


def test_circuit_agents_resolve_exactly_one_manifest():
    cfg = CircuitConfig.from_disk()
    for agent in cfg.circuit_agents:
        assert agent.manifests
        selected = resolve_prism_manifest(agent, "joy")
        assert selected
        fallback = resolve_prism_manifest(agent, "not-a-prism")
        assert fallback


def test_assemble_system_prompt_circuit_includes_only_selected_prism():
    cfg = CircuitConfig.from_disk()
    prompt = assemble_system_prompt(cfg.agents["creative-1"], prism="joy")
    assert "Ты — Лиза" in prompt
    assert "Активная призма настроения: joy" in prompt
    assert "Активная призма настроения: anger" not in prompt


def test_assemble_system_prompt_synthesis_uses_meta_not_manifests():
    cfg = CircuitConfig.from_disk()
    prompt = assemble_system_prompt(cfg.synthesis)
    meta = load_meta_instruction(cfg.synthesis)
    assert meta and meta[:20] in prompt
