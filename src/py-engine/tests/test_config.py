"""Tests for configuration, shared persona, and topology validation."""

from agentic_circuit.config import (
    CircuitConfig,
    load_meta_instruction,
    load_personality_core,
    resolve_prism_manifest,
)
from agentic_circuit.graph.prompts import assemble_system_prompt


def test_load_all_agents_and_providers():
    cfg = CircuitConfig.from_disk()
    assert "opencode-zen" in cfg.providers.providers
    assert {
        "router",
        "memory",
        "creative-1",
        "creative-2",
        "pragmatic-1",
        "pragmatic-2",
        "effective-1",
        "effective-2",
        "synthesis",
    } == set(cfg.agents)


def test_router_synthesis_memory_and_collection_topology():
    cfg = CircuitConfig.from_disk()
    assert cfg.router.is_router
    assert cfg.memory.is_memory
    assert cfg.memory.model.temperature == 0.0
    assert cfg.synthesis.is_synthesis
    assert cfg.synthesis.meta_instruction is not None
    assert cfg.synthesis.manifests
    assert cfg.circuit_collections == {
        "creative": "creative",
        "pragmatic": "pragmatic",
        "effective": "effective",
    }


def test_every_visible_agent_uses_same_personality_core():
    cfg = CircuitConfig.from_disk()
    core = load_personality_core().strip()
    for agent in cfg.agents.values():
        prompt = assemble_system_prompt(agent, prism="joy")
        assert core in prompt


def test_memory_manager_has_no_emotional_prism():
    cfg = CircuitConfig.from_disk()
    prompt = assemble_system_prompt(cfg.memory, prism="joy")
    assert "Активная эмоциональная призма" not in prompt
    assert resolve_prism_manifest(cfg.memory, "joy") is None


def test_shared_prism_is_identical_across_directions():
    cfg = CircuitConfig.from_disk()
    selected = {
        resolve_prism_manifest(agent, "joy")
        for agent in [
            cfg.agents["creative-1"],
            cfg.agents["pragmatic-2"],
            cfg.agents["effective-1"],
            cfg.synthesis,
        ]
    }
    assert len(selected) == 1


def test_assemble_system_prompt_includes_only_selected_prism():
    cfg = CircuitConfig.from_disk()
    prompt = assemble_system_prompt(cfg.agents["creative-1"], prism="joy")
    assert "Активная эмоциональная призма: joy" in prompt
    assert "Призма: радость" in prompt
    assert "Призма: злость" not in prompt


def test_synthesis_uses_meta_and_active_prism():
    cfg = CircuitConfig.from_disk()
    prompt = assemble_system_prompt(cfg.synthesis, prism="sadness")
    meta = load_meta_instruction(cfg.synthesis)
    assert meta and meta[:30] in prompt
    assert "Активная эмоциональная призма: sadness" in prompt
    assert "Призма: грусть" in prompt
