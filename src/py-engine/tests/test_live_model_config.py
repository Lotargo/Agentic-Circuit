from agentic_circuit.config import CircuitConfig


def test_every_agent_resolves_model_chain_from_environment(monkeypatch):
    monkeypatch.setenv("AGENTIC_PRIMARY_MODEL", "test-primary-model")
    monkeypatch.setenv("AGENTIC_FALLBACK_MODEL", "test-fallback-model")

    config = CircuitConfig.from_disk()

    assert config.providers.providers["opencode-zen"].models == []
    for agent in config.agents.values():
        assert agent.model.model == "test-primary-model", agent.name
        assert agent.model.fallback_models == ["test-fallback-model"], agent.name
        assert agent.model.model_chain == ["test-primary-model", "test-fallback-model"]
