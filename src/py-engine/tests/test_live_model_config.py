from agentic_circuit.config import CircuitConfig


def test_every_agent_uses_big_pickle_with_mimo_fallback():
    config = CircuitConfig.from_disk()

    for agent in config.agents.values():
        assert agent.model.model == "big-pickle", agent.name
        assert agent.model.fallback_models == ["mimo-v2.5-free"], agent.name
        assert agent.model.model_chain == ["big-pickle", "mimo-v2.5-free"]
