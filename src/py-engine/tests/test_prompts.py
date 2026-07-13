"""Tests for prompt message builders."""

from agentic_circuit.config import CircuitConfig
from agentic_circuit.graph.prompts import (
    phase1_messages,
    phase2_messages,
    router_messages,
    synthesis_messages,
)

cfg = CircuitConfig.from_disk()
CONVERSATION = [
    {"role": "user", "content": "Меня зовут Олег"},
    {"role": "assistant", "content": "Запомнила"},
    {"role": "user", "content": "Как меня зовут?"},
]


def test_router_messages_use_config_and_ask_for_decision():
    agent = cfg.agents["router"]
    messages = router_messages(agent, "Привет")
    assert agent.base_prompt.strip() in messages[0]["content"]
    assert "fast" in messages[0]["content"] and "slow" in messages[0]["content"]


def test_phase1_preserves_history_and_uses_one_prism():
    agent = cfg.agents["creative-1"]
    messages = phase1_messages(agent, CONVERSATION, ["память"], prism="joy")
    contents = "\n".join(message["content"] for message in messages)
    assert "Меня зовут Олег" in contents
    assert "Как меня зовут?" in contents
    assert "память" in contents
    assert "Активная призма настроения: joy" in messages[0]["content"]
    assert "resentment" not in messages[0]["content"].lower()


def test_phase2_sees_history_and_own_phase1():
    agent = cfg.agents["creative-2"]
    messages = phase2_messages(agent, CONVERSATION, "сырой ответ", [], prism="neutral")
    contents = "\n".join(message["content"] for message in messages)
    assert "Меня зовут Олег" in contents
    assert "сырой ответ" in contents


def test_synthesis_aggregates_all_circuits_and_history():
    agent = cfg.agents["synthesis"]
    messages = synthesis_messages(
        agent,
        CONVERSATION,
        {"creative": "креатив", "pragmatic": "прагма", "effective": "эффект"},
        {"creative": "креатив2", "pragmatic": "прагма2", "effective": "эффект2"},
        contexts=["воспоминание"],
        web_results=["факт из сети"],
        prism="neutral",
    )
    contents = "\n".join(message["content"] for message in messages)
    assert "Меня зовут Олег" in contents
    for circuit in ("creative", "pragmatic", "effective"):
        assert circuit in contents
    assert "факт из сети" in contents
    assert "воспоминание" in contents
