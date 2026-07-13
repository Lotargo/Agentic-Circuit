"""Tests: prompt message builders for every node."""

from agentic_circuit.config import CircuitConfig
from agentic_circuit.graph.prompts import (
    phase1_messages,
    phase2_messages,
    router_messages,
    synthesis_messages,
)

cfg = CircuitConfig.from_disk()


def test_router_messages_use_config_and_ask_for_decision():
    agent = cfg.agents["router"]
    msgs = router_messages(agent, "Привет, как дела?")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert agent.base_prompt.strip() in msgs[0]["content"]
    assert "fast" in msgs[0]["content"] and "slow" in msgs[0]["content"]


def test_phase1_messages_have_system_and_user():
    agent = cfg.agents["creative-1"]
    msgs = phase1_messages(agent, "Напиши стих", contexts=["прошлое"])
    assert msgs[0]["role"] == "system"
    assert "Напиши стих" in msgs[1]["content"]
    assert "прошлое" in msgs[1]["content"]


def test_phase2_sees_only_its_own_phase1():
    agent = cfg.agents["creative-2"]
    msgs = phase2_messages(agent, "Напиши стих", "сырой ответ", contexts=[])
    user = msgs[1]["content"]
    assert "сырой ответ" in user
    assert "Запрос пользователя" in user


def test_synthesis_messages_aggregate_all_circuits_and_treat_as_own():
    agent = cfg.agents["synthesis"]
    msgs = synthesis_messages(
        agent,
        "Какой план?",
        {"creative": "креатив", "pragmatic": "прагма", "effective": "эффект"},
        {"creative": "креатив2", "pragmatic": "прагма2", "effective": "эффект2"},
        contexts=["воспоминание"],
        web_results=["факт из сети"],
    )
    user = msgs[1]["content"]
    for circuit in ("creative", "pragmatic", "effective"):
        assert circuit in user
    assert "ТВОИ мысли" in user or "твои мысли" in user
    assert "факт из сети" in user
    assert "воспоминание" in user
