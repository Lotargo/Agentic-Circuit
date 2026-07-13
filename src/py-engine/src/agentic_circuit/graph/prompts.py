"""System/user prompt assembly for every node of the circuit."""

from __future__ import annotations

from ..config import AgentConfig, load_meta_instruction, resolve_prism_manifest


def assemble_system_prompt(agent: AgentConfig, prism: str | None = None) -> str:
    parts = [agent.base_prompt.strip()]
    if agent.is_synthesis:
        meta = load_meta_instruction(agent)
        if meta:
            parts.extend(["", "## Мета-инструкция", meta.strip()])
    elif not agent.is_router:
        manifest = resolve_prism_manifest(agent, prism)
        if manifest:
            parts.extend(
                [
                    "",
                    f"## Активная призма настроения: {prism or agent.default_prism}",
                    manifest.strip(),
                ]
            )
    return "\n".join(parts)


def _format_context(contexts: list[str]) -> str:
    if not contexts:
        return ""
    lines = [
        "## Релевантные воспоминания",
        "Используй их как скрытый контекст, не цитируя явно:",
    ]
    lines.extend(f"{index}. {context}" for index, context in enumerate(contexts, 1))
    return "\n".join(lines)


def _conversation_messages(conversation: list[dict]) -> list[dict]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in conversation
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]


def router_messages(agent: AgentConfig, user_input: str) -> list[dict]:
    system = (
        assemble_system_prompt(agent)
        + "\n\n## Формат решения\n"
        + "Ответь только одним словом: fast или slow.\n"
        + "slow: сложный, творческий, неоднозначный, эмоциональный запрос или запрос, требующий критики.\n"
        + "fast: простой, фактический или рутинный запрос."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]


def phase1_messages(
    agent: AgentConfig,
    conversation: list[dict],
    contexts: list[str],
    prism: str | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": assemble_system_prompt(agent, prism)}]
    messages.extend(_conversation_messages(conversation))
    context = _format_context(contexts)
    if context:
        messages.append({"role": "system", "content": context})
    return messages


def phase2_messages(
    agent: AgentConfig,
    conversation: list[dict],
    phase1_answer: str,
    contexts: list[str],
    prism: str | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": assemble_system_prompt(agent, prism)}]
    messages.extend(_conversation_messages(conversation))
    content = (
        "## Твой сырой ответ текущего хода\n"
        f"{phase1_answer}\n\n"
        "Покритикуй его и подготовь улучшенную версию."
    )
    context = _format_context(contexts)
    if context:
        content += f"\n\n{context}"
    messages.append({"role": "user", "content": content})
    return messages


def synthesis_messages(
    agent: AgentConfig,
    conversation: list[dict],
    circuit_phase1: dict[str, str],
    circuit_phase2: dict[str, str],
    contexts: list[str],
    web_results: list[str] | None = None,
    prism: str | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": assemble_system_prompt(agent, prism)}]
    messages.extend(_conversation_messages(conversation[:-1]))

    blocks = ["## Текущий запрос", conversation[-1]["content"]]
    blocks.append("\n## Твои собственные потоки мыслей")
    for circuit in sorted(set(circuit_phase1) | set(circuit_phase2)):
        blocks.append(f"\n### Контур: {circuit}")
        if circuit in circuit_phase1:
            blocks.append(f"Сырая мысль: {circuit_phase1[circuit]}")
        if circuit in circuit_phase2:
            blocks.append(f"Отточенная мысль: {circuit_phase2[circuit]}")
    if web_results:
        blocks.append("\n## Данные внешнего поиска")
        blocks.extend(f"{index}. {value}" for index, value in enumerate(web_results, 1))
    context = _format_context(contexts)
    if context:
        blocks.append(f"\n{context}")
    blocks.append(
        "\nСформируй единый цельный ответ от лица Лизы. Отклоняй ошибочные или "
        "противоречивые мысли, но воспринимай остальное как своё."
    )
    messages.append({"role": "user", "content": "\n".join(blocks)})
    return messages
