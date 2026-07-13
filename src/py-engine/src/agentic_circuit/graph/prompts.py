"""System/user prompt assembly for every node of the circuit."""

from __future__ import annotations

from ..config import AgentConfig, load_agent_manifests, load_meta_instruction


def assemble_system_prompt(agent: AgentConfig) -> str:
    """Build the system prompt.

    Circuits: base_prompt + per-agent manifests (mood prisms).
    Synthesis: base_prompt + meta_instruction (no manifests).
    """
    parts = [agent.base_prompt.strip()]
    if agent.is_synthesis:
        meta = load_meta_instruction(agent)
        if meta:
            parts.append("")
            parts.append("## Мета-инструкция")
            parts.append(meta.strip())
    else:
        manifests = load_agent_manifests(agent)
        if manifests:
            parts.append("")
            parts.append("## Призмы настроения (твои ипостаси)")
            parts.append("Ты ведёшь себя согласно этим призмам, раскрывая свою личность:")
            for manifest in manifests:
                parts.append("")
                parts.append(manifest.strip())
    return "\n".join(parts)


def _format_context(contexts: list[str]) -> str:
    if not contexts:
        return ""
    lines = [
        "",
        "## Релевантные воспоминания (твои прошлые мысли)",
        "Используй их как контекст, не цитируя явно:",
    ]
    for index, context in enumerate(contexts, 1):
        lines.append(f"{index}. {context}")
    return "\n".join(lines)


def router_messages(agent: AgentConfig, user_input: str) -> list[dict]:
    """Build the router prompt from its YAML config plus a strict output contract."""
    system = (
        assemble_system_prompt(agent)
        + "\n\n## Формат решения\n"
        + "- Ответь только одним словом: fast или slow.\n"
        + "- slow: сложный, творческий, неоднозначный, эмоциональный запрос или запрос, требующий критики.\n"
        + "- fast: простой, фактический или рутинный запрос."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]


def phase1_messages(agent: AgentConfig, user_input: str, contexts: list[str]) -> list[dict]:
    system = assemble_system_prompt(agent)
    user = user_input + _format_context(contexts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def phase2_messages(
    agent: AgentConfig,
    user_input: str,
    phase1_answer: str,
    contexts: list[str],
) -> list[dict]:
    system = assemble_system_prompt(agent)
    user = (
        f"Запрос пользователя:\n{user_input}\n\n"
        f"Твой сырой ответ (фаза 1), который нужно покритиковать и улучшить:\n"
        f"{phase1_answer}"
        + _format_context(contexts)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def synthesis_messages(
    agent: AgentConfig,
    user_input: str,
    circuit_phase1: dict[str, str],
    circuit_phase2: dict[str, str],
    contexts: list[str],
    web_results: list[str] | None = None,
) -> list[dict]:
    system = assemble_system_prompt(agent)
    blocks: list[str] = [f"Запрос пользователя:\n{user_input}\n"]
    blocks.append("## Твои собственные потоки мыслей (это ТВОИ мысли, не чужие)")
    for circuit in sorted(set(circuit_phase1) | set(circuit_phase2)):
        blocks.append(f"\n### Контур: {circuit}")
        if circuit in circuit_phase1:
            blocks.append(f"Сырая мысль: {circuit_phase1[circuit]}")
        if circuit in circuit_phase2:
            blocks.append(f"Отточенная мысль: {circuit_phase2[circuit]}")
    if web_results:
        blocks.append("\n## Данные из внешнего поиска (твои уточнения)")
        for index, result in enumerate(web_results, 1):
            blocks.append(f"{index}. {result}")
    blocks.append(_format_context(contexts))
    blocks.append(
        "\nСформируй единый цельный ответ от лица Лизы. Отклоняй ошибочные или "
        "противоречивые мысли, но воспринимай остальное как своё."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(blocks)},
    ]
