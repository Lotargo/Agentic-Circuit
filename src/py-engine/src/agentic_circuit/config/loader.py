"""Configuration loading with env substitution and shared persona resolution."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .schema import AgentConfig, AgentRole, PrismName, ProvidersFile


def _find_config_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parents[0]]
    for parent in here.parents:
        candidates.append(parent / "config")
    env = os.environ.get("CONFIG_DIR")
    if env:
        candidates.insert(0, Path(env))
    for candidate in candidates:
        if (candidate / "providers.yaml").exists():
            return candidate
    return here.parents[3] / "config"


CONFIG_DIR = _find_config_dir()
AGENTS_DIR = CONFIG_DIR / "agents"
MANIFESTS_DIR = CONFIG_DIR / "manifests"
PRISMS_DIR = MANIFESTS_DIR / "prisms"
PERSONALITY_CORE_PATH = MANIFESTS_DIR / "personality_core.md"


def _substitute_env(value: str) -> str:
    if not isinstance(value, str) or "${" not in value and "$" not in value:
        return value
    import re

    def repl(match: "re.Match[str]") -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, match.group(0))

    return re.sub(r"\$\{([^}]+)\}|\$(\w+)", repl, value)


def _deep_substitute(obj):
    if isinstance(obj, dict):
        return {key: _deep_substitute(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_deep_substitute(value) for value in obj]
    if isinstance(obj, str):
        return _substitute_env(obj)
    return obj


def load_providers(path: Optional[Path] = None) -> ProvidersFile:
    path = path or (CONFIG_DIR / "providers.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        raw = _deep_substitute(yaml.safe_load(handle))
    return ProvidersFile.model_validate(raw)


def load_agent(path: Path) -> AgentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = _deep_substitute(yaml.safe_load(handle))
    return AgentConfig.model_validate(raw)


def load_all_agents(agents_dir: Optional[Path] = None) -> dict[str, AgentConfig]:
    agents_dir = agents_dir or AGENTS_DIR
    agents: dict[str, AgentConfig] = {}
    for path in sorted(agents_dir.glob("*.yaml")):
        agent = load_agent(path)
        if agent.name in agents:
            raise ValueError(f"Duplicate agent name: {agent.name}")
        agents[agent.name] = agent
    return agents


def load_personality_core() -> str:
    if not PERSONALITY_CORE_PATH.exists():
        raise FileNotFoundError(f"Personality core not found: {PERSONALITY_CORE_PATH}")
    return PERSONALITY_CORE_PATH.read_text(encoding="utf-8")


def resolve_prism_manifest(agent: AgentConfig, prism: PrismName | str | None) -> str | None:
    """Return one shared emotional prism allowed by the current agent config."""
    if agent.is_router or not agent.manifests:
        return None
    selected = str(prism or agent.default_prism)
    filename = f"{selected}.md"
    if filename not in agent.manifests:
        filename = f"{agent.default_prism}.md"
    if filename not in agent.manifests:
        return None
    path = PRISMS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Shared prism manifest not found: {path}")
    return path.read_text(encoding="utf-8")


def load_meta_instruction(agent: AgentConfig) -> Optional[str]:
    if not agent.meta_instruction:
        return None
    candidate = MANIFESTS_DIR / agent.meta_instruction
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return agent.meta_instruction


def config_fingerprint(config_dir: Optional[Path] = None) -> str:
    root = config_dir or CONFIG_DIR
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".md"}
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class CircuitConfig(BaseModel):
    providers: ProvidersFile = Field(default_factory=ProvidersFile)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_topology(self) -> "CircuitConfig":
        routers = [agent for agent in self.agents.values() if agent.is_router]
        synthesis = [agent for agent in self.agents.values() if agent.is_synthesis]
        if len(routers) != 1:
            raise ValueError("Configuration must define exactly one router")
        if len(synthesis) != 1:
            raise ValueError("Configuration must define exactly one synthesis agent")

        by_circuit: dict[str, dict[AgentRole, AgentConfig]] = {}
        for agent in self.agents.values():
            if agent.model.provider not in self.providers.providers:
                raise ValueError(
                    f"Agent {agent.name} references unknown provider {agent.model.provider}"
                )
            provider = self.providers.providers[agent.model.provider]
            if provider.models and agent.model.model not in provider.models:
                raise ValueError(
                    f"Agent {agent.name} references model {agent.model.model} "
                    f"not declared by provider {agent.model.provider}"
                )
            if agent.role in (AgentRole.router, AgentRole.synthesis):
                continue
            if not agent.circuit:
                raise ValueError(f"Agent {agent.name} has no circuit")
            if agent.tools.rag and not agent.collection:
                raise ValueError(f"RAG-enabled agent {agent.name} requires collection")
            by_circuit.setdefault(agent.circuit, {})[agent.role] = agent

        for circuit, phases in by_circuit.items():
            if set(phases) != {AgentRole.circuit_phase1, AgentRole.circuit_phase2}:
                raise ValueError(f"Circuit {circuit} must contain phase-1 and phase-2")
            first = phases[AgentRole.circuit_phase1]
            second = phases[AgentRole.circuit_phase2]
            if first.collection != second.collection:
                raise ValueError(
                    f"Circuit {circuit} phases must use the same RAG collection"
                )
        return self

    @classmethod
    def from_disk(cls, config_dir: Optional[Path] = None) -> "CircuitConfig":
        config_dir = config_dir or CONFIG_DIR
        providers = load_providers(config_dir / "providers.yaml")
        agents = load_all_agents(config_dir / "agents")
        return cls(providers=providers, agents=agents)

    def get(self, name: str) -> AgentConfig:
        return self.agents[name]

    @property
    def router(self) -> AgentConfig:
        return next(agent for agent in self.agents.values() if agent.is_router)

    @property
    def synthesis(self) -> AgentConfig:
        return next(agent for agent in self.agents.values() if agent.is_synthesis)

    @property
    def circuit_agents(self) -> list[AgentConfig]:
        return [
            agent
            for agent in self.agents.values()
            if agent.role not in (AgentRole.router, AgentRole.synthesis)
        ]

    @property
    def circuit_collections(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for agent in self.circuit_agents:
            if agent.circuit and agent.collection:
                result[agent.circuit] = agent.collection
        return result


@lru_cache(maxsize=1)
def get_config() -> CircuitConfig:
    return CircuitConfig.from_disk()


def clear_config_cache() -> None:
    get_config.cache_clear()
