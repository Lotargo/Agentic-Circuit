"""Configuration loading with env substitution and manifest resolution."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .schema import AgentConfig, AgentRole, PrismName, Provider, ProvidersFile


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
        agents[agent.name] = agent
    return agents


def load_manifest(agent_name: str, manifest_file: str) -> str:
    path = MANIFESTS_DIR / agent_name / manifest_file
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return path.read_text(encoding="utf-8")


def resolve_prism_manifest(agent: AgentConfig, prism: PrismName | str | None) -> str | None:
    """Return exactly one allowed manifest for the requested/default prism."""
    if agent.is_router or agent.is_synthesis or not agent.manifests:
        return None
    selected = str(prism or agent.default_prism)
    filename = f"{selected}.md"
    if filename not in agent.manifests:
        filename = f"{agent.default_prism}.md"
    if filename not in agent.manifests:
        return None
    return load_manifest(agent.name, filename)


def load_agent_manifests(agent: AgentConfig) -> list[str]:
    """Compatibility helper; callers should prefer ``resolve_prism_manifest``."""
    manifest = resolve_prism_manifest(agent, agent.default_prism)
    return [manifest] if manifest else []


def load_meta_instruction(agent: AgentConfig) -> Optional[str]:
    if not agent.meta_instruction:
        return None
    candidate = MANIFESTS_DIR / agent.meta_instruction
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return agent.meta_instruction


def config_fingerprint(config_dir: Optional[Path] = None) -> str:
    """Hash all YAML/Markdown configuration files for cheap hot reload checks."""
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


@lru_cache(maxsize=1)
def get_config() -> CircuitConfig:
    return CircuitConfig.from_disk()


def clear_config_cache() -> None:
    get_config.cache_clear()
