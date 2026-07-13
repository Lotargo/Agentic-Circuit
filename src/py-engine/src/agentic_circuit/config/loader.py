"""Configuration loading with env substitution and manifest resolution."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .schema import AgentConfig, AgentRole, Provider, ProvidersFile

def _find_config_dir() -> Path:
    """Locate the ``config`` directory containing providers.yaml, walking upward."""
    here = Path(__file__).resolve()
    candidates = [here.parents[0]]  # config/ next to this module's dir
    for p in here.parents:
        candidates.append(p / "config")
    env = os.environ.get("CONFIG_DIR")
    if env:
        candidates.insert(0, Path(env))
    for c in candidates:
        if (c / "providers.yaml").exists():
            return c
    # fall back to conventional layout
    return here.parents[3] / "config"


CONFIG_DIR = _find_config_dir()
AGENTS_DIR = CONFIG_DIR / "agents"
MANIFESTS_DIR = CONFIG_DIR / "manifests"


def _substitute_env(value: str) -> str:
    """Replace ${VAR} / $VAR tokens from environment, leaving unknowns intact."""
    if not isinstance(value, str) or "${" not in value and "$" not in value:
        return value
    import re

    def repl(match: "re.Match[str]") -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, match.group(0))

    return re.sub(r"\$\{([^}]+)\}|\$(\w+)", repl, value)


def _deep_substitute(obj):
    if isinstance(obj, dict):
        return {k: _deep_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_substitute(v) for v in obj]
    if isinstance(obj, str):
        return _substitute_env(obj)
    return obj


def load_providers(path: Optional[Path] = None) -> ProvidersFile:
    path = path or (CONFIG_DIR / "providers.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        raw = _deep_substitute(yaml.safe_load(fh))
    return ProvidersFile.model_validate(raw)


def load_agent(path: Path) -> AgentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = _deep_substitute(yaml.safe_load(fh))
    return AgentConfig.model_validate(raw)


def load_all_agents(agents_dir: Optional[Path] = None) -> dict[str, AgentConfig]:
    agents_dir = agents_dir or AGENTS_DIR
    agents: dict[str, AgentConfig] = {}
    for p in sorted(agents_dir.glob("*.yaml")):
        agent = load_agent(p)
        agents[agent.name] = agent
    return agents


def load_manifest(agent_name: str, manifest_file: str) -> str:
    """Load a single manifest markdown file for an agent."""
    path = MANIFESTS_DIR / agent_name / manifest_file
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return path.read_text(encoding="utf-8")


def load_agent_manifests(agent: AgentConfig) -> list[str]:
    """Load and concatenate all manifest texts for an agent."""
    return [load_manifest(agent.name, m) for m in agent.manifests]


def load_meta_instruction(agent: AgentConfig) -> Optional[str]:
    """Synthesis has a meta_instruction: either inline text or a file path."""
    if not agent.meta_instruction:
        return None
    candidate = MANIFESTS_DIR / agent.meta_instruction
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return agent.meta_instruction


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
        return next(a for a in self.agents.values() if a.is_router)

    @property
    def synthesis(self) -> AgentConfig:
        return next(a for a in self.agents.values() if a.is_synthesis)

    @property
    def circuit_agents(self) -> list[AgentConfig]:
        return [a for a in self.agents.values() if a.role != AgentRole.router and a.role != AgentRole.synthesis]


@lru_cache(maxsize=1)
def get_config() -> CircuitConfig:
    return CircuitConfig.from_disk()
