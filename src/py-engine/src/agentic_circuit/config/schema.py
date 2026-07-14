"""Pydantic schema for the agentic-circuit configuration."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

ThinkingLevel = Literal["off", "low", "medium", "high"]
PrismName = Literal[
    "joy",
    "flirt",
    "resentment",
    "arousal",
    "anger",
    "apathy",
    "neutral",
    "sadness",
]


class AgentRole(str, Enum):
    router = "router"
    synthesis = "synthesis"
    memory = "memory"
    circuit_phase1 = "circuit_phase1"
    circuit_phase2 = "circuit_phase2"


class Provider(BaseModel):
    type: str = "openai-compatible"
    base_url: str
    api_key_env: str
    models: list[str] = Field(default_factory=list)


class ProvidersFile(BaseModel):
    providers: dict[str, Provider]


class ModelConfig(BaseModel):
    provider: str
    model: str
    fallback_models: list[str] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    thinking_level: ThinkingLevel = "off"

    @field_validator("fallback_models")
    @classmethod
    def unique_fallback_models(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("fallback model names must be unique")
        return values

    @model_validator(mode="after")
    def validate_model_chain(self) -> "ModelConfig":
        if self.model in self.fallback_models:
            raise ValueError("primary model must not be repeated in fallback_models")
        return self

    @property
    def model_chain(self) -> list[str]:
        return [self.model, *self.fallback_models]


class ToolsConfig(BaseModel):
    web_search: bool = False
    rag: bool = True


class AgentConfig(BaseModel):
    name: str
    role: AgentRole = AgentRole.circuit_phase1
    base_prompt: str
    manifests: list[str] = Field(default_factory=list)
    default_prism: PrismName = "neutral"
    meta_instruction: Optional[str] = None
    model: ModelConfig
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    collection: Optional[str] = None

    @field_validator("manifests")
    @classmethod
    def unique_manifests(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("manifest names must be unique")
        return values

    @property
    def is_synthesis(self) -> bool:
        return self.role == AgentRole.synthesis

    @property
    def is_router(self) -> bool:
        return self.role == AgentRole.router

    @property
    def is_memory(self) -> bool:
        return self.role == AgentRole.memory

    @property
    def circuit(self) -> Optional[str]:
        """Circuit name derived from agent name (e.g. creative-1 -> creative)."""
        if self.role in (AgentRole.router, AgentRole.synthesis, AgentRole.memory):
            return None
        return self.name.rsplit("-", 1)[0]
