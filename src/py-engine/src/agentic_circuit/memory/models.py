"""Typed contracts used by the memory gate and retrieval policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MemoryType = Literal[
    "user_fact",
    "user_preference",
    "negative_preference",
    "project_decision",
    "project_state",
    "temporary_context",
    "relationship_context",
    "assistant_conclusion",
]
MemorySource = Literal[
    "user_explicit",
    "user_correction",
    "project_decision",
    "assistant_verified",
]


@dataclass(frozen=True)
class MemoryContext:
    """Opaque namespaces derived from request metadata."""

    scope: str = ""
    workspace_id: str = ""
    project_id: str = ""
    conversation_id: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.scope)


class MemoryCandidate(BaseModel):
    should_store: bool = True
    sensitive: bool = False
    memory_type: MemoryType
    canonical_key: str = Field(min_length=3, max_length=160)
    content: str = Field(min_length=1, max_length=6000)
    source: MemorySource = "user_explicit"
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_days: int | None = Field(default=None, ge=1, le=3650)

    @field_validator("canonical_key")
    @classmethod
    def normalize_key(cls, value: str) -> str:
        normalized = ".".join(
            part.strip().lower().replace(" ", "_")
            for part in value.strip().split(".")
            if part.strip()
        )
        if not normalized:
            raise ValueError("canonical_key must not be empty")
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-:")
        if any(character not in allowed for character in normalized):
            raise ValueError("canonical_key must use lowercase latin identifiers")
        return normalized

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        return " ".join(value.split())


class MemoryGateResult(BaseModel):
    memories: list[MemoryCandidate] = Field(default_factory=list, max_length=8)


class MemorySelection(BaseModel):
    selected_ids: list[str] = Field(default_factory=list, max_length=12)
    outdated_ids: list[str] = Field(default_factory=list, max_length=12)
