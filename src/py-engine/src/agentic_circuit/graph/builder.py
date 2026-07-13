"""LangGraph circuit: router -> parallel circuits (phase1/phase2) -> synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..config import CircuitConfig
from ..providers import ClientRegistry
from ..rag import EmbeddingClient, RerankClient, VectorMemory
from ..tools import WebSearchTool
from .prompts import (
    phase1_messages,
    phase2_messages,
    router_messages,
    synthesis_messages,
)


def _merge(left, right):
    """Merge dictionaries written by parallel circuit branches."""
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


def _merge_errors(left, right):
    """Collect errors from parallel branches instead of causing a state conflict."""
    return list(left or []) + list(right or [])


class CircuitState(TypedDict, total=False):
    user_input: str
    route: str
    router_raw: str
    circuit: str
    circuit_phase1: Annotated[dict[str, str], _merge]
    circuit_phase2: Annotated[dict[str, str], _merge]
    synthesis_output: str
    errors: Annotated[list[str], _merge_errors]


@dataclass
class EngineContext:
    config: CircuitConfig
    clients: ClientRegistry
    embeddings: EmbeddingClient
    rerank: Optional[RerankClient] = None
    web: Optional[WebSearchTool] = None
    memories: dict[str, VectorMemory] = field(default_factory=dict)
    synthesis_memory: Optional[object] = None


class CompositeMemory:
    """Synthesis retriever over every circuit collection."""

    def __init__(self, memories: list[VectorMemory]):
        self.memories = memories

    async def ensure_collection(self) -> None:
        for memory in self.memories:
            await memory.ensure_collection()

    async def retrieve(self, query: str, top_k: int = 5, use_rerank: bool = True) -> list[str]:
        if not self.memories:
            return []
        per_collection = max(1, top_k // len(self.memories))
        results: list[str] = []
        for memory in self.memories:
            results.extend(
                await memory.retrieve(
                    query,
                    top_k=per_collection,
                    use_rerank=use_rerank,
                )
            )
        return results[:top_k]

    async def upsert(self, text: str, doc_id: Optional[str] = None) -> str:
        return ""


def _parse_route(raw: str) -> str:
    """Accept only an unambiguous router decision; default safely to fast."""
    tokens = re.findall(r"[a-z]+", (raw or "").lower())
    if tokens == ["slow"]:
        return "slow"
    return "fast"


def build_graph(ctx: EngineContext):
    circuits = sorted({agent.circuit for agent in ctx.config.agents.values() if agent.circuit})

    async def router_node(state: CircuitState) -> dict:
        agent = ctx.config.router
        messages = router_messages(agent, state["user_input"])
        result = await ctx.clients.get(agent.model.provider).acomplete(messages, agent.model)
        update: dict = {
            "router_raw": result.content,
            "route": _parse_route(result.content),
        }
        if result.error:
            update["errors"] = [f"router: {result.error}"]
        return update

    async def circuit_node(state: CircuitState) -> dict:
        circuit = state["circuit"]
        phase1 = ctx.config.get(f"{circuit}-1")
        phase2 = ctx.config.get(f"{circuit}-2")
        memory = ctx.memories[circuit]
        errors: list[str] = []

        contexts1 = await memory.retrieve(state["user_input"]) if phase1.tools.rag else []
        result1 = await ctx.clients.get(phase1.model.provider).acomplete(
            phase1_messages(phase1, state["user_input"], contexts1),
            phase1.model,
        )
        if phase1.tools.rag and result1.content:
            await memory.upsert(result1.content)
        if result1.error:
            errors.append(f"{circuit}-1: {result1.error}")

        contexts2 = await memory.retrieve(state["user_input"]) if phase2.tools.rag else []
        result2 = await ctx.clients.get(phase2.model.provider).acomplete(
            phase2_messages(phase2, state["user_input"], result1.content, contexts2),
            phase2.model,
        )
        if phase2.tools.rag and result2.content:
            await memory.upsert(result2.content)
        if result2.error:
            errors.append(f"{circuit}-2: {result2.error}")

        update: dict = {
            "circuit_phase1": {circuit: result1.content},
            "circuit_phase2": {circuit: result2.content},
        }
        if errors:
            update["errors"] = errors
        return update

    async def synthesis_node(state: CircuitState) -> dict:
        agent = ctx.config.synthesis
        memory = ctx.synthesis_memory
        contexts = (
            await memory.retrieve(state["user_input"])
            if memory and agent.tools.rag
            else []
        )
        web_results: list[str] = []
        errors: list[str] = []
        if agent.tools.web_search and ctx.web:
            try:
                web_results = await ctx.web.search(state["user_input"])
            except Exception as exc:
                errors.append(f"web_search: {type(exc).__name__}: {exc}")

        result = await ctx.clients.get(agent.model.provider).acomplete(
            synthesis_messages(
                agent,
                state["user_input"],
                state.get("circuit_phase1", {}),
                state.get("circuit_phase2", {}),
                contexts,
                web_results,
            ),
            agent.model,
        )
        update: dict = {"synthesis_output": result.content}
        if result.error:
            errors.append(f"synthesis: {result.error}")
        if errors:
            update["errors"] = errors
        return update

    def route_decision(state: CircuitState):
        if state.get("route") == "slow":
            return [
                Send("circuit", {"circuit": circuit, "user_input": state["user_input"]})
                for circuit in circuits
            ]
        return "fast"

    graph = StateGraph(CircuitState)
    graph.add_node("router", router_node)
    graph.add_node("circuit", circuit_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_decision, {"fast": "synthesis"})
    graph.add_edge("circuit", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()
