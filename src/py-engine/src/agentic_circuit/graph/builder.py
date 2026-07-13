"""LangGraph circuit: router -> parallel circuits (phase1/phase2) -> synthesis."""

from __future__ import annotations

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


def _merge(left, right):  # reducer for parallel circuit writes
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


class CircuitState(TypedDict, total=False):
    user_input: str
    route: str
    router_raw: str
    circuit: str
    circuit_phase1: Annotated[dict[str, str], _merge]
    circuit_phase2: Annotated[dict[str, str], _merge]
    synthesis_output: str
    errors: list[str]


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
        for m in self.memories:
            await m.ensure_collection()

    async def retrieve(self, query: str, top_k: int = 5, use_rerank: bool = True) -> list[str]:
        if not self.memories:
            return []
        per = max(1, top_k // len(self.memories))
        out: list[str] = []
        for m in self.memories:
            out.extend(await m.retrieve(query, top_k=per, use_rerank=use_rerank))
        return out[:top_k]

    async def upsert(self, text: str, doc_id: Optional[str] = None) -> str:
        return ""


def _parse_route(raw: str) -> str:
    text = (raw or "").lower()
    return "slow" if "slow" in text else "fast"


def build_graph(ctx: EngineContext):
    circuits = sorted({a.circuit for a in ctx.config.agents.values() if a.circuit})

    async def router_node(state: CircuitState) -> dict:
        agent = ctx.config.router
        msgs = router_messages(state["user_input"])
        res = await ctx.clients.get(agent.model.provider).acomplete(msgs, agent.model)
        route = _parse_route(res.content)
        update: dict = {"router_raw": res.content, "route": route}
        if res.error:
            update.setdefault("errors", []).append(f"router: {res.error}")
        return update

    async def circuit_node(state: CircuitState) -> dict:
        circuit = state["circuit"]
        p1 = ctx.config.get(f"{circuit}-1")
        p2 = ctx.config.get(f"{circuit}-2")
        mem = ctx.memories[circuit]
        errors: list[str] = []

        c1 = await mem.retrieve(state["user_input"]) if p1.tools.rag else []
        r1 = await ctx.clients.get(p1.model.provider).acomplete(
            phase1_messages(p1, state["user_input"], c1), p1.model
        )
        if p1.tools.rag and r1.content:
            await mem.upsert(r1.content)
        if r1.error:
            errors.append(f"{circuit}-1: {r1.error}")

        c2 = await mem.retrieve(state["user_input"]) if p2.tools.rag else []
        r2 = await ctx.clients.get(p2.model.provider).acomplete(
            phase2_messages(p2, state["user_input"], r1.content, c2), p2.model
        )
        if p2.tools.rag and r2.content:
            await mem.upsert(r2.content)
        if r2.error:
            errors.append(f"{circuit}-2: {r2.error}")

        update = {
            "circuit_phase1": {circuit: r1.content},
            "circuit_phase2": {circuit: r2.content},
        }
        if errors:
            update["errors"] = errors
        return update

    async def synthesis_node(state: CircuitState) -> dict:
        agent = ctx.config.synthesis
        mem = ctx.synthesis_memory
        contexts = await mem.retrieve(state["user_input"]) if (mem and agent.tools.rag) else []
        web: list[str] = []
        if agent.tools.web_search and ctx.web:
            try:
                web = await ctx.web.search(state["user_input"])
            except Exception as exc:  # non-fatal
                web = []
        res = await ctx.clients.get(agent.model.provider).acomplete(
            synthesis_messages(
                agent,
                state["user_input"],
                state.get("circuit_phase1", {}),
                state.get("circuit_phase2", {}),
                contexts,
                web,
            ),
            agent.model,
        )
        update: dict = {"synthesis_output": res.content}
        if res.error:
            update["errors"] = [f"synthesis: {res.error}"]
        return update

    def route_decision(state: CircuitState):
        if state.get("route") == "slow":
            return [Send("circuit", {"circuit": c, "user_input": state["user_input"]}) for c in circuits]
        return "fast"

    g = StateGraph(CircuitState)
    g.add_node("router", router_node)
    g.add_node("circuit", circuit_node)
    g.add_node("synthesis", synthesis_node)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_decision, {"fast": "synthesis"})
    g.add_edge("circuit", "synthesis")
    g.add_edge("synthesis", END)
    return g.compile()
