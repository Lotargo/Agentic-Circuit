from agentic_circuit.graph import CompositeMemory
from agentic_circuit.rag import MemoryHit


def hit(doc_id, text, source, score):
    return MemoryHit(
        doc_id=doc_id,
        text=text,
        score=score,
        collection=source,
        scope="user:test",
        kind="assistant_conclusion",
        memory_type="assistant_conclusion",
        source=source,
        query="query",
    )


class FakeMemory:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def retrieve(
        self,
        query,
        *,
        scope,
        project_id="",
        conversation_id="",
        top_k,
        use_rerank,
    ):
        self.calls.append(
            {
                "scope": scope,
                "project_id": project_id,
                "conversation_id": conversation_id,
            }
        )
        assert use_rerank is False
        return self.hits[:top_k]


class FakeReranker:
    async def rerank_indices(self, query, documents, top_n):
        # Prefer the last collection candidate to prove ranking is global rather
        # than simple collection-order concatenation.
        return [(len(documents) - 1, 0.99), (0, 0.5)][:top_n]


async def test_composite_memory_reranks_globally_and_passes_namespaces():
    creative = FakeMemory([hit("c1", "creative answer", "creative", 0.8)])
    pragmatic = FakeMemory([hit("p1", "pragmatic answer", "pragmatic", 0.7)])
    effective = FakeMemory([hit("e1", "effective answer", "effective", 0.6)])
    composite = CompositeMemory(
        [creative, pragmatic, effective],
        rerank_client=FakeReranker(),
    )

    result = await composite.retrieve(
        "query",
        scope="user:test",
        project_id="project:test",
        conversation_id="conversation:test",
        top_k=2,
    )

    assert result[0].text == "effective answer"
    assert {result[0].source, result[1].source} == {"effective", "creative"}
    expected = {
        "scope": "user:test",
        "project_id": "project:test",
        "conversation_id": "conversation:test",
    }
    assert creative.calls == pragmatic.calls == effective.calls == [expected]


async def test_composite_memory_deduplicates_same_canonical_memory():
    duplicate_a = hit("a", "same", "creative", 0.8)
    duplicate_a = duplicate_a.__class__(
        **{**duplicate_a.__dict__, "canonical_key": "shared.key"}
    )
    duplicate_b = hit("b", "same", "pragmatic", 0.7)
    duplicate_b = duplicate_b.__class__(
        **{**duplicate_b.__dict__, "canonical_key": "shared.key"}
    )
    composite = CompositeMemory(
        [FakeMemory([duplicate_a]), FakeMemory([duplicate_b])],
        rerank_client=None,
    )

    result = await composite.retrieve(
        "query",
        scope="user:test",
        top_k=5,
        use_rerank=False,
    )

    assert [item.text for item in result] == ["same"]
