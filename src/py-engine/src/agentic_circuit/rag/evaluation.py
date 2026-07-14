"""Small deterministic evaluation harness for memory retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    query: str
    scope: str
    project_id: str = ""
    conversation_id: str = ""
    expected_ids: tuple[str, ...] = ()
    forbidden_ids: tuple[str, ...] = ()
    top_k: int = 5


@dataclass(frozen=True)
class RetrievalReport:
    cases: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    forbidden_case_rate: float

    def as_dict(self) -> dict:
        return {
            "cases": self.cases,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "forbidden_case_rate": self.forbidden_case_rate,
        }


async def evaluate_retriever(retriever, cases: list[RetrievalCase]) -> RetrievalReport:
    recall_values: list[float] = []
    precision_values: list[float] = []
    reciprocal_ranks: list[float] = []
    forbidden_cases = 0

    for case in cases:
        hits = await retriever.retrieve(
            case.query,
            scope=case.scope,
            project_id=case.project_id,
            conversation_id=case.conversation_id,
            top_k=case.top_k,
            use_rerank=False,
        )
        returned = [hit.doc_id for hit in hits]
        expected = set(case.expected_ids)
        forbidden = set(case.forbidden_ids)
        relevant_count = len(expected.intersection(returned))
        recall_values.append(relevant_count / len(expected) if expected else 1.0)
        precision_values.append(relevant_count / len(returned) if returned else (1.0 if not expected else 0.0))

        rank = next(
            (index for index, doc_id in enumerate(returned, 1) if doc_id in expected),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else (1.0 if not expected and not returned else 0.0))
        if forbidden.intersection(returned):
            forbidden_cases += 1

    count = len(cases)
    if not count:
        return RetrievalReport(0, 0.0, 0.0, 0.0, 0.0)
    return RetrievalReport(
        cases=count,
        recall_at_k=sum(recall_values) / count,
        precision_at_k=sum(precision_values) / count,
        mrr=sum(reciprocal_ranks) / count,
        forbidden_case_rate=forbidden_cases / count,
    )


def load_cases(path: str | Path) -> list[RetrievalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        RetrievalCase(
            name=item["name"],
            query=item["query"],
            scope=item["scope"],
            project_id=item.get("project_id", ""),
            conversation_id=item.get("conversation_id", ""),
            expected_ids=tuple(item.get("expected_ids", [])),
            forbidden_ids=tuple(item.get("forbidden_ids", [])),
            top_k=int(item.get("top_k", 5)),
        )
        for item in raw
    ]
