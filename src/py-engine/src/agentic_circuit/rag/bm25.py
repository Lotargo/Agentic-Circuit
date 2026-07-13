"""Small in-process BM25 index used beside Qdrant dense retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: dict[str, list[str]] = {}
        self.docs: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.df: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avg_len = 0.0

    def add(self, doc_id: str, text: str) -> None:
        """Insert or replace a document without duplicating its statistics."""
        self._documents[doc_id] = _tokenize(text)
        self._rebuild()

    def clear(self) -> None:
        self._documents.clear()
        self._rebuild()

    def _rebuild(self) -> None:
        self.doc_ids = list(self._documents)
        self.docs = [self._documents[doc_id] for doc_id in self.doc_ids]
        self.doc_len = [len(tokens) for tokens in self.docs]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

        frequencies: dict[str, int] = defaultdict(int)
        for tokens in self.docs:
            for term in set(tokens):
                frequencies[term] += 1
        self.df = frequencies

        count = len(self.docs)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in self.df.items()
        }

    def search(self, query: str, top_n: int = 5) -> list[tuple[str, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self.docs:
            return []

        scores: list[float] = []
        for tokens, document_length in zip(self.docs, self.doc_len):
            term_frequencies = Counter(tokens)
            score = 0.0
            for term in query_tokens:
                frequency = term_frequencies.get(term, 0)
                if not frequency or term not in self.idf:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b
                    + self.b * document_length / max(self.avg_len, 1e-6)
                )
                score += self.idf[term] * (
                    frequency * (self.k1 + 1)
                ) / denominator
            scores.append(score)

        ranked = sorted(
            zip(self.doc_ids, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_n]
