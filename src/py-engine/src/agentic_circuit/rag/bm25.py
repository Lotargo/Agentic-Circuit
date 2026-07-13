"""Lightweight in-process BM25 lexical index (no external dependency).

Used as the lexical component of hybrid retrieval. The Qdrant store keeps an
in-memory mirror of each collection's texts so BM25 can score them locally.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.df: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avg_len = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = _tokenize(text)
        self.docs.append(tokens)
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(tokens))
        for term in set(tokens):
            self.df[term] += 1
        self._recompute_idf()
        self.avg_len = sum(self.doc_len) / max(len(self.doc_len), 1)

    def _recompute_idf(self) -> None:
        n = len(self.docs)
        for term, freq in self.df.items():
            self.idf[term] = math.log(1 + (n - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_n: int = 5) -> list[tuple[str, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self.docs:
            return []
        scores: list[float] = []
        for tokens, dlen in zip(self.docs, self.doc_len):
            score = 0.0
            term_freqs = defaultdict(int)
            for t in tokens:
                term_freqs[t] += 1
            for term, tf in term_freqs.items():
                if term not in self.idf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dlen / max(self.avg_len, 1e-6))
                score += self.idf[term] * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
