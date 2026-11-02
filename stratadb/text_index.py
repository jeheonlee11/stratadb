"""BM25 keyword index over one metadata text field.

Embeddings blur exact tokens: an error code, a function name or a part number
is often what a query hinges on. The keyword index catches those, and
``fuse_rrf`` merges its ranking with the vector ranking.

The index is derived state: it is rebuilt from the records when a collection
opens, so it needs no persistence of its own.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

_TOKEN = re.compile(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = {}
        self._lengths: dict[str, int] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._lengths)

    def add(self, rid: str, text: str) -> None:
        self.remove(rid)
        tokens = tokenize(text or "")
        self._lengths[rid] = len(tokens)
        self._total_len += len(tokens)
        for term, tf in Counter(tokens).items():
            self._postings.setdefault(term, {})[rid] = tf

    def remove(self, rid: str) -> None:
        length = self._lengths.pop(rid, None)
        if length is None:
            return
        self._total_len -= length
        empty = []
        for term, docs in self._postings.items():
            if docs.pop(rid, None) is not None and not docs:
                empty.append(term)
        for term in empty:
            del self._postings[term]

    def search(self, query: str, k: int, allow: Callable[[str], bool] | None = None) -> list[tuple[float, str]]:
        n = len(self._lengths)
        if n == 0:
            return []
        avg_len = self._total_len / n or 1.0
        scores: dict[str, float] = {}
        for term in set(tokenize(query)):
            docs = self._postings.get(term)
            if not docs:
                continue
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for rid, tf in docs.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self._lengths[rid] / avg_len)
                scores[rid] = scores.get(rid, 0.0) + idf * tf * (self.k1 + 1) / norm
        ranked = sorted(((s, rid) for rid, s in scores.items() if allow is None or allow(rid)), reverse=True)
        return ranked[:k]


def fuse_rrf(rankings: list[list[str]], k: int, c: int = 60) -> list[tuple[float, str]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (c + rank(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (c + rank)
    return sorted(((s, rid) for rid, s in scores.items()), reverse=True)[:k]
