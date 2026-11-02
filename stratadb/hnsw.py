"""Hierarchical Navigable Small World (HNSW) graph index.

A from-scratch implementation of Malkov & Yashunin (2016) on top of numpy.
Nodes are dense integer labels ``0..n-1`` assigned in insertion order; the
caller maps them to user ids. Distances are "smaller is closer":

- ``cosine``: ``1 - dot(a, b)`` on unit-normalized vectors
- ``ip``:     ``-dot(a, b)``
- ``l2``:     squared euclidean distance

Deletion is soft (``mark_deleted``): deleted nodes still route searches but
never appear in results. Segments rebuild their index on compaction.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable

import numpy as np

METRICS = ("cosine", "ip", "l2")


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 128,
        seed: int = 42,
        capacity: int = 1024,
    ):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.dim = dim
        self.metric = metric
        self.m = m
        self.m0 = 2 * m  # layer 0 is denser, as in the paper
        self.ef_construction = max(ef_construction, m)
        self.level_mult = 1.0 / math.log(m)
        self._rng = np.random.default_rng(seed)

        self._vectors = np.zeros((capacity, dim), dtype=np.float32)
        self._levels: list[int] = []
        # _links[node][level] -> list of neighbour labels
        self._links: list[list[list[int]]] = []
        self._deleted: set[int] = set()
        self.entry_point: int | None = None
        self.max_level = -1

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self._levels)

    @property
    def live_count(self) -> int:
        return len(self._levels) - len(self._deleted)

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors[: len(self)]

    def _prepare(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        if self.metric == "cosine":
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    def _dist_many(self, q: np.ndarray, labels) -> np.ndarray:
        vecs = self._vectors[labels]
        if self.metric == "l2":
            diff = vecs - q
            return np.einsum("ij,ij->i", diff, diff)
        dots = vecs @ q
        return 1.0 - dots if self.metric == "cosine" else -dots

    def _pairwise(self, labels) -> np.ndarray:
        vecs = self._vectors[labels]
        gram = vecs @ vecs.T
        if self.metric == "l2":
            sq = np.einsum("ij,ij->i", vecs, vecs)
            return sq[:, None] + sq[None, :] - 2 * gram
        return 1.0 - gram if self.metric == "cosine" else -gram

    def _dist(self, q: np.ndarray, label: int) -> float:
        v = self._vectors[label]
        if self.metric == "l2":
            d = v - q
            return float(d @ d)
        dot = float(v @ q)
        return 1.0 - dot if self.metric == "cosine" else -dot

    def _random_level(self) -> int:
        return int(-math.log(1.0 - self._rng.random()) * self.level_mult)

    def _grow(self) -> None:
        bigger = np.zeros((self._vectors.shape[0] * 2, self.dim), dtype=np.float32)
        bigger[: self._vectors.shape[0]] = self._vectors
        self._vectors = bigger

    # ----------------------------------------------------------------- search
    def _search_layer(self, q: np.ndarray, entries: list[tuple[float, int]], ef: int, level: int):
        """Greedy best-first search on one layer. Returns [(dist, label)] sorted ascending."""
        visited = {label for _, label in entries}
        candidates = list(entries)
        heapq.heapify(candidates)
        best = [(-d, label) for d, label in entries]  # max-heap of the ef closest
        heapq.heapify(best)
        while len(best) > ef:
            heapq.heappop(best)

        while candidates:
            d, label = heapq.heappop(candidates)
            if d > -best[0][0] and len(best) >= ef:
                break
            neighbours = [n for n in self._links[label][level] if n not in visited]
            if not neighbours:
                continue
            visited.update(neighbours)
            dists = self._dist_many(q, neighbours)
            worst = -best[0][0]
            for nd, n in zip(dists.tolist(), neighbours):
                if len(best) < ef or nd < worst:
                    heapq.heappush(candidates, (nd, n))
                    heapq.heappush(best, (-nd, n))
                    if len(best) > ef:
                        heapq.heappop(best)
                    worst = -best[0][0]
        return sorted((-nd, n) for nd, n in best)

    def _select_neighbours(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Heuristic selection (paper, alg. 4): keep a candidate only if it is closer
        to the base node than to every neighbour already kept. Spreads links out so
        the graph stays navigable across clusters."""
        if len(candidates) <= 1:
            return [c for _, c in candidates]
        labels = [c for _, c in candidates]
        base = [d for d, _ in candidates]
        # One matrix op for all candidate-to-candidate distances, then a running
        # "distance to nearest kept neighbour" vector: far cheaper than one numpy
        # call per (candidate, kept) pair.
        pair = self._pairwise(labels)
        nearest_kept = np.full(len(labels), np.inf, dtype=np.float32)
        selected: list[int] = []
        discarded: list[int] = []
        for i, c in enumerate(labels):
            if len(selected) >= m:
                break
            if base[i] < nearest_kept[i]:
                selected.append(c)
                np.minimum(nearest_kept, pair[i], out=nearest_kept)
            else:
                discarded.append(c)
        # Top up with the closest discarded ones so low-degree nodes aren't stranded.
        for c in discarded:
            if len(selected) >= m:
                break
            selected.append(c)
        return selected

    # ----------------------------------------------------------------- insert
    def add(self, vector) -> int:
        q = self._prepare(vector)
        label = len(self._levels)
        if label >= self._vectors.shape[0]:
            self._grow()
        self._vectors[label] = q
        level = self._random_level()
        self._levels.append(level)
        self._links.append([[] for _ in range(level + 1)])

        if self.entry_point is None:
            self.entry_point = label
            self.max_level = level
            return label

        ep = self.entry_point
        entries = [(self._dist(q, ep), ep)]
        for lc in range(self.max_level, level, -1):
            entries = self._search_layer(q, entries, 1, lc)[:1]

        for lc in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(q, entries, self.ef_construction, lc)
            m_max = self.m0 if lc == 0 else self.m
            neighbours = self._select_neighbours(found, self.m)
            self._links[label][lc] = neighbours
            for n in neighbours:
                links = self._links[n][lc]
                links.append(label)
                if len(links) > m_max:
                    dists = self._dist_many(self._vectors[n], links)
                    order = sorted(zip(dists.tolist(), links))
                    self._links[n][lc] = self._select_neighbours(order, m_max)
            entries = found

        if level > self.max_level:
            self.max_level = level
            self.entry_point = label
        return label

    def add_batch(self, vectors) -> list[int]:
        return [self.add(v) for v in np.asarray(vectors, dtype=np.float32)]

    def mark_deleted(self, label: int) -> None:
        if 0 <= label < len(self._levels):
            self._deleted.add(label)

    def is_deleted(self, label: int) -> bool:
        return label in self._deleted

    def search(
        self,
        vector,
        k: int = 10,
        ef: int | None = None,
        allow: Callable[[int], bool] | None = None,
    ) -> list[tuple[int, float]]:
        """Return up to k (label, distance) pairs, closest first.

        ``allow`` filters labels (e.g. a metadata predicate). Filtering happens
        during the search by widening ``ef`` until enough allowed hits are found.
        """
        if self.entry_point is None or k <= 0:
            return []
        q = self._prepare(vector)
        ef = max(ef or 64, k)

        def keep(label: int) -> bool:
            return label not in self._deleted and (allow is None or allow(label))

        while True:
            ep = self.entry_point
            entries = [(self._dist(q, ep), ep)]
            for lc in range(self.max_level, 0, -1):
                entries = self._search_layer(q, entries, 1, lc)[:1]
            found = self._search_layer(q, entries, ef, 0)
            hits = [(label, d) for d, label in found if keep(label)]
            if len(hits) >= k or ef >= len(self._levels):
                return hits[:k]
            ef = min(ef * 4, len(self._levels))

    def brute_force(self, vector, k: int = 10, allow: Callable[[int], bool] | None = None):
        """Exact search, used for small/filtered sets and as ground truth in tests."""
        n = len(self._levels)
        if n == 0:
            return []
        q = self._prepare(vector)
        labels = [i for i in range(n) if i not in self._deleted and (allow is None or allow(i))]
        if not labels:
            return []
        dists = self._dist_many(q, labels)
        top = np.argsort(dists)[:k]
        return [(labels[i], float(dists[i])) for i in top]

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        n = len(self._levels)
        flat: list[int] = []
        offsets = [0]
        for node in range(n):
            for lvl in range(self._levels[node] + 1):
                flat.extend(self._links[node][lvl])
                offsets.append(len(flat))
        np.savez(
            path,
            vectors=self._vectors[:n],
            levels=np.asarray(self._levels, dtype=np.int32),
            links=np.asarray(flat, dtype=np.int32),
            offsets=np.asarray(offsets, dtype=np.int64),
            deleted=np.asarray(sorted(self._deleted), dtype=np.int64),
            params=np.asarray(
                [self.dim, self.m, self.ef_construction, -1 if self.entry_point is None else self.entry_point,
                 self.max_level],
                dtype=np.int64,
            ),
            metric=np.asarray(self.metric),
        )

    @classmethod
    def load(cls, path) -> "HNSWIndex":
        with np.load(path, allow_pickle=False) as data:
            dim, m, efc, ep, max_level = (int(x) for x in data["params"])
            idx = cls(dim, metric=str(data["metric"]), m=m, ef_construction=efc,
                      capacity=max(1, len(data["levels"])))
            n = len(data["levels"])
            idx._vectors[:n] = data["vectors"]
            idx._levels = data["levels"].tolist()
            flat, offsets = data["links"], data["offsets"]
            pos = 0
            for node in range(n):
                node_links = []
                for _ in range(idx._levels[node] + 1):
                    node_links.append(flat[offsets[pos]: offsets[pos + 1]].tolist())
                    pos += 1
                idx._links.append(node_links)
            idx._deleted = set(data["deleted"].tolist())
            idx.entry_point = None if ep < 0 else ep
            idx.max_level = max_level
        return idx
