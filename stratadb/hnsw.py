"""Hierarchical Navigable Small World (HNSW) graph index.

A from-scratch implementation of Malkov & Yashunin (2016). Two interchangeable
implementations share one on-disk format:

- ``PyHNSWIndex``: pure numpy, always available.
- ``NativeHNSWIndex``: the same algorithm in C (``native/hnsw.c``), loaded
  through ctypes. Much faster to build and query; used automatically when the
  library is present (``HNSWIndex`` points at whichever is in use).

Nodes are dense integer labels ``0..n-1`` assigned in insertion order; the
caller maps them to user ids. Distances are "smaller is closer":

- ``cosine``: ``1 - dot(a, b)`` on unit-normalized vectors
- ``ip``:     ``-dot(a, b)``
- ``l2``:     squared euclidean distance

Deletion is soft (``mark_deleted``): deleted nodes still route searches but
never appear in results. Segments rebuild their index on compaction.
"""

from __future__ import annotations

import ctypes
import heapq
import math
from pathlib import Path
from typing import Callable

import numpy as np

from . import native

METRICS = ("cosine", "ip", "l2")


def _combine_allow(allow, allow_labels):
    if allow_labels is None:
        return allow
    allowed = allow_labels if isinstance(allow_labels, (set, frozenset)) else set(np.asarray(allow_labels).tolist())
    if allow is None:
        return allowed.__contains__
    return lambda label: label in allowed and allow(label)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class PyHNSWIndex:
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
        self._readonly = False
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
        if self._readonly:
            raise RuntimeError("this index is memory-mapped and read-only")
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

    def deleted_labels(self) -> list[int]:
        return sorted(self._deleted)

    def search(
        self,
        vector,
        k: int = 10,
        ef: int | None = None,
        allow: Callable[[int], bool] | None = None,
        allow_labels=None,
    ) -> list[tuple[int, float]]:
        """Return up to k (label, distance) pairs, closest first.

        ``allow`` filters labels (e.g. a metadata predicate); ``allow_labels``
        is the same restriction given as a collection of labels. Filtering
        happens during the search by widening ``ef`` until enough allowed hits
        are found.
        """
        if self.entry_point is None or k <= 0:
            return []
        q = self._prepare(vector)
        ef = max(ef or 64, k)
        allow = _combine_allow(allow, allow_labels)

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

    def brute_force(self, vector, k: int = 10, allow: Callable[[int], bool] | None = None, allow_labels=None):
        """Exact search, used for small/filtered sets and as ground truth in tests."""
        n = len(self._levels)
        if n == 0:
            return []
        q = self._prepare(vector)
        allow = _combine_allow(allow, allow_labels)
        labels = [i for i in range(n) if i not in self._deleted and (allow is None or allow(i))]
        if not labels:
            return []
        dists = self._dist_many(q, labels)
        top = np.argsort(dists)[:k]
        return [(labels[i], float(dists[i])) for i in top]

    # ------------------------------------------------------------ persistence
    def save(self, path, vectors_path=None) -> None:
        """Write the index to ``path`` (.npz). With ``vectors_path`` the vectors
        go to that separate .npy file instead, so ``load`` can memory-map them."""
        n = len(self._levels)
        flat: list[int] = []
        offsets = [0]
        for node in range(n):
            for lvl in range(self._levels[node] + 1):
                flat.extend(self._links[node][lvl])
                offsets.append(len(flat))
        arrays = {}
        if vectors_path is None:
            arrays["vectors"] = self._vectors[:n]
        else:
            np.save(vectors_path, np.ascontiguousarray(self._vectors[:n]))
        np.savez(
            path,
            **arrays,
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
    def load(cls, path, mmap: bool = False) -> "PyHNSWIndex":
        """Load an index. With ``mmap`` and vectors stored in a sibling
        ``vectors.npy`` (see ``save``), the vectors are memory-mapped read-only:
        reopening costs no copying, and pages load from disk on first touch.
        A memory-mapped index is read-only: ``add`` raises."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            dim, m, efc, ep, max_level = (int(x) for x in data["params"])
            n = len(data["levels"])
            idx = cls(dim, metric=str(data["metric"]), m=m, ef_construction=efc, capacity=1)
            if "vectors" in data:
                idx._vectors = np.array(data["vectors"], dtype=np.float32)
            else:
                vec_path = path.with_name("vectors.npy")
                loaded = np.load(vec_path, mmap_mode="r" if mmap else None)
                # A plain ndarray view over the mapping: np.memmap is a subclass,
                # and its per-operation overhead shows up in the search loop.
                idx._vectors = np.asarray(loaded)
                idx._readonly = mmap
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


# ---------------------------------------------------------------------------
# Native implementation
# ---------------------------------------------------------------------------

class NativeHNSWIndex:
    """HNSW index backed by ``native/hnsw.c``. Same API and file format as
    ``PyHNSWIndex``; vectors live in C memory (or in a memory map after a
    ``load(mmap=True)``).

    ``quantize()`` adds 8-bit codes for the stored vectors (``codes.npy`` next
    to ``index.npz`` once saved). Searches then walk the graph on the codes, a
    quarter of the memory traffic, and re-rank what they found with the exact
    vectors, so returned distances are exact. ``PyHNSWIndex`` ignores the codes.
    """

    _METRIC_CODES = {"cosine": 0, "ip": 1, "l2": 2}

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
        lib = native.load()
        if lib is None:
            raise RuntimeError(native.status())
        self._lib = lib
        self.dim = dim
        self.metric = metric
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = max(ef_construction, m)
        self.level_mult = 1.0 / math.log(m)
        self._rng = np.random.default_rng(seed)
        self._vectors: np.ndarray | None = None  # external (memory-mapped) storage, if any
        self._codes: np.ndarray | None = None    # external (memory-mapped) codes, if any
        self._readonly = False
        self._h = lib.hnsw_new(dim, self._METRIC_CODES[metric], m, self.ef_construction, max(1, capacity))
        if not self._h:
            raise MemoryError("could not allocate native HNSW index")

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.hnsw_free(h)
            self._h = None

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self._lib.hnsw_len(self._h))

    @property
    def live_count(self) -> int:
        return len(self) - int(self._lib.hnsw_deleted_count(self._h))

    @property
    def entry_point(self) -> int | None:
        ep = int(self._lib.hnsw_entry(self._h))
        return None if ep < 0 else ep

    @property
    def max_level(self) -> int:
        return int(self._lib.hnsw_max_level(self._h))

    @property
    def vectors(self) -> np.ndarray:
        """The stored (normalized, for cosine) vectors as an ``(n, dim)`` view.
        The view is valid until the next ``add``."""
        n = len(self)
        if self._vectors is not None:
            return self._vectors[:n]
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        ptr = ctypes.cast(self._lib.hnsw_vectors(self._h), ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(ptr, shape=(n, self.dim))

    @staticmethod
    def _ptr(arr: np.ndarray | None):
        return None if arr is None else arr.ctypes.data_as(ctypes.c_void_p)

    def _mask(self, allow, allow_labels) -> np.ndarray | None:
        n = len(self)
        if allow is None and allow_labels is None:
            return None
        mask = np.zeros(n, dtype=np.uint8)
        if allow_labels is not None:
            labels = np.fromiter(allow_labels, dtype=np.int64) if isinstance(allow_labels, (set, frozenset)) \
                else np.asarray(list(allow_labels) if not isinstance(allow_labels, np.ndarray) else allow_labels, dtype=np.int64)
            labels = labels[(labels >= 0) & (labels < n)]
            mask[labels] = 1
            if allow is not None:
                for label in labels.tolist():
                    if not allow(label):
                        mask[label] = 0
        else:
            # A Python predicate has to be evaluated for every label; prefer allow_labels.
            for label in range(n):
                if allow(label):
                    mask[label] = 1
        return mask

    # ----------------------------------------------------------------- insert
    def add_batch(self, vectors) -> list[int]:
        if self._readonly:
            raise RuntimeError("this index is memory-mapped and read-only")
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vecs.shape[-1] if vecs.ndim else 0}")
        n = vecs.shape[0]
        if n == 0:
            return []
        levels = (-np.log(1.0 - self._rng.random(n)) * self.level_mult).astype(np.int32)
        first = int(self._lib.hnsw_add_batch(self._h, self._ptr(vecs), self._ptr(levels), n))
        if first == -2:
            raise RuntimeError("this index is memory-mapped and read-only")
        if first < 0:
            raise MemoryError("native HNSW insert failed")
        return list(range(first, first + n))

    def add(self, vector) -> int:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        return self.add_batch(v.reshape(1, -1))[0]

    def mark_deleted(self, label: int) -> None:
        self._lib.hnsw_mark_deleted(self._h, int(label))

    def is_deleted(self, label: int) -> bool:
        return bool(self._lib.hnsw_is_deleted(self._h, int(label)))

    def deleted_labels(self) -> list[int]:
        return self._deleted().tolist()

    # ----------------------------------------------------------- quantization
    @property
    def quantized(self) -> bool:
        return bool(self._lib.hnsw_quantized(self._h))

    def quantize(self) -> None:
        """Build 8-bit scalar codes (per-dimension min/max) for the stored
        vectors. Adding vectors afterwards drops them; call again when done."""
        if self._lib.hnsw_quantize(self._h) != 0:
            raise MemoryError("native HNSW quantization failed")
        self._codes = None

    def _qarray(self, getter, shape, ctype, dtype) -> np.ndarray:
        ptr = ctypes.cast(getter(self._h), ctypes.POINTER(ctype))
        return np.array(np.ctypeslib.as_array(ptr, shape=shape), dtype=dtype)

    def _quant_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self)
        return (self._qarray(self._lib.hnsw_qmin, (self.dim,), ctypes.c_float, np.float32),
                self._qarray(self._lib.hnsw_qscale, (self.dim,), ctypes.c_float, np.float32),
                self._qarray(self._lib.hnsw_qnorm, (n,), ctypes.c_float, np.float32))

    @property
    def codes(self) -> np.ndarray | None:
        """The ``(n, dim)`` uint8 codes, or None when not quantized."""
        if not self.quantized:
            return None
        if self._codes is not None:
            return self._codes
        ptr = ctypes.cast(self._lib.hnsw_codes(self._h), ctypes.POINTER(ctypes.c_uint8))
        return np.ctypeslib.as_array(ptr, shape=(len(self), self.dim))

    # ----------------------------------------------------------------- search
    def search(
        self,
        vector,
        k: int = 10,
        ef: int | None = None,
        allow: Callable[[int], bool] | None = None,
        allow_labels=None,
    ) -> list[tuple[int, float]]:
        """Return up to k (label, distance) pairs, closest first. See ``PyHNSWIndex.search``."""
        n = len(self)
        if n == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {q.shape[0]}")
        mask = self._mask(allow, allow_labels)
        labels = np.empty(k, dtype=np.int64)
        dists = np.empty(k, dtype=np.float32)
        cnt = int(self._lib.hnsw_search(self._h, self._ptr(q), k, int(ef or 0), self._ptr(mask),
                                        self._ptr(labels), self._ptr(dists)))
        if cnt < 0:
            raise MemoryError("native HNSW search failed")
        return list(zip(labels[:cnt].tolist(), dists[:cnt].tolist()))

    def brute_force(self, vector, k: int = 10, allow: Callable[[int], bool] | None = None, allow_labels=None):
        """Exact search, used for small/filtered sets and as ground truth in tests."""
        n = len(self)
        if n == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {q.shape[0]}")
        mask = self._mask(allow, allow_labels)
        k = min(k, n)
        labels = np.empty(k, dtype=np.int64)
        dists = np.empty(k, dtype=np.float32)
        cnt = int(self._lib.hnsw_brute_force(self._h, self._ptr(q), k, self._ptr(mask),
                                             self._ptr(labels), self._ptr(dists)))
        if cnt < 0:
            raise MemoryError("native HNSW search failed")
        return list(zip(labels[:cnt].tolist(), dists[:cnt].tolist()))

    # ------------------------------------------------------------ persistence
    def _levels(self) -> np.ndarray:
        n = len(self)
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        ptr = ctypes.cast(self._lib.hnsw_levels(self._h), ctypes.POINTER(ctypes.c_int32))
        return np.array(np.ctypeslib.as_array(ptr, shape=(n,)), dtype=np.int32)

    def _deleted(self) -> np.ndarray:
        n = len(self)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        ptr = ctypes.cast(self._lib.hnsw_deleted(self._h), ctypes.POINTER(ctypes.c_uint8))
        return np.flatnonzero(np.ctypeslib.as_array(ptr, shape=(n,))).astype(np.int64)

    def save(self, path, vectors_path=None) -> None:
        """Write the index to ``path`` (.npz), in the format ``PyHNSWIndex`` uses."""
        n = len(self)
        levels = self._levels()
        total = int(self._lib.hnsw_links_total(self._h))
        flat = np.empty(total, dtype=np.int32)
        offsets = np.empty(int(levels.sum()) + n + 1, dtype=np.int64)
        if n:
            self._lib.hnsw_export_links(self._h, self._ptr(flat), self._ptr(offsets))
        else:
            offsets[0] = 0
        arrays = {}
        vectors = np.ascontiguousarray(self.vectors)
        if vectors_path is None:
            arrays["vectors"] = vectors
        else:
            np.save(vectors_path, vectors)
        if self.quantized and n:
            qmin, qscale, qnorm = self._quant_params()
            arrays.update(quant_min=qmin, quant_scale=qscale, quant_norm=qnorm)
            if vectors_path is None:
                arrays["codes"] = np.ascontiguousarray(self.codes)
            else:
                np.save(Path(vectors_path).with_name("codes.npy"), np.ascontiguousarray(self.codes))
        ep = self.entry_point
        np.savez(
            path,
            **arrays,
            levels=levels,
            links=flat,
            offsets=offsets,
            deleted=self._deleted(),
            params=np.asarray([self.dim, self.m, self.ef_construction, -1 if ep is None else ep, self.max_level],
                              dtype=np.int64),
            metric=np.asarray(self.metric),
        )

    @classmethod
    def load(cls, path, mmap: bool = False) -> "NativeHNSWIndex":
        """Load an index written by either implementation. See ``PyHNSWIndex.load``."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            dim, m, efc, ep, max_level = (int(x) for x in data["params"])
            levels = np.ascontiguousarray(data["levels"], dtype=np.int32)
            n = len(levels)
            idx = cls(dim, metric=str(data["metric"]), m=m, ef_construction=efc, capacity=max(1, n))
            external = None
            if "vectors" in data:
                vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)
            else:
                loaded = np.load(path.with_name("vectors.npy"), mmap_mode="r" if mmap else None)
                vectors = np.asarray(loaded)
                if mmap:
                    external = vectors
            flat = np.ascontiguousarray(data["links"], dtype=np.int32)
            offsets = np.ascontiguousarray(data["offsets"], dtype=np.int64)
            deleted = np.zeros(n, dtype=np.uint8)
            if len(data["deleted"]):
                deleted[np.asarray(data["deleted"], dtype=np.int64)] = 1
            quant = None
            if "quant_min" in data:
                if "codes" in data:
                    codes, codes_external = np.ascontiguousarray(data["codes"], dtype=np.uint8), None
                else:
                    loaded = np.load(path.with_name("codes.npy"), mmap_mode="r" if mmap else None)
                    codes = np.asarray(loaded)
                    codes_external = codes if mmap else None
                quant = (codes, codes_external,
                         np.ascontiguousarray(data["quant_min"], dtype=np.float32),
                         np.ascontiguousarray(data["quant_scale"], dtype=np.float32),
                         np.ascontiguousarray(data["quant_norm"], dtype=np.float32))
        if n == 0:
            return idx
        if vectors.shape != (n, dim):
            raise ValueError(f"vectors have shape {vectors.shape}, expected {(n, dim)}")
        rc = idx._lib.hnsw_import(idx._h, n, cls._ptr(vectors), 0 if external is not None else 1,
                                  cls._ptr(levels), cls._ptr(flat), cls._ptr(offsets), cls._ptr(deleted),
                                  ep, max_level)
        if rc != 0:
            raise MemoryError(f"native HNSW import failed ({rc})")
        if external is not None:
            idx._vectors = external  # keep the mapping alive; C reads from it
            idx._readonly = True
        if quant is not None:
            codes, codes_external, qmin, qscale, qnorm = quant
            if codes.shape != (n, dim):
                raise ValueError(f"codes have shape {codes.shape}, expected {(n, dim)}")
            rc = idx._lib.hnsw_import_codes(idx._h, cls._ptr(codes), 0 if codes_external is not None else 1,
                                            cls._ptr(qmin), cls._ptr(qscale), cls._ptr(qnorm))
            if rc != 0:
                raise MemoryError("native HNSW code import failed")
            if codes_external is not None:
                idx._codes = codes_external
        return idx


HNSWIndex = NativeHNSWIndex if native.available() else PyHNSWIndex
