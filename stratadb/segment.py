"""Segments: the units records are stored and searched in.

New writes land in a single *growing* segment, which is searched by brute
force. When it reaches the collection's seal threshold it is frozen into an
immutable *sealed* segment with its own HNSW index and written to disk.
Deleting or overwriting a sealed record only adds a tombstone; compaction
later rewrites segments that carry too many of them.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .hnsw import HNSWIndex

Hit = tuple[float, str]  # (distance, id)


def prepare(vectors: np.ndarray, metric: str) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if metric == "cosine":
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
    return vectors


def distances(q: np.ndarray, vecs: np.ndarray, metric: str) -> np.ndarray:
    if metric == "l2":
        diff = vecs - q
        return np.einsum("ij,ij->i", diff, diff)
    dots = vecs @ q
    return 1.0 - dots if metric == "cosine" else -dots


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode())


class GrowingSegment:
    def __init__(self, dim: int, metric: str, capacity: int = 256):
        self.dim = dim
        self.metric = metric
        self._vectors = np.zeros((capacity, dim), dtype=np.float32)
        self.ids: list[str] = []
        self.metas: list[dict] = []
        self.alive: list[bool] = []
        self.pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.pos)

    def put(self, rid: str, vector: np.ndarray, meta: dict) -> None:
        """Insert or overwrite. ``vector`` must already be prepared for the metric."""
        if rid in self.pos:
            label = self.pos[rid]
            self._vectors[label] = vector
            self.metas[label] = meta
            return
        label = len(self.ids)
        if label >= self._vectors.shape[0]:
            bigger = np.zeros((self._vectors.shape[0] * 2, self.dim), dtype=np.float32)
            bigger[:label] = self._vectors[:label]
            self._vectors = bigger
        self._vectors[label] = vector
        self.ids.append(rid)
        self.metas.append(meta)
        self.alive.append(True)
        self.pos[rid] = label

    def delete(self, rid: str) -> bool:
        label = self.pos.pop(rid, None)
        if label is None:
            return False
        self.alive[label] = False
        self.metas[label] = {}
        return True

    def get(self, rid: str) -> tuple[np.ndarray, dict] | None:
        label = self.pos.get(rid)
        if label is None:
            return None
        return self._vectors[label], self.metas[label]

    def records(self) -> Iterator[tuple[str, np.ndarray, dict]]:
        for rid, label in self.pos.items():
            yield rid, self._vectors[label], self.metas[label]

    def search(self, q: np.ndarray, k: int, pred: Callable[[dict], bool] | None = None) -> list[Hit]:
        labels = [
            label for label in self.pos.values()
            if pred is None or pred(self.metas[label])
        ]
        if not labels:
            return []
        dists = distances(q, self._vectors[labels], self.metric)
        top = np.argsort(dists)[:k]
        return [(float(dists[i]), self.ids[labels[i]]) for i in top]

    # Snapshot: only live records are written, so a reload compacts the segment.
    def save(self, npz_path: Path, json_path: Path) -> None:
        live = list(self.pos.items())
        labels = [label for _, label in live]
        tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
        np.savez(tmp, vectors=self._vectors[labels] if labels else np.zeros((0, self.dim), np.float32))
        os.replace(tmp, npz_path)
        atomic_write_json(json_path, {"ids": [rid for rid, _ in live], "metas": [self.metas[l] for l in labels]})

    @classmethod
    def load(cls, dim: int, metric: str, npz_path: Path, json_path: Path) -> "GrowingSegment":
        seg = cls(dim, metric)
        with np.load(npz_path) as data:
            vectors = data["vectors"]
        info = json.loads(json_path.read_text())
        for rid, vec, meta in zip(info["ids"], vectors, info["metas"]):
            seg.put(rid, vec, meta)
        return seg


class SealedSegment:
    def __init__(self, seg_id: int, path: Path, index: HNSWIndex, ids: list[str], metas: list[dict]):
        self.seg_id = seg_id
        self.path = path
        self.index = index
        self.ids = ids
        self.metas = metas
        self.pos = {rid: label for label, rid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.pos)

    @property
    def total(self) -> int:
        return len(self.ids)

    @property
    def deleted_ratio(self) -> float:
        return 0.0 if not self.ids else 1.0 - len(self.pos) / len(self.ids)

    @property
    def tombstones(self) -> list[int]:
        return sorted(self.index._deleted)

    @classmethod
    def build(
        cls,
        seg_id: int,
        path: Path,
        records: list[tuple[str, np.ndarray, dict]],
        metric: str,
        m: int,
        ef_construction: int,
    ) -> "SealedSegment":
        dim = len(records[0][1])
        index = HNSWIndex(dim, metric=metric, m=m, ef_construction=ef_construction, capacity=len(records))
        for _, vec, _ in records:
            index.add(vec)
        ids = [rid for rid, _, _ in records]
        metas = [meta for _, _, meta in records]
        tmp = path.with_name(path.name + ".building")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        index.save(tmp / "index.npz")
        atomic_write_json(tmp / "records.json", {"ids": ids, "metas": metas})
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
        return cls(seg_id, path, index, ids, metas)

    @classmethod
    def load(cls, seg_id: int, path: Path, tombstones: list[int]) -> "SealedSegment":
        index = HNSWIndex.load(path / "index.npz")
        info = json.loads((path / "records.json").read_text())
        seg = cls(seg_id, path, index, info["ids"], info["metas"])
        for label in tombstones:
            seg._tombstone(label)
        return seg

    def _tombstone(self, label: int) -> None:
        self.index.mark_deleted(label)
        self.pos.pop(self.ids[label], None)

    def delete(self, rid: str) -> bool:
        label = self.pos.get(rid)
        if label is None:
            return False
        self._tombstone(label)
        return True

    def get(self, rid: str) -> tuple[np.ndarray, dict] | None:
        label = self.pos.get(rid)
        if label is None:
            return None
        return self.index.vectors[label], self.metas[label]

    def records(self) -> Iterator[tuple[str, np.ndarray, dict]]:
        vecs = self.index.vectors
        for rid, label in self.pos.items():
            yield rid, vecs[label], self.metas[label]

    def search(
        self, q: np.ndarray, k: int, ef: int | None = None, pred: Callable[[dict], bool] | None = None
    ) -> list[Hit]:
        if not self.pos:
            return []
        if pred is None:
            hits = self.index.search(q, k, ef=ef)
            return [(d, self.ids[label]) for label, d in hits]

        allowed = [label for label in self.pos.values() if pred(self.metas[label])]
        if not allowed:
            return []
        # Selective filters: exact search over the few matches beats walking a
        # graph where almost every node is rejected.
        if len(allowed) <= max(k * 32, len(self.pos) // 20):
            dists = distances(q, self.index.vectors[allowed], self.index.metric)
            top = np.argsort(dists)[:k]
            return [(float(dists[i]), self.ids[allowed[i]]) for i in top]
        allowed_set = set(allowed)
        hits = self.index.search(q, k, ef=ef, allow=allowed_set.__contains__)
        return [(d, self.ids[label]) for label, d in hits]
