"""A collection: one vector space with metadata, typed edges and durable storage.

On-disk layout of ``<root>/<name>/``::

    config.json            dim, metric, index parameters (written once)
    manifest.json          checkpoint: LSN, sealed segments + tombstones, snapshot files
    wal.log                mutations since the checkpoint
    growing-<lsn>.npz/json snapshot of the growing segment
    graph-<lsn>.json       snapshot of the edge store
    seg-<n>/               sealed segments (HNSW index + records)

Write path: append to the WAL, then apply in memory. ``checkpoint()`` writes
the snapshot files under new names, atomically swaps the manifest to point at
them, and only then empties the WAL. Opening a collection loads the manifest
and replays the WAL on top, so a crash at any point recovers every
acknowledged write.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .filters import compile_filter
from .graph import GraphStore
from .hnsw import METRICS
from .segment import GrowingSegment, SealedSegment, atomic_write_json, distances, prepare, save_records
from .wal import WriteAheadLog

log = logging.getLogger("stratadb")

GROWING = -1  # locator value: record lives in the growing segment
FROZEN = -2   # locator value: record lives in the segment being sealed


@dataclass
class CollectionConfig:
    dim: int
    metric: str = "cosine"
    m: int = 16
    ef_construction: int = 128
    ef_search: int = 64
    seal_threshold: int = 16384
    sync: str = "flush"
    wal_checkpoint_bytes: int = 64 * 1024 * 1024
    compact_deleted_ratio: float = 0.3
    # Build segment indexes on a worker thread instead of blocking writes.
    background_seal: bool = True
    # Segments with (live records x dim) at or below this are scanned exactly:
    # a single matrix product beats a Python graph walk at that size.
    exact_search_limit: int = 3_000_000

    def validate(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if self.seal_threshold < 2:
            raise ValueError("seal_threshold must be at least 2")


def similarity(distance: float, metric: str) -> float:
    """Turn a distance into a "bigger is better" score."""
    return 1.0 - distance if metric == "cosine" else -distance


class Collection:
    def __init__(self, path: str | Path, config: CollectionConfig | None = None):
        self.path = Path(path)
        self.name = self.path.name
        self._lock = threading.RLock()
        cfg_path = self.path / "config.json"
        if cfg_path.exists():
            stored = CollectionConfig(**json.loads(cfg_path.read_text()))
            if config is not None and (config.dim, config.metric) != (stored.dim, stored.metric):
                raise ValueError(
                    f"collection {self.name!r} exists with dim={stored.dim} metric={stored.metric}"
                )
            self.config = stored
        else:
            if config is None:
                raise FileNotFoundError(f"no collection at {self.path}")
            config.validate()
            self.path.mkdir(parents=True, exist_ok=True)
            atomic_write_json(cfg_path, asdict(config))
            self.config = config

        self.growing = GrowingSegment(self.config.dim, self.config.metric)
        self.sealed: dict[int, SealedSegment] = {}
        self.locator: dict[str, int] = {}
        self.graph = GraphStore()
        self._next_seg = 1
        self._checkpoint_lsn = 0
        self._files: dict[str, str] = {}
        self._generation = 0
        self.frozen: GrowingSegment | None = None
        self._seal_thread: threading.Thread | None = None
        self._seal_error: Exception | None = None
        self._closed = False
        self._recover()

    # ------------------------------------------------------------- recovery
    def _recover(self) -> None:
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self._checkpoint_lsn = manifest["lsn"]
            self._next_seg = manifest["next_segment"]
            self._generation = manifest.get("generation", 0)
            self._files = manifest.get("files", {})
            for seg in manifest["segments"]:
                seg_path = self.path / f"seg-{seg['id']}"
                segment = SealedSegment.load(seg["id"], seg_path, seg["tombstones"])
                self.sealed[seg["id"]] = segment
                for rid in segment.pos:
                    self.locator[rid] = seg["id"]
            if "growing" in self._files:
                npz, js = self._files["growing"]
                self.growing = GrowingSegment.load(
                    self.config.dim, self.config.metric, self.path / npz, self.path / js
                )
                for rid in self.growing.pos:
                    self.locator[rid] = GROWING
            if "graph" in self._files:
                self.graph = GraphStore.from_list(json.loads((self.path / self._files["graph"]).read_text()))
        self._remove_orphans()

        self.wal = WriteAheadLog(self.path / "wal.log", sync=self.config.sync)
        if self.wal.last_lsn < self._checkpoint_lsn:
            self.wal.last_lsn = self._checkpoint_lsn
        replayed = 0
        for entry in self.wal.replay(after_lsn=self._checkpoint_lsn):
            self._apply(entry.op, entry.data, entry.vectors)
            replayed += 1
        self.replayed_entries = replayed
        if self._growing_full:
            self.seal(only_if_full=True)

    def _remove_orphans(self) -> None:
        """Delete files a crash left behind that the manifest doesn't reference."""
        keep = {f"seg-{sid}" for sid in self.sealed} | {"config.json", "manifest.json", "wal.log"}
        for value in self._files.values():
            keep.update([value] if isinstance(value, str) else value)
        for child in self.path.iterdir():
            if child.name in keep:
                continue
            if child.is_dir() and child.name.startswith("seg-"):
                shutil.rmtree(child, ignore_errors=True)
            elif child.is_file() and child.name.startswith(("growing-", "graph-", "manifest.json.tmp")):
                child.unlink(missing_ok=True)

    # --------------------------------------------------------------- apply
    def _apply(self, op: str, data: dict, vectors: np.ndarray | None) -> None:
        if op == "upsert":
            metas = data.get("metadata") or [{} for _ in data["ids"]]
            for rid, vec, meta in zip(data["ids"], vectors, metas):
                self._remove_location(rid)
                self.growing.put(rid, vec, meta)
                self.locator[rid] = GROWING
        elif op == "delete":
            for rid in data["ids"]:
                self._remove_location(rid)
                self.graph.remove_node(rid)
        elif op == "link":
            for e in data["edges"]:
                self.graph.add(e["src"], e["dst"], e.get("rel", "related"), e.get("weight", 1.0), e.get("props"))
        elif op == "unlink":
            for e in data["edges"]:
                self.graph.remove(e["src"], e["dst"], e.get("rel"))
        else:
            raise ValueError(f"unknown WAL op {op}")

    def _segment(self, where: int):
        if where == GROWING:
            return self.growing
        if where == FROZEN:
            return self.frozen
        return self.sealed[where]

    def _remove_location(self, rid: str) -> bool:
        where = self.locator.pop(rid, None)
        if where is None:
            return False
        self._segment(where).delete(rid)
        return True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"collection {self.name!r} is closed")

    def _maybe_checkpoint(self) -> None:
        if self.wal.size_bytes() >= self.config.wal_checkpoint_bytes:
            self.checkpoint()

    @property
    def _growing_full(self) -> bool:
        return len(self.growing) >= self.config.seal_threshold

    # ------------------------------------------------------------ write API
    def upsert(self, ids: list[str], vectors, metadata: list[dict] | None = None) -> int:
        ids = [str(i) for i in ids]
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape != (len(ids), self.config.dim):
            raise ValueError(f"expected vectors of shape ({len(ids)}, {self.config.dim}), got {vecs.shape}")
        if not np.all(np.isfinite(vecs)):
            raise ValueError("vectors contain NaN or inf")
        if metadata is not None and len(metadata) != len(ids):
            raise ValueError("metadata must have one entry per id")
        if len(set(ids)) != len(ids):
            # Keep the last occurrence, like applying the writes one by one.
            last = {rid: i for i, rid in enumerate(ids)}
            order = sorted(last.values())
            ids = [ids[i] for i in order]
            vecs = vecs[order]
            metadata = [metadata[i] for i in order] if metadata is not None else None
        vecs = prepare(vecs, self.config.metric)
        # Feed the growing segment in slices so a big batch seals into
        # threshold-sized segments instead of one oversized one. Sealing runs
        # outside the lock: it may wait for a previous background build.
        start = 0
        while start < len(ids):
            with self._lock:
                self._check_open()
                room = max(1, self.config.seal_threshold - len(self.growing))
                end = min(len(ids), start + room)
                data = {"ids": ids[start:end], "metadata": metadata[start:end] if metadata is not None else None}
                self.wal.append("upsert", data, vecs[start:end])
                self._apply("upsert", data, vecs[start:end])
                full = self._growing_full
                if not full:
                    self._maybe_checkpoint()
            if full:
                self.seal(only_if_full=True)
            start = end
        return len(ids)

    def delete(self, ids: Iterable[str]) -> int:
        ids = [str(i) for i in ids]
        with self._lock:
            self._check_open()
            present = [rid for rid in ids if rid in self.locator]
            # Edges can exist for ids that were never inserted, so log those too.
            targets = [rid for rid in ids if rid in self.locator or self.graph.degree(rid)]
            if targets:
                self.wal.append("delete", {"ids": targets})
                self._apply("delete", {"ids": targets}, None)
                self._maybe_checkpoint()
            return len(present)

    def link(self, edges: list[dict]) -> int:
        """Add edges: ``[{"src", "dst", "rel"?, "weight"?, "props"?}]``."""
        clean = []
        for e in edges:
            if "src" not in e or "dst" not in e:
                raise ValueError("each edge needs src and dst")
            clean.append({
                "src": str(e["src"]), "dst": str(e["dst"]), "rel": str(e.get("rel", "related")),
                "weight": float(e.get("weight", 1.0)), "props": dict(e.get("props") or {}),
            })
        with self._lock:
            self._check_open()
            if clean:
                self.wal.append("link", {"edges": clean})
                self._apply("link", {"edges": clean}, None)
                self._maybe_checkpoint()
        return len(clean)

    def unlink(self, edges: list[dict]) -> int:
        clean = [{"src": str(e["src"]), "dst": str(e["dst"]), "rel": e.get("rel")} for e in edges]
        with self._lock:
            self._check_open()
            before = len(self.graph)
            if clean:
                self.wal.append("unlink", {"edges": clean})
                self._apply("unlink", {"edges": clean}, None)
            return before - len(self.graph)

    # ------------------------------------------------------------- read API
    def __len__(self) -> int:
        return len(self.locator)

    def __contains__(self, rid: str) -> bool:
        return rid in self.locator

    def _locate(self, rid: str):
        where = self.locator.get(rid)
        if where is None:
            return None
        return self._segment(where).get(rid)

    def get(self, ids: Iterable[str], include_vectors: bool = False) -> list[dict]:
        out = []
        with self._lock:
            for rid in ids:
                found = self._locate(str(rid))
                if found is None:
                    continue
                vec, meta = found
                rec: dict[str, Any] = {"id": str(rid), "metadata": meta}
                if include_vectors:
                    rec["vector"] = vec.tolist()
                out.append(rec)
        return out

    def scan(self, filter: dict | None = None, limit: int | None = None) -> list[dict]:
        pred = compile_filter(filter)
        out = []
        with self._lock:
            segments = [self.growing, *([self.frozen] if self.frozen is not None else []), *self.sealed.values()]
            for seg in segments:
                for rid, _, meta in seg.records():
                    if pred is None or pred(meta):
                        out.append({"id": rid, "metadata": meta})
                        if limit is not None and len(out) >= limit:
                            return out
        return out

    def search(
        self,
        vector,
        k: int = 10,
        filter: dict | None = None,
        ef: int | None = None,
        include_vectors: bool = False,
        expand_hops: int = 0,
        graph_boost: float = 0.0,
    ) -> list[dict]:
        """Nearest-neighbour search across all segments.

        ``filter``       metadata filter (see ``stratadb.filters``)
        ``expand_hops``  attach graph neighbours (up to N hops) to each hit
        ``graph_boost``  0..1; re-rank with spreading activation over edges, so
                         records strongly linked to many good matches move up
                         and linked records outside the vector top-k can enter
        """
        if k <= 0:
            return []
        if not 0.0 <= graph_boost <= 1.0:
            raise ValueError("graph_boost must be between 0 and 1")
        q = prepare(np.asarray(vector, dtype=np.float32).reshape(-1), self.config.metric)
        if q.shape[0] != self.config.dim:
            raise ValueError(f"expected dim {self.config.dim}, got {q.shape[0]}")
        pred = compile_filter(filter)
        ef = ef or self.config.ef_search
        pool_k = max(k * 4, 32) if graph_boost > 0 else k

        with self._lock:
            self._check_open()
            hits = self.growing.search(q, pool_k, pred)
            if self.frozen is not None:
                hits.extend(self.frozen.search(q, pool_k, pred))
            for seg in self.sealed.values():
                hits.extend(seg.search(q, pool_k, ef=max(ef, pool_k), pred=pred,
                                       exact_limit=self.config.exact_search_limit))
            hits.sort()
            hits = hits[:pool_k]
            scored = [(similarity(d, self.config.metric), d, rid) for d, rid in hits]
            if graph_boost > 0 and scored:
                scored = self._graph_rerank(q, scored, graph_boost, pred)
            results = []
            for item in scored[:k]:
                score, dist, rid = item[0], item[1], item[2]
                vec, meta = self._locate(rid)
                res: dict[str, Any] = {"id": rid, "score": score, "distance": dist, "metadata": meta}
                if graph_boost > 0:
                    res["vector_score"] = item[3]
                    res["graph_score"] = item[4]
                if include_vectors:
                    res["vector"] = vec.tolist()
                if expand_hops > 0:
                    res["neighbors"] = self.graph.neighbors(rid, hops=expand_hops)
                results.append(res)
            return results

    def _graph_rerank(self, q, scored, alpha, pred):
        """Spreading activation over the edge graph.

        Only candidates that stand out from the pool spread activation: a seed's
        strength is how far its similarity sits above the pool mean, so a weak,
        barely-related match can't promote its neighbours. Each seed splits its
        strength across its edges by weight. Activation is scaled so a record
        linked only to the strongest seed gets 1.0, and the final score blends
        the record's own similarity with the activation it received.
        """
        sims = {rid: s for s, _, rid in scored}
        dists = {rid: d for _, d, rid in scored}
        background = sum(sims.values()) / len(sims)
        strength = {rid: max(sim - background, 0.0) for rid, sim in sims.items()}
        top_strength = max(strength.values(), default=0.0) or 1.0
        activation: dict[str, float] = {}
        for rid, s in strength.items():
            if s <= 0:
                continue
            nbrs = self.graph.weighted_neighbors(rid)
            if not nbrs:
                continue
            total_w = sum(abs(w) for _, w in nbrs) or 1.0
            for nbr, w in nbrs:
                activation[nbr] = activation.get(nbr, 0.0) + s * abs(w) / total_w

        # Pull in linked records that vector search missed (if they pass the filter).
        for nbr in activation:
            if nbr in sims:
                continue
            found = self._locate(nbr)
            if found is None or (pred is not None and not pred(found[1])):
                continue
            d = float(distances(q, found[0].reshape(1, -1), self.config.metric)[0])
            sims[nbr], dists[nbr] = similarity(d, self.config.metric), d

        reranked = []
        for rid, sim in sims.items():
            act = min(activation.get(rid, 0.0) / top_strength, 1.0)
            reranked.append(((1 - alpha) * sim + alpha * act, dists[rid], rid, sim, act))
        reranked.sort(key=lambda t: -t[0])
        return reranked

    def neighbors(self, rid: str, hops: int = 1, rel: str | None = None,
                  direction: str = "both", limit: int | None = None, hydrate: bool = False) -> list[dict]:
        with self._lock:
            result = self.graph.neighbors(str(rid), hops=hops, rel=rel, direction=direction, limit=limit)
            if hydrate:
                for item in result:
                    found = self._locate(item["id"])
                    item["metadata"] = found[1] if found else None
            return result

    def edges(self, rid: str, direction: str = "both", rel: str | None = None) -> list[dict]:
        with self._lock:
            return self.graph.edges(str(rid), direction=direction, rel=rel)

    # ---------------------------------------------------------- maintenance
    def seal(self, wait: bool | None = None, only_if_full: bool = False) -> int | None:
        """Freeze the growing segment and build its HNSW index.

        With ``background_seal`` (the default) the index builds on a worker
        thread: writes go to a fresh growing segment and the frozen one is still
        searched by brute force until the indexed segment is swapped in. Only
        one build runs at a time; a second seal waits for the first.
        """
        background = self.config.background_seal if wait is None else not wait
        self._check_open()
        self.wait_for_seal()
        with self._lock:
            self._check_open()
            if not len(self.growing) or (only_if_full and not self._growing_full):
                return None
            seg_id = self._next_seg
            self._next_seg += 1
            frozen = self.growing
            self.frozen = frozen
            self.growing = GrowingSegment(self.config.dim, self.config.metric)
            for rid in frozen.pos:
                self.locator[rid] = FROZEN
            records = [(rid, vec.copy(), meta) for rid, vec, meta in frozen.records()]
            if background:
                self._seal_thread = threading.Thread(
                    target=self._build_and_install, args=(seg_id, frozen, records),
                    name=f"stratadb-seal-{self.name}-{seg_id}", daemon=True,
                )
                self._seal_thread.start()
                return seg_id
        self._build_and_install(seg_id, frozen, records)
        return seg_id

    def _build_and_install(self, seg_id: int, frozen: GrowingSegment, records) -> None:
        try:
            segment = SealedSegment.build(
                seg_id, self.path / f"seg-{seg_id}", records,
                self.config.metric, self.config.m, self.config.ef_construction,
            )
        except Exception as e:  # noqa: BLE001 - keep the data searchable, surface the error
            log.exception("sealing segment %s of %s failed", seg_id, self.name)
            with self._lock:
                self._seal_error = e
                for rid, vec, meta in frozen.records():
                    if rid not in self.growing.pos:
                        self.growing.put(rid, vec, meta)
                        self.locator[rid] = GROWING
                self.frozen = None
            return
        with self._lock:
            # Records deleted or overwritten while the index was building left
            # the frozen segment; tombstone them in the new one.
            for rid in list(segment.pos):
                if rid in frozen.pos:
                    self.locator[rid] = seg_id
                else:
                    segment.delete(rid)
            self.sealed[seg_id] = segment
            self.frozen = None
            self._seal_error = None
            if not self._closed:
                self.checkpoint()

    def wait_for_seal(self) -> None:
        """Block until a background seal (if any) has finished."""
        thread = self._seal_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join()

    def compact(self) -> dict:
        """Rewrite sealed segments that are mostly tombstones, and merge small ones."""
        self.wait_for_seal()
        with self._lock:
            self._check_open()
            threshold = self.config.seal_threshold
            victims = [
                s for s in self.sealed.values()
                if s.deleted_ratio >= self.config.compact_deleted_ratio or len(s) < threshold // 2
            ]
            if not victims or (len(victims) == 1 and victims[0].deleted_ratio == 0):
                return {"rewritten": 0, "segments": len(self.sealed)}
            records = [r for s in victims for r in s.records()]
            for s in victims:
                del self.sealed[s.seg_id]
            new_ids = []
            for start in range(0, len(records), threshold):
                chunk = [(rid, vec.copy(), meta) for rid, vec, meta in records[start:start + threshold]]
                seg_id = self._next_seg
                self._next_seg += 1
                segment = SealedSegment.build(
                    seg_id, self.path / f"seg-{seg_id}", chunk,
                    self.config.metric, self.config.m, self.config.ef_construction,
                )
                self.sealed[seg_id] = segment
                for rid in segment.pos:
                    self.locator[rid] = seg_id
                new_ids.append(seg_id)
            self.checkpoint()
            for s in victims:
                shutil.rmtree(s.path, ignore_errors=True)
            return {"rewritten": len(victims), "new_segments": new_ids, "segments": len(self.sealed)}

    def checkpoint(self) -> int:
        with self._lock:
            self._check_open()
            lsn = self.wal.last_lsn
            # Snapshot names carry a generation that bumps on every checkpoint,
            # so a new snapshot never overwrites the one the live manifest uses.
            self._generation += 1
            gen = self._generation
            files = {
                "growing": [f"growing-{gen}.npz", f"growing-{gen}.json"],
                "graph": f"graph-{gen}.json",
            }
            # A segment being sealed in the background isn't on disk yet, so its
            # records go into the growing snapshot too.
            records = list(self.frozen.records()) if self.frozen is not None else []
            records += list(self.growing.records())
            save_records(records, self.config.dim, self.path / files["growing"][0], self.path / files["growing"][1])
            atomic_write_json(self.path / files["graph"], self.graph.to_list())
            manifest = {
                "lsn": lsn,
                "generation": gen,
                "next_segment": self._next_seg,
                "segments": [{"id": sid, "tombstones": s.tombstones} for sid, s in sorted(self.sealed.items())],
                "files": files,
            }
            atomic_write_json(self.path / "manifest.json", manifest)
            old = self._files
            self._files = files
            self._checkpoint_lsn = lsn
            self.wal.reset(next_lsn=lsn)
            for value in old.values():
                for name in [value] if isinstance(value, str) else value:
                    if name not in (files["graph"], *files["growing"]):
                        (self.path / name).unlink(missing_ok=True)
            return lsn

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "count": len(self.locator),
                "dim": self.config.dim,
                "metric": self.config.metric,
                "growing": len(self.growing),
                "sealing": len(self.frozen) if self.frozen is not None else 0,
                "sealed_segments": [
                    {"id": sid, "live": len(s), "total": s.total, "deleted_ratio": round(s.deleted_ratio, 3)}
                    for sid, s in sorted(self.sealed.items())
                ],
                "edges": len(self.graph),
                "wal_bytes": self.wal.size_bytes(),
                "checkpoint_lsn": self._checkpoint_lsn,
                "last_lsn": self.wal.last_lsn,
            }

    def close(self) -> None:
        self.wait_for_seal()
        with self._lock:
            if self._closed:
                return
            if self.wal.last_lsn > self._checkpoint_lsn:
                self.checkpoint()
            self.wal.close()
            self._closed = True
