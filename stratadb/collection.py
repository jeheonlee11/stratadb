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
from .lockfile import DirectoryLock
from .text_index import BM25Index, fuse_rrf
from .metaindex import MetaIndex
from .segment import GrowingSegment, SealedSegment, atomic_write_json, distances, prepare, save_records
from .wal import WriteAheadLog

log = logging.getLogger("stratadb")

GROWING = -1  # locator value: record lives in the growing segment
# A record being sealed is located by its future segment id; _segment() checks
# the in-progress (frozen) segments before the sealed ones.


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
    # Build indexes in worker processes (one per concurrent build) so builds
    # use other CPU cores instead of sharing this process's GIL. Only segments
    # with at least build_process_min_records go to a worker: starting one
    # costs about a second, which small segments don't repay.
    build_processes: int = 2
    build_process_min_records: int = 2000
    # Segments with (live records x dim) at or below this are scanned exactly:
    # a single matrix product beats a Python graph walk at that size.
    exact_search_limit: int = 3_000_000
    # Compact in the background when a sealed segment's tombstone ratio
    # reaches compact_deleted_ratio.
    auto_compact: bool = True
    # Metadata field to index for keyword (BM25) and hybrid search, e.g. "text".
    text_field: str | None = None
    # Metadata fields with secondary indexes (equality, $in, numeric ranges).
    index_fields: list[str] | None = None

    def validate(self) -> None:
        if self.dim < 0:
            raise ValueError("dim must be positive (or 0 for a graph-only collection)")
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
        self.frozen: dict[int, GrowingSegment] = {}   # seg_id -> segment being indexed
        self._seal_threads: dict[int, threading.Thread] = {}
        self._seal_error: Exception | None = None
        self._compact_thread: threading.Thread | None = None
        self._pool = None
        self._pool_broken = False
        self._compacting: set[int] = set()
        self.text_index: BM25Index | None = BM25Index() if self.config.text_field else None
        self.meta_index: MetaIndex | None = MetaIndex(self.config.index_fields) if self.config.index_fields else None
        self._closed = False
        self._dir_lock = DirectoryLock(self.path)
        self._dir_lock.acquire()
        try:
            self._recover()
        except BaseException:
            self._dir_lock.release()
            raise

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
                self.graph = GraphStore.from_json(json.loads((self.path / self._files["graph"]).read_text()))
        self._remove_orphans()
        if self.text_index is not None or self.meta_index is not None:
            for seg in [self.growing, *self.sealed.values()]:
                for rid, _, meta in seg.records():
                    if self.text_index is not None:
                        self._index_text(rid, meta)
                    if self.meta_index is not None:
                        self.meta_index.add(rid, meta)

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
        keep = {f"seg-{sid}" for sid in self.sealed} | {"config.json", "manifest.json", "wal.log", "LOCK"}
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
                if self.text_index is not None:
                    self._index_text(rid, meta)
                if self.meta_index is not None:
                    self.meta_index.add(rid, meta)
        elif op == "delete":
            for rid in data["ids"]:
                self._remove_location(rid)
                self.graph.remove_node(rid)
                if self.text_index is not None:
                    self.text_index.remove(rid)
                if self.meta_index is not None:
                    self.meta_index.remove(rid)
        elif op == "link":
            for e in data["edges"]:
                self.graph.add(e["src"], e["dst"], e.get("rel", "related"), e.get("weight", 1.0), e.get("props"))
        elif op == "unlink":
            for e in data["edges"]:
                self.graph.remove(e["src"], e["dst"], e.get("rel"))
        elif op == "put_nodes":
            for n in data["nodes"]:
                self.graph.put_node(n["id"], n.get("props"))
        else:
            raise ValueError(f"unknown WAL op {op}")

    def _index_text(self, rid: str, meta: dict) -> None:
        value = meta
        for part in self.config.text_field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, str) and value:
            self.text_index.add(rid, value)
        else:
            self.text_index.remove(rid)

    def _segment(self, where: int):
        if where == GROWING:
            return self.growing
        frozen = self.frozen.get(where)
        if frozen is not None:
            return frozen
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
    def _require_vectors(self) -> None:
        if self.config.dim == 0:
            raise ValueError(f"collection {self.name!r} is graph-only (dim=0); use put_nodes / link")

    def upsert(self, ids: list[str], vectors, metadata: list[dict] | None = None) -> int:
        self._require_vectors()
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
        with self._lock:
            self._check_open()
            full = self._write_upsert(ids, vecs, metadata)
        self._after_upsert(full)
        return len(ids)

    def _write_upsert(self, ids: list[str], vecs: np.ndarray, metadata: list[dict] | None) -> bool:
        """Log and apply one batch (caller holds the lock). Returns whether a seal is due."""
        # One WAL entry per batch, so a crash keeps all of it or none of it.
        data = {"ids": ids, "metadata": metadata}
        self.wal.append("upsert", data, vecs)
        self._apply("upsert", data, vecs)
        full = self._growing_full
        if not full:
            self._maybe_checkpoint()
        return full

    def _after_upsert(self, full: bool) -> None:
        # A big batch seals into threshold-sized segments, one after another.
        # Sealing runs outside the lock: it may wait for a previous build.
        while full:
            self.seal(only_if_full=True)
            with self._lock:
                full = not self._closed and self._growing_full
        self._maybe_auto_compact()  # overwrites tombstone sealed records too

    def update_metadata(self, ids: Iterable[str], patch: dict, replace: bool = False) -> int:
        """Change metadata without resending vectors.

        Merges ``patch`` into each record's metadata (a ``None`` value removes
        that key), or with ``replace=True`` makes ``patch`` the whole metadata.
        Ids that don't exist are skipped. Returns how many records changed.

        Implemented as an upsert of the stored vectors, so it is atomic, logged
        and indexed exactly like any other write.
        """
        if not isinstance(patch, dict):
            raise ValueError("patch must be an object")
        self._require_vectors()
        with self._lock:
            self._check_open()
            new_ids, vecs, metas = [], [], []
            for rid in dict.fromkeys(str(r) for r in ids):
                located = self._locate(rid)
                if located is None:
                    continue
                vec, meta = located
                if replace:
                    merged = {k: v for k, v in patch.items() if v is not None}
                else:
                    merged = {**meta, **patch}
                    merged = {k: v for k, v in merged.items() if v is not None}
                new_ids.append(rid)
                vecs.append(vec.copy())
                metas.append(merged)
            if not new_ids:
                return 0
            full = self._write_upsert(new_ids, np.stack(vecs), metas)
        self._after_upsert(full)
        return len(new_ids)

    def delete_where(self, filter: dict) -> int:
        """Delete every record matching ``filter`` (required: no accidental wipe)."""
        if not filter:
            raise ValueError("delete_where needs a non-empty filter")
        ids = [r["id"] for r in self.scan(filter)]
        return self.delete(ids) if ids else 0

    def count(self, filter: dict | None = None) -> int:
        if not filter:
            with self._lock:
                return len(self.locator)
        return len(self.scan(filter))

    def delete(self, ids: Iterable[str]) -> int:
        ids = [str(i) for i in ids]
        with self._lock:
            self._check_open()
            present = [rid for rid in ids if rid in self.locator or self.graph.has_node(rid)]
            # Edges can exist for ids that were never inserted, so log those too.
            targets = [rid for rid in ids
                       if rid in self.locator or self.graph.degree(rid) or self.graph.has_node(rid)]
            if targets:
                self.wal.append("delete", {"ids": targets})
                self._apply("delete", {"ids": targets}, None)
                self._maybe_checkpoint()
        self._maybe_auto_compact()
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

    def put_nodes(self, nodes: list[dict]) -> int:
        """Set node properties: ``[{"id": ..., "props": {...}}]`` (replaces existing props)."""
        clean = []
        for n in nodes:
            if "id" not in n:
                raise ValueError("each node needs an id")
            props = n.get("props") or {}
            if not isinstance(props, dict):
                raise ValueError("node props must be an object")
            clean.append({"id": str(n["id"]), "props": props})
        with self._lock:
            self._check_open()
            if clean:
                self.wal.append("put_nodes", {"nodes": clean})
                self._apply("put_nodes", {"nodes": clean}, None)
                self._maybe_checkpoint()
        return len(clean)

    def get_nodes(self, ids: Iterable[str]) -> dict[str, dict]:
        with self._lock:
            return {str(i): dict(p) for i in ids if (p := self.graph.get_node(str(i))) is not None}

    def node_ids(self) -> list[str]:
        with self._lock:
            return self.graph.node_ids()

    def list_nodes(self, limit: int = 100, offset: int = 0, prefix: str | None = None) -> dict:
        """Page through nodes (sorted by id), optionally by id prefix."""
        with self._lock:
            ids = sorted(n for n in self.graph.node_ids() if prefix is None or n.startswith(prefix))
            page = ids[offset:offset + limit]
            return {"total": len(ids), "nodes": [{"id": n, "props": dict(self.graph.get_node(n)),
                                                  "degree": len(self.graph.neighbor_ids(n))} for n in page]}

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

    def scan(self, filter: dict | None = None, limit: int | None = None, offset: int = 0) -> list[dict]:
        """Records matching ``filter``, in id order; page with ``offset`` / ``limit``."""
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("offset and limit must be non-negative")
        pred = compile_filter(filter)
        with self._lock:
            cands, exact = (self.meta_index.plan(filter) if self.meta_index is not None and pred
                            else (None, False))
            if cands is not None:
                matched = []
                for rid in cands:
                    found = self._locate(rid)
                    if found is not None and (exact or pred(found[1])):
                        matched.append((rid, found[1]))
            else:
                segments = [self.growing, *self.frozen.values(),
                            *self.sealed.values()]
                matched = [(rid, meta) for seg in segments for rid, _, meta in seg.records()
                           if pred is None or pred(meta)]
        matched.sort(key=lambda rm: rm[0])
        end = None if limit is None else offset + limit
        return [{"id": rid, "metadata": meta} for rid, meta in matched[offset:end]]

    def search(
        self,
        vector=None,
        k: int = 10,
        filter: dict | None = None,
        ef: int | None = None,
        include_vectors: bool = False,
        expand_hops: int = 0,
        graph_boost: float = 0.0,
        text: str | None = None,
    ) -> list[dict]:
        """Vector, keyword or hybrid search across all segments.

        ``vector``       query embedding (nearest-neighbour search)
        ``text``         keyword query against ``text_field`` (BM25); with a
                         vector too, the two rankings are merged by reciprocal
                         rank fusion
        ``filter``       metadata filter (see ``stratadb.filters``)
        ``expand_hops``  attach graph neighbours (up to N hops) to each hit
        ``graph_boost``  0..1; re-rank with spreading activation over edges, so
                         records strongly linked to many good matches move up
                         and linked records outside the vector top-k can enter
        """
        if k <= 0:
            return []
        if vector is not None:
            self._require_vectors()
        if vector is None and text is None:
            raise ValueError("search needs a vector, text, or both")
        if text is not None and self.text_index is None:
            raise ValueError(f"collection {self.name!r} has no text_field; keyword search is unavailable")
        if not 0.0 <= graph_boost <= 1.0:
            raise ValueError("graph_boost must be between 0 and 1")
        if graph_boost > 0 and vector is None:
            raise ValueError("graph_boost needs a query vector")
        q = None
        if vector is not None:
            q = prepare(np.asarray(vector, dtype=np.float32).reshape(-1), self.config.metric)
            if q.shape[0] != self.config.dim:
                raise ValueError(f"expected dim {self.config.dim}, got {q.shape[0]}")
        pred = compile_filter(filter)
        ef = ef or self.config.ef_search
        hybrid = q is not None and text is not None
        pool_k = max(k * 4, 32) if (graph_boost > 0 or hybrid) else k

        with self._lock:
            self._check_open()
            vec_info: dict[str, dict] = {}
            vec_rank: list[str] = []
            cands, exact = (self.meta_index.plan(filter) if self.meta_index is not None and pred
                            else (None, False))
            seg_pred = None if exact else pred  # exact index answers need no re-check
            if q is not None:
                hits = self.growing.search(q, pool_k, seg_pred, cands)
                for frozen in self.frozen.values():
                    hits.extend(frozen.search(q, pool_k, seg_pred, cands))
                for seg in self.sealed.values():
                    hits.extend(seg.search(q, pool_k, ef=max(ef, pool_k), pred=seg_pred,
                                           exact_limit=self.config.exact_search_limit, candidates=cands))
                hits.sort()
                scored = [(similarity(d, self.config.metric), d, rid) for d, rid in hits[:pool_k]]
                if graph_boost > 0 and scored:
                    scored = self._graph_rerank(q, scored, graph_boost, pred)
                for item in scored:
                    info = {"score": item[0], "distance": item[1]}
                    if graph_boost > 0:
                        info["vector_score"], info["graph_score"] = item[3], item[4]
                    vec_info[item[2]] = info
                    vec_rank.append(item[2])

            kw_scores: dict[str, float] = {}
            kw_rank: list[str] = []
            if text is not None:
                def allow(rid: str) -> bool:
                    found = self._locate(rid)
                    return found is not None and (pred is None or pred(found[1]))

                for score, rid in self.text_index.search(text, pool_k, allow=allow):
                    kw_scores[rid] = score
                    kw_rank.append(rid)

            if hybrid:
                order = fuse_rrf([vec_rank, kw_rank], k)
            elif q is not None:
                order = [(vec_info[rid]["score"], rid) for rid in vec_rank[:k]]
            else:
                order = [(kw_scores[rid], rid) for rid in kw_rank[:k]]

            results = []
            for score, rid in order:
                vec, meta = self._locate(rid)
                res: dict[str, Any] = {"id": rid, "score": score, "metadata": meta}
                if q is not None and not hybrid:
                    res.update(vec_info[rid])
                if hybrid:
                    if rid in vec_info:
                        res["vector_score"] = vec_info[rid].get("vector_score", vec_info[rid]["score"])
                    res["keyword_score"] = kw_scores.get(rid, 0.0)
                elif text is not None:
                    res["keyword_score"] = score
                if include_vectors:
                    res["vector"] = vec.tolist()
                if expand_hops > 0:
                    res["neighbors"] = self.graph.neighbors(rid, hops=expand_hops)
                results.append(res)
            return results

    def search_many(self, vectors, k: int = 10, **kwargs) -> list[list[dict]]:
        """Run one search per row of ``vectors`` (same options for all).

        Everything the single-query search does, including filters, hybrid
        text (one string for all rows) and graph_boost, applies to each row.
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.ndim != 2:
            raise ValueError("vectors must be a 2-D array of query vectors")
        return [self.search(row, k, **kwargs) for row in vecs]

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

    def shortest_path(self, src: str, dst: str, max_hops: int = 6, rel: str | None = None,
                      direction: str = "both") -> list[dict] | None:
        with self._lock:
            return self.graph.shortest_path(str(src), str(dst), max_hops=max_hops, rel=rel, direction=direction)

    def pagerank(self, top: int | None = None, seeds: dict[str, float] | list[str] | None = None,
                 damping: float = 0.85, directed: bool = True) -> list[tuple[str, float]]:
        """Nodes ranked by (personalized, if ``seeds``) PageRank, highest first."""
        if isinstance(seeds, (list, tuple)):
            seeds = {str(s): 1.0 for s in seeds}
        with self._lock:
            scores = self.graph.pagerank(damping=damping, personalization=seeds, directed=directed)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top] if top is not None else ranked

    def components(self) -> list[list[str]]:
        with self._lock:
            return self.graph.components()

    def edges(self, rid: str, direction: str = "both", rel: str | None = None) -> list[dict]:
        with self._lock:
            return self.graph.edges(str(rid), direction=direction, rel=rel)

    # ---------------------------------------------------------- maintenance
    def seal(self, wait: bool | None = None, only_if_full: bool = False) -> int | None:
        """Freeze the growing segment and build its HNSW index.

        With ``background_seal`` (the default) the index builds on a worker
        thread: writes go to a fresh growing segment and the frozen one is still
        searched by brute force until the indexed segment is swapped in. Up to
        ``build_processes`` builds run at once (each in its own worker process);
        without worker processes, one at a time.
        """
        background = self.config.background_seal if wait is None else not wait
        self._check_open()
        while True:
            with self._lock:
                self._check_open()
                if not len(self.growing) or (only_if_full and not self._growing_full):
                    return None
                # Slot check and freeze happen under one lock hold, so the
                # concurrency limit is exact and no thread registration is lost.
                busy = self._live_seal_threads()
                if len(busy) < self._max_concurrent_builds():
                    return self._freeze_and_build(background)
            busy[0].join()

    def _freeze_and_build(self, background: bool) -> int:
        """Caller holds the lock. Freeze the growing segment and start its build."""
        seg_id = self._next_seg
        self._next_seg += 1
        frozen = self.growing
        self.growing = GrowingSegment(self.config.dim, self.config.metric)
        if len(frozen) > self.config.seal_threshold:
            # Seal one threshold-sized chunk; the rest stays in growing.
            chunk, rest = GrowingSegment(self.config.dim, self.config.metric), self.growing
            for i, (rid, vec, meta) in enumerate(frozen.records()):
                (chunk if i < self.config.seal_threshold else rest).put(rid, vec, meta)
            frozen = chunk
        self.frozen[seg_id] = frozen
        for rid in frozen.pos:
            self.locator[rid] = seg_id
        records = [(rid, vec.copy(), meta) for rid, vec, meta in frozen.records()]
        if background:
            thread = threading.Thread(
                target=self._build_and_install, args=(seg_id, frozen, records),
                name=f"stratadb-seal-{self.name}-{seg_id}", daemon=True,
            )
            self._seal_threads[seg_id] = thread
            thread.start()
            return seg_id
        # Synchronous build: release the lock while building, like a thread would.
        self._lock.release()
        try:
            self._build_and_install(seg_id, frozen, records)
        finally:
            self._lock.acquire()
        return seg_id

    def _build_segment(self, seg_id: int, records) -> SealedSegment:
        """Build a sealed segment, in a worker process when configured."""
        args = (seg_id, self.path / f"seg-{seg_id}", records,
                self.config.metric, self.config.m, self.config.ef_construction)
        pool = self._executor() if len(records) >= self.config.build_process_min_records else None
        if pool is not None:
            try:
                return SealedSegment.build_in_process(*args, pool)
            except Exception:  # noqa: BLE001 - worker died (e.g. spawn can't re-import __main__)
                log.warning("worker-process build failed for %s; building in-process instead",
                            self.name, exc_info=True)
                self._pool = None
                self._pool_broken = True
        return SealedSegment.build(*args)

    def _executor(self):
        if self.config.build_processes <= 0 or self._pool_broken:
            return None
        if self._pool is None:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            # spawn, never fork: this process runs background threads holding
            # locks, and a fork would copy those locks in a held state and
            # deadlock the worker. spawn re-imports __main__, which fails under
            # a REPL, notebook or `python -`; that case falls back to
            # in-process builds (see _build_segment).
            self._pool = ProcessPoolExecutor(max_workers=self.config.build_processes,
                                             mp_context=multiprocessing.get_context("spawn"))
        return self._pool

    def _build_and_install(self, seg_id: int, frozen: GrowingSegment, records) -> None:
        try:
            segment = self._build_segment(seg_id, records)
        except Exception as e:  # noqa: BLE001 - keep the data searchable, surface the error
            log.exception("sealing segment %s of %s failed", seg_id, self.name)
            with self._lock:
                self._seal_error = e
                for rid, vec, meta in frozen.records():
                    if rid not in self.growing.pos:
                        self.growing.put(rid, vec, meta)
                        self.locator[rid] = GROWING
                self.frozen.pop(seg_id, None)
            return
        with self._lock:
            # Records deleted or overwritten while the index was building left
            # the frozen segment; tombstone them in the new one.
            for rid in list(segment.pos):
                if rid not in frozen.pos:
                    segment.delete(rid)
            # Live records already locate to seg_id; it now resolves to the
            # sealed segment instead of the frozen one.
            self.sealed[seg_id] = segment
            self.frozen.pop(seg_id, None)
            self._seal_error = None
            if not self._closed:
                self.checkpoint()

    def _live_seal_threads(self) -> list:
        """Prune finished build threads in place (caller holds the lock) and
        return the ones still running, excluding the calling thread."""
        me = threading.current_thread()
        for sid in [sid for sid, t in self._seal_threads.items() if not t.is_alive()]:
            del self._seal_threads[sid]
        return [t for t in self._seal_threads.values() if t is not me]

    def wait_for_seal(self) -> None:
        """Block until every background seal has finished."""
        while True:
            with self._lock:
                busy = self._live_seal_threads()
            if not busy:
                return
            for t in busy:
                t.join()

    def _max_concurrent_builds(self) -> int:
        if self.config.build_processes > 0 and not self._pool_broken:
            return self.config.build_processes
        return 1  # in-process builds share the GIL; running several gains nothing

    def compact(self, wait: bool = True) -> dict:
        """Rewrite sealed segments that are mostly tombstones, and merge small ones.

        The live records are snapshotted under the lock, the new segments are
        built outside it (reads and writes keep going), and on install any
        record deleted or overwritten during the build is tombstoned in the new
        segment. With ``wait=False`` the build runs on a worker thread.
        """
        self._check_open()
        self.wait_for_compaction()
        with self._lock:
            self._check_open()
            threshold = self.config.seal_threshold
            victims = [
                s for s in self.sealed.values()
                if s.deleted_ratio >= self.config.compact_deleted_ratio or len(s) < threshold // 2
            ]
            if not victims or (len(victims) == 1 and victims[0].deleted_ratio == 0):
                return {"rewritten": 0, "segments": len(self.sealed)}
            victim_ids = [s.seg_id for s in victims]
            records = [(rid, vec.copy(), meta) for s in victims for rid, vec, meta in s.records()]
            chunks = []
            for start in range(0, len(records), threshold):
                chunks.append((self._next_seg, records[start:start + threshold]))
                self._next_seg += 1
            self._compacting = set(victim_ids)
            if not wait:
                self._compact_thread = threading.Thread(
                    target=self._compact_build, args=(victim_ids, chunks),
                    name=f"stratadb-compact-{self.name}", daemon=True,
                )
                self._compact_thread.start()
                return {"started": True, "rewriting": victim_ids}
        return self._compact_build(victim_ids, chunks)

    def _compact_build(self, victim_ids: list[int], chunks) -> dict:
        try:
            built = [self._build_segment(seg_id, chunk) for seg_id, chunk in chunks]
        except Exception as e:  # noqa: BLE001 - old segments stay in place; nothing is lost
            log.exception("compacting %s failed", self.name)
            with self._lock:
                self._compacting = set()
            return {"rewritten": 0, "error": str(e)}
        victim_set = set(victim_ids)
        with self._lock:
            if self._closed:
                return {"rewritten": 0, "error": "closed"}
            for segment in built:
                for rid in list(segment.pos):
                    if self.locator.get(rid) in victim_set:
                        self.locator[rid] = segment.seg_id
                    else:  # deleted or overwritten while the new segment was building
                        segment.delete(rid)
                self.sealed[segment.seg_id] = segment
            victims = [self.sealed.pop(v) for v in victim_ids if v in self.sealed]
            self._compacting = set()
            self.checkpoint()
            result = {"rewritten": len(victims), "new_segments": [s.seg_id for s in built],
                      "segments": len(self.sealed)}
        for v in victims:
            shutil.rmtree(v.path, ignore_errors=True)
        return result

    def wait_for_compaction(self) -> None:
        thread = self._compact_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join()

    def _maybe_auto_compact(self) -> None:
        if not self.config.auto_compact or self._closed:
            return
        thread = self._compact_thread
        if thread is not None and thread.is_alive():
            return
        with self._lock:
            due = any(s.deleted_ratio >= self.config.compact_deleted_ratio and len(s.ids) >= 64
                      for s in self.sealed.values())
        if due:
            self.compact(wait=False)

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
            records = [r for frozen in self.frozen.values() for r in frozen.records()]
            records += list(self.growing.records())
            save_records(records, self.config.dim, self.path / files["growing"][0], self.path / files["growing"][1])
            atomic_write_json(self.path / files["graph"], self.graph.to_json())
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

    def export_jsonl(self, fp, batch: int = 1000) -> dict:
        """Write every record, node and edge as one JSON object per line.

        Lines are ``{"type": "record", "id", "vector", "metadata"}``,
        ``{"type": "node", "id", "props"}`` and ``{"type": "edge", "src", "dst",
        "rel", "weight", "props"}``, preceded by a ``{"type": "collection", ...}``
        header with dim and metric. Vectors are written as stored (unit length
        for cosine collections). Version-independent, unlike ``backup``.
        """
        counts = {"records": 0, "nodes": 0, "edges": 0}
        with self._lock:
            self._check_open()
            fp.write(json.dumps({"type": "collection", "name": self.name, "dim": self.config.dim,
                                 "metric": self.config.metric}) + "\n")
            segments = [self.growing, *self.frozen.values(), *self.sealed.values()]
            for seg in segments:
                for rid, vec, meta in seg.records():
                    fp.write(json.dumps({"type": "record", "id": rid, "vector": [float(x) for x in vec],
                                         "metadata": meta}, ensure_ascii=False) + "\n")
                    counts["records"] += 1
            for nid in self.graph.node_ids():
                fp.write(json.dumps({"type": "node", "id": nid, "props": self.graph.get_node(nid)},
                                    ensure_ascii=False) + "\n")
                counts["nodes"] += 1
            for src, rel, dst, w, props in self.graph.all_edges():
                fp.write(json.dumps({"type": "edge", "src": src, "dst": dst, "rel": rel, "weight": w,
                                     "props": props}, ensure_ascii=False) + "\n")
                counts["edges"] += 1
        return counts

    def import_jsonl(self, fp, batch: int = 1000) -> dict:
        """Load lines written by ``export_jsonl`` (upserting; existing ids are
        overwritten). The header's dim and metric must match this collection."""
        counts = {"records": 0, "nodes": 0, "edges": 0}
        ids, vecs, metas, nodes, edges = [], [], [], [], []

        def flush():
            if ids:
                self.upsert(list(ids), np.asarray(vecs, dtype=np.float32), list(metas))
                ids.clear(); vecs.clear(); metas.clear()
            if nodes:
                self.put_nodes(list(nodes)); nodes.clear()
            if edges:
                self.link(list(edges)); edges.clear()

        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "collection":
                if obj.get("dim") != self.config.dim or obj.get("metric") != self.config.metric:
                    raise ValueError(f"export is dim={obj.get('dim')} metric={obj.get('metric')!r}; "
                                     f"this collection is dim={self.config.dim} metric={self.config.metric!r}")
            elif kind == "record":
                ids.append(obj["id"]); vecs.append(obj["vector"]); metas.append(obj.get("metadata") or {})
                counts["records"] += 1
            elif kind == "node":
                nodes.append({"id": obj["id"], "props": obj.get("props") or {}}); counts["nodes"] += 1
            elif kind == "edge":
                edges.append({k: obj[k] for k in ("src", "dst", "rel", "weight", "props") if k in obj})
                counts["edges"] += 1
            else:
                raise ValueError(f"unknown line type {kind!r}")
            if len(ids) >= batch or len(nodes) >= batch or len(edges) >= batch:
                flush()
        flush()
        return counts

    def backup(self, dest: str | Path) -> Path:
        """Write a consistent copy of this collection to ``dest`` (a new directory).

        Checkpoints first, then copies exactly the files the manifest references
        while holding the lock, so the copy is a point-in-time snapshot that
        opens like any collection. Writes resume as soon as the copy finishes.
        """
        dest = Path(dest)
        if dest.exists():
            raise FileExistsError(f"{dest} already exists")
        self.wait_for_seal()
        self.wait_for_compaction()
        with self._lock:
            self._check_open()
            self.checkpoint()
            tmp = dest.with_name(dest.name + ".partial")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            for name in ("config.json", "manifest.json"):
                shutil.copy2(self.path / name, tmp / name)
            for value in self._files.values():
                for name in [value] if isinstance(value, str) else value:
                    shutil.copy2(self.path / name, tmp / name)
            for sid in self.sealed:
                shutil.copytree(self.path / f"seg-{sid}", tmp / f"seg-{sid}")
            (tmp / "wal.log").touch()
        tmp.rename(dest)
        return dest

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "count": len(self.locator),
                "dim": self.config.dim,
                "metric": self.config.metric,
                "growing": len(self.growing),
                "sealing": sum(len(f) for f in self.frozen.values()),
                "sealed_segments": [
                    {"id": sid, "live": len(s), "total": s.total, "deleted_ratio": round(s.deleted_ratio, 3)}
                    for sid, s in sorted(self.sealed.items())
                ],
                "edges": len(self.graph),
                "nodes": self.graph.node_count,
                "wal_bytes": self.wal.size_bytes(),
                "checkpoint_lsn": self._checkpoint_lsn,
                "last_lsn": self.wal.last_lsn,
            }

    def close(self) -> None:
        self.wait_for_seal()
        self.wait_for_compaction()
        with self._lock:
            if self._closed:
                return
            if self.wal.last_lsn > self._checkpoint_lsn:
                self.checkpoint()
            self.wal.close()
            self._closed = True
            self._dir_lock.release()
        if self._pool is not None:
            # Wait for the idle workers to exit: with wait=False they outlive
            # this process briefly and keep inherited pipes open.
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
