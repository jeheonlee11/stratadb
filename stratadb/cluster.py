"""Router for a StrataDB cluster: shards, replicas, fan-out search.

A cluster is a list of shards; each shard is a list of ordinary ``stratadb
serve`` nodes holding the same data (the first is the primary, the rest are
replicas). The router speaks the single-node HTTP API, so ``Client`` and the
console work against it unchanged::

    {"shards": [{"nodes": ["http://a:7701", "http://b:7702"]},
                {"nodes": ["http://c:7703", "http://d:7704"]}],
     "api_key": "nodes-key"}

    stratadb cluster cluster.json --port 7700

**Placement.** A collection is created on every node. With ``sharding:
"hash"`` (the default) a record lives on shard ``crc32(id) % shards``; with
``sharding: "none"`` the whole collection lives on shard 0 (replicated), which
is the mode for collections that use the graph features across all their
records (see below).

**Writes** go to every node of the shard at once and are acknowledged when
the primary has them; a replica that fails a write is marked *stale* and is
re-synced from the primary (export -> import) by the health loop once it
answers again, or on ``POST /cluster/resync``. **Reads** go to the primary,
or to any in-sync replica when the primary is down.

**Search** fans out to every shard, takes each shard's top k, and merges by
distance (or fused score). Filters, hybrid search and ``search_many`` work the
same way. ``expand_hops`` and ``graph_boost`` see the edges stored on each
shard: an edge is stored on the shards of both its endpoints, so one-hop
expansion is complete, but activation does not cross shards. Whole-graph
algorithms (``path``, ``pagerank``, ``components``) need ``sharding: "none"``.

Everything here is standard library: one ``ThreadPoolExecutor`` for fan-out,
``urllib`` for node requests.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .server import MAX_BODY, ApiError

log = logging.getLogger("stratadb.cluster")

HEALTH_INTERVAL = 2.0
NODE_TIMEOUT = 60.0


class NodeError(Exception):
    def __init__(self, url: str, status: int, message: str):
        super().__init__(f"{url}: {status}: {message}")
        self.url, self.status, self.message = url, status, message


class Node:
    """One ``stratadb serve`` process and what the router knows about it."""

    def __init__(self, url: str, api_key: str | None):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.up = True
        self.stale = False      # missed a write; needs a resync before serving reads
        self.last_error: str | None = None

    def request(self, method: str, path: str, body: dict | None = None, raw: bytes | None = None,
                timeout: float = NODE_TIMEOUT, stream: bool = False):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Content-Type", "application/x-ndjson" if raw is not None else "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", e.reason)
            except (json.JSONDecodeError, AttributeError):
                message = e.reason
            raise NodeError(self.url, e.code, message) from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            self.up = False
            self.last_error = str(e)
            raise NodeError(self.url, 503, f"node unreachable: {e}") from None
        self.up = True
        if stream:
            return resp
        with resp:
            return json.loads(resp.read())

    def state(self) -> dict:
        return {"url": self.url, "up": self.up, "stale": self.stale, "error": self.last_error}


class Shard:
    def __init__(self, index: int, nodes: list[Node]):
        self.index = index
        self.nodes = nodes

    @property
    def primary(self) -> Node:
        return self.nodes[0]

    def reader(self) -> Node:
        """A node to read from: the primary, else any in-sync replica."""
        for node in self.nodes:
            if node.up and not node.stale:
                return node
        raise ApiError(503, f"shard {self.index}: no node available")

    def writers(self) -> list[Node]:
        return [n for n in self.nodes if n.up]


class Router:
    def __init__(self, config: dict, state_path: Path | None = None):
        key = config.get("api_key")
        self.shards = [Shard(i, [Node(u, key) for u in s["nodes"]]) for i, s in enumerate(config["shards"])]
        if not self.shards or any(not s.nodes for s in self.shards):
            raise ValueError("cluster config needs at least one shard with at least one node")
        self.state_path = state_path
        self.collections: dict[str, dict] = {}   # name -> {"sharding": "hash" | "none"}
        if state_path and state_path.exists():
            self.collections = json.loads(state_path.read_text())
        workers = max(4, 2 * sum(len(s.nodes) for s in self.shards))
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stratadb-router")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._health_thread: threading.Thread | None = None

    # ------------------------------------------------------------- lifecycle
    def start_health_loop(self, interval: float = HEALTH_INTERVAL) -> None:
        def loop():
            while not self._stop.wait(interval):
                try:
                    self.check_health()
                except Exception:  # noqa: BLE001 - keep the loop alive
                    log.exception("health check failed")
        self._health_thread = threading.Thread(target=loop, daemon=True, name="stratadb-router-health")
        self._health_thread.start()

    def close(self) -> None:
        self._stop.set()
        self.pool.shutdown(wait=False)

    def _save_state(self) -> None:
        if self.state_path:
            tmp = self.state_path.with_name(self.state_path.name + ".tmp")
            tmp.write_text(json.dumps(self.collections, indent=2))
            tmp.replace(self.state_path)

    # ---------------------------------------------------------------- health
    def check_health(self) -> dict:
        """Probe every node; resync replicas that came back stale."""
        def probe(node: Node):
            try:
                node.request("GET", "/health", timeout=5)
            except NodeError:
                pass
        list(self.pool.map(probe, [n for s in self.shards for n in s.nodes]))
        for shard in self.shards:
            for node in shard.nodes[1:]:
                if node.up and node.stale and shard.primary.up:
                    self.resync_node(shard, node)
        return self.status()

    def status(self) -> dict:
        return {
            "version": __version__,
            "shards": [{"index": s.index, "nodes": [n.state() for n in s.nodes]} for s in self.shards],
            "collections": self.collections,
        }

    def resync_node(self, shard: Shard, node: Node) -> dict:
        """Copy every collection from the shard's primary to ``node``."""
        copied = {}
        for name in list(self.collections):
            try:
                # Start from empty so deletes the replica missed are not kept.
                try:
                    node.request("DELETE", f"/collections/{name}")
                except NodeError as e:
                    if e.status != 404:
                        raise
                self._ensure_collection_on(node, name)
                resp = shard.primary.request("GET", f"/collections/{name}/export", stream=True)
                with resp:
                    data = resp.read()
                copied[name] = node.request("POST", f"/collections/{name}/import", raw=data)["imported"]
            except NodeError as e:
                log.warning("resync of %s to %s failed: %s", name, node.url, e)
                return {"resynced": False, "error": str(e)}
        node.stale = False
        node.last_error = None
        log.warning("resynced %s from %s: %s", node.url, shard.primary.url, copied)
        return {"resynced": True, "collections": copied}

    def _ensure_collection_on(self, node: Node, name: str) -> None:
        spec = self.collections[name].get("create")
        if spec:
            try:
                node.request("POST", "/collections", spec)
            except NodeError as e:
                if e.status != 409:
                    raise

    # --------------------------------------------------------------- routing
    def _shard_of(self, rid: str, sharding: str) -> Shard:
        if sharding == "none" or len(self.shards) == 1:
            return self.shards[0]
        return self.shards[zlib.crc32(str(rid).encode()) % len(self.shards)]

    def _partition(self, ids: list, sharding: str) -> dict[int, list[int]]:
        """shard index -> positions in ``ids``."""
        out: dict[int, list[int]] = {}
        for pos, rid in enumerate(ids):
            out.setdefault(self._shard_of(rid, sharding).index, []).append(pos)
        return out

    def _write(self, shard: Shard, path: str, body: dict) -> Any:
        """Send a write to every live node of the shard; the primary's answer wins."""
        nodes = shard.writers()
        if not nodes or not shard.primary.up:
            # Without a primary we would have to promote a replica; refuse
            # instead so no two nodes diverge.
            raise ApiError(503, f"shard {shard.index}: primary {shard.primary.url} is down; writes refused")
        results = list(self.pool.map(lambda n: self._try_write(n, path, body), nodes))
        primary_result = results[0]
        if isinstance(primary_result, NodeError):
            raise ApiError(primary_result.status, primary_result.message)
        for node, result in zip(nodes[1:], results[1:]):
            if isinstance(result, NodeError):
                node.stale = True
                log.warning("replica %s missed a write (%s); marked stale", node.url, result)
        return primary_result

    @staticmethod
    def _try_write(node: Node, path: str, body: dict):
        try:
            return node.request("POST", path, body)
        except NodeError as e:
            return e

    def _read(self, shard: Shard, method: str, path: str, body: dict | None = None) -> Any:
        node = shard.reader()
        try:
            return node.request(method, path, body)
        except NodeError as e:
            if e.status == 503 and node is not shard.primary or e.status == 503:
                # Try the next in-sync node before giving up.
                for other in shard.nodes:
                    if other is not node and other.up and not other.stale:
                        try:
                            return other.request(method, path, body)
                        except NodeError:
                            continue
            raise ApiError(e.status, e.message)

    def _fanout(self, fn, shards=None) -> list:
        shards = self.shards if shards is None else shards
        results = list(self.pool.map(lambda s: _capture(fn, s), shards))
        for r in results:
            if isinstance(r, ApiError):
                raise r
        return results

    # ----------------------------------------------------------- collections
    def _sharding(self, name: str) -> str:
        if name not in self.collections:
            raise ApiError(404, f"no collection named {name!r}")
        return self.collections[name].get("sharding", "hash")

    def create_collection(self, body: dict) -> dict:
        name = body.get("name")
        if not name:
            raise ApiError(400, "name is required")
        sharding = body.pop("sharding", "hash")
        if sharding not in ("hash", "none"):
            raise ApiError(400, "sharding must be 'hash' or 'none'")
        with self._lock:
            if name in self.collections:
                raise ApiError(409, f"collection {name!r} exists")
            targets = [n for s in self.shards for n in s.nodes if n.up]
            errors = []
            for node in targets:
                try:
                    node.request("POST", "/collections", body)
                except NodeError as e:
                    if e.status != 409:
                        errors.append(str(e))
            if errors:
                raise ApiError(502, "could not create on every node: " + "; ".join(errors))
            self.collections[name] = {"sharding": sharding, "create": body}
            self._save_state()
        return {"created": name, "sharding": sharding}

    def drop_collection(self, name: str) -> dict:
        self._sharding(name)
        with self._lock:
            for node in [n for s in self.shards for n in s.nodes if n.up]:
                try:
                    node.request("DELETE", f"/collections/{name}")
                except NodeError as e:
                    if e.status != 404:
                        raise ApiError(e.status, e.message)
            del self.collections[name]
            self._save_state()
        return {"dropped": True}

    def stats(self, name: str) -> dict:
        sharding = self._sharding(name)
        shards = self.shards if sharding == "hash" else self.shards[:1]
        per = self._fanout(lambda s: self._read(s, "GET", f"/collections/{name}"), shards)
        total = {"name": name, "count": sum(p["count"] for p in per), "edges": sum(p["edges"] for p in per),
                 "nodes": sum(p["nodes"] for p in per), "dim": per[0]["dim"], "metric": per[0]["metric"],
                 "sharding": sharding, "shards": per}
        return total

    # ---------------------------------------------------------------- writes
    def _scatter(self, name: str, body: dict, action: str, list_keys: tuple, count_key: str) -> dict:
        """Split a batch write by the shard of each id and sum the counts."""
        sharding = self._sharding(name)
        ids = body[list_keys[0]]
        if sharding == "none" or len(self.shards) == 1:
            return self._write(self.shards[0], f"/collections/{name}/{action}", body)
        parts = self._partition(ids, sharding)

        def one(item):
            index, positions = item
            sub = dict(body)
            for key in list_keys:
                if key in body and body[key] is not None:
                    sub[key] = [body[key][p] for p in positions]
            return self._write(self.shards[index], f"/collections/{name}/{action}", sub)

        results = list(self.pool.map(lambda it: _capture(one, it), parts.items()))
        for r in results:
            if isinstance(r, ApiError):
                raise r
        return {count_key: sum(r[count_key] for r in results)}

    def link(self, name: str, body: dict, action: str) -> dict:
        """Edges live on the shard of src and the shard of dst."""
        sharding = self._sharding(name)
        edges = body["edges"]
        if sharding == "none" or len(self.shards) == 1:
            return self._write(self.shards[0], f"/collections/{name}/{action}", body)
        by_shard: dict[int, list] = {}
        for e in edges:
            for rid in {e.get("src"), e.get("dst")}:
                by_shard.setdefault(self._shard_of(rid, sharding).index, []).append(e)
        key = "linked" if action == "link" else "unlinked"
        results = self._fanout(lambda s: self._write(s, f"/collections/{name}/{action}", {"edges": by_shard[s.index]}),
                               [self.shards[i] for i in by_shard])
        # An edge whose endpoints share a shard is counted once; otherwise it is
        # stored twice, so report the number of edges the caller sent.
        return {key: min(len(edges), sum(r[key] for r in results))}

    def broadcast(self, name: str, action: str, body: dict) -> dict:
        """Maintenance on every node holding the collection."""
        self._sharding(name)
        out = {}
        for shard in self.shards:
            for node in shard.nodes:
                if node.up:
                    try:
                        out[node.url] = node.request("POST", f"/collections/{name}/{action}", body)
                    except NodeError as e:
                        out[node.url] = {"error": e.message}
        return out

    # ----------------------------------------------------------------- reads
    def gather_by_id(self, name: str, body: dict, action: str, result_key: str, id_key: str = "ids"):
        """Route an id list to the shards holding them and reassemble in order."""
        sharding = self._sharding(name)
        ids = body[id_key]
        if sharding == "none" or len(self.shards) == 1:
            return self._read(self.shards[0], "POST", f"/collections/{name}/{action}", body)
        parts = self._partition(ids, sharding)

        def one(item):
            index, positions = item
            sub = dict(body, **{id_key: [ids[p] for p in positions]})
            return self._read(self.shards[index], "POST", f"/collections/{name}/{action}", sub)[result_key]

        results = list(self.pool.map(lambda it: _capture(one, it), parts.items()))
        for r in results:
            if isinstance(r, ApiError):
                raise r
        if isinstance(results[0], dict):      # get_nodes: id -> props
            merged: dict = {}
            for r in results:
                merged.update(r)
            return {result_key: merged}
        by_id = {rec["id"]: rec for r in results for rec in r}
        return {result_key: [by_id[rid] for rid in ids if rid in by_id]}

    def search(self, name: str, body: dict) -> dict:
        sharding = self._sharding(name)
        shards = self.shards if sharding == "hash" else self.shards[:1]
        k = int(body.get("k", 10))
        per = self._fanout(lambda s: self._read(s, "POST", f"/collections/{name}/search", body)["hits"], shards)
        return {"hits": _merge_hits(per, k)}

    def search_many(self, name: str, body: dict) -> dict:
        sharding = self._sharding(name)
        shards = self.shards if sharding == "hash" else self.shards[:1]
        k = int(body.get("k", 10))
        per = self._fanout(lambda s: self._read(s, "POST", f"/collections/{name}/search_many", body)["results"],
                           shards)
        n = len(body["vectors"])
        return {"results": [_merge_hits([p[i] for p in per], k) for i in range(n)]}

    def scan(self, name: str, body: dict) -> dict:
        sharding = self._sharding(name)
        shards = self.shards if sharding == "hash" else self.shards[:1]
        offset = int(body.get("offset") or 0)
        limit = body.get("limit")
        sub = dict(body, offset=0, limit=None if limit is None else offset + int(limit))
        per = self._fanout(lambda s: self._read(s, "POST", f"/collections/{name}/scan", sub)["records"], shards)
        records = [r for p in per for r in p]
        return {"records": records[offset: None if limit is None else offset + int(limit)]}

    def sum_over_shards(self, name: str, action: str, body: dict, key: str) -> dict:
        sharding = self._sharding(name)
        shards = self.shards if sharding == "hash" else self.shards[:1]
        method = "write" if action == "delete_where" else "read"
        if method == "write":
            per = self._fanout(lambda s: self._write(s, f"/collections/{name}/{action}", body)[key], shards)
        else:
            per = self._fanout(lambda s: self._read(s, "POST", f"/collections/{name}/{action}", body)[key], shards)
        return {key: sum(per)}

    def neighbors(self, name: str, body: dict) -> dict:
        sharding = self._sharding(name)
        shard = self._shard_of(body["id"], sharding)
        result = self._read(shard, "POST", f"/collections/{name}/neighbors", body)
        if sharding == "hash" and (body.get("hydrate") or body.get("filter")):
            # Neighbours on other shards come back without records; fetch them.
            missing = [nb["id"] for nb in result["neighbors"] if nb.get("metadata") is None]
            if missing:
                fetched = {r["id"]: r for r in self.gather_by_id(name, {"ids": missing}, "get", "records")["records"]}
                for nb in result["neighbors"]:
                    rec = fetched.get(nb["id"])
                    if rec is not None:
                        nb["metadata"] = rec.get("metadata")
        return result

    def whole_graph(self, name: str, action: str, body: dict) -> dict:
        if self._sharding(name) != "none":
            raise ApiError(400, f"{action} needs the whole graph on one shard: create the collection "
                                f"with sharding='none'")
        return self._read(self.shards[0], "POST", f"/collections/{name}/{action}", body)

    def list_nodes(self, name: str, body: dict) -> dict:
        sharding = self._sharding(name)
        shards = self.shards if sharding == "hash" else self.shards[:1]
        offset = int(body.get("offset") or 0)
        limit = int(body.get("limit", 100))
        sub = dict(body, offset=0, limit=offset + limit)
        per = self._fanout(lambda s: self._read(s, "POST", f"/collections/{name}/list_nodes", sub), shards)
        nodes = [n for p in per for n in p["nodes"]]
        return {"nodes": nodes[offset: offset + limit], "total": sum(p.get("total", len(p["nodes"])) for p in per)}

    def export(self, name: str, write) -> None:
        """Stream every shard's export, keeping only the first header."""
        sharding = self._sharding(name)
        shards = self.shards if sharding == "hash" else self.shards[:1]
        first = True
        for shard in shards:
            node = shard.reader()
            resp = node.request("GET", f"/collections/{name}/export", stream=True)
            with resp:
                for line in resp:
                    text = line.decode()
                    if text.startswith('{"type": "collection"'):
                        if not first:
                            continue
                        first = False
                    write(text)

    # -------------------------------------------------------------- dispatch
    def handle(self, method: str, path: str, body: dict) -> Any:
        parts = [p for p in path.split("?")[0].split("/") if p]
        if method == "GET" and parts == ["health"]:
            up = sum(n.up for s in self.shards for n in s.nodes)
            total = sum(len(s.nodes) for s in self.shards)
            return {"status": "ok" if up == total else "degraded", "version": __version__,
                    "nodes_up": up, "nodes": total, "role": "router"}
        if method == "GET" and parts == ["cluster"]:
            return self.status()
        if method == "POST" and parts == ["cluster", "resync"]:
            out = {}
            for shard in self.shards:
                for node in shard.nodes[1:]:
                    if node.up:
                        out[node.url] = self.resync_node(shard, node)
            return out
        if method == "POST" and parts == ["cluster", "health"]:
            return self.check_health()
        if parts == ["collections"]:
            if method == "GET":
                return {"collections": sorted(self.collections)}
            if method == "POST":
                return self.create_collection(body)
        if len(parts) == 2 and parts[0] == "collections":
            if method == "GET":
                return self.stats(parts[1])
            if method == "DELETE":
                return self.drop_collection(parts[1])
        if len(parts) == 3 and parts[0] == "collections" and method == "POST":
            name, action = parts[1], parts[2]
            if action == "upsert":
                return self._scatter(name, body, action, ("ids", "vectors", "metadata"), "upserted")
            if action == "delete":
                return self._scatter(name, body, action, ("ids",), "deleted")
            if action == "update":
                return self._scatter(name, body, action, ("ids",), "updated")
            if action == "nodes":
                sharding = self._sharding(name)
                nodes = body["nodes"]
                if sharding == "none" or len(self.shards) == 1:
                    return self._write(self.shards[0], f"/collections/{name}/nodes", body)
                parts_ = self._partition([n["id"] for n in nodes], sharding)
                results = self._fanout(
                    lambda s: self._write(s, f"/collections/{name}/nodes", {"nodes": [nodes[p] for p in parts_[s.index]]}),
                    [self.shards[i] for i in parts_])
                return {"put": sum(r["put"] for r in results)}
            if action in ("get",):
                return self.gather_by_id(name, body, action, "records")
            if action == "get_nodes":
                return self.gather_by_id(name, body, action, "nodes")
            if action in ("link", "unlink"):
                return self.link(name, body, action)
            if action == "search":
                return self.search(name, body)
            if action == "search_many":
                return self.search_many(name, body)
            if action == "scan":
                return self.scan(name, body)
            if action == "count":
                return self.sum_over_shards(name, action, body, "count")
            if action == "delete_where":
                return self.sum_over_shards(name, action, body, "deleted")
            if action == "neighbors":
                return self.neighbors(name, body)
            if action == "list_nodes":
                return self.list_nodes(name, body)
            if action in ("path", "pagerank", "components"):
                return self.whole_graph(name, action, body)
            if action in ("checkpoint", "compact", "merge", "tune_ef"):
                return self.broadcast(name, action, body)
            raise ApiError(404, f"unknown action {action!r}")
        raise ApiError(404, f"no route for {method} {path}")


def _capture(fn, arg):
    try:
        return fn(arg)
    except ApiError as e:
        return e


def _merge_hits(per_shard: list[list[dict]], k: int) -> list[dict]:
    hits = [h for p in per_shard for h in p]
    if all("distance" in h for h in hits):
        hits.sort(key=lambda h: h["distance"])
    else:
        hits.sort(key=lambda h: -h.get("score", 0.0))
    return hits[:k]


# ------------------------------------------------------------------ server

def make_router_handler(router: Router, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"StrataDB-router/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if not api_key:
                return True
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied.encode(), f"Bearer {api_key}".encode())

        def _dispatch(self, method: str) -> None:
            path = self.path.split("?")[0]
            parts = [p for p in path.split("/") if p]
            if method == "GET" and path in ("/", "/console"):
                from importlib import resources
                data = resources.files("stratadb").joinpath("console.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path != "/health" and not self._authorized():
                self._send(401, {"error": "missing or wrong API key"})
                return
            if method == "GET" and len(parts) == 3 and parts[0] == "collections" and parts[2] == "export":
                try:
                    router._sharding(parts[1])
                except ApiError as e:
                    self._send(e.status, {"error": str(e)})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                out = self.wfile

                def write(text: str) -> None:
                    data = text.encode()
                    out.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                router.export(parts[1], write)
                out.write(b"0\r\n\r\n")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    raise ApiError(413, "request body too large")
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else {}
                except json.JSONDecodeError as e:
                    raise ApiError(400, f"invalid JSON: {e}")
                if not isinstance(body, dict):
                    raise ApiError(400, "request body must be a JSON object")
                self._send(200, router.handle(method, path, body))
            except ApiError as e:
                self._send(e.status, {"error": str(e)})
            except (ValueError, TypeError, KeyError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                log.exception("unhandled router error")
                self._send(500, {"error": f"internal error: {type(e).__name__}"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

    return Handler


def make_router_server(router: Router, host: str = "127.0.0.1", port: int = 7700, api_key: str | None = None):
    server = ThreadingHTTPServer((host, port), make_router_handler(router, api_key))
    server.daemon_threads = True
    return server


def serve_cluster(config_path: str, host: str = "127.0.0.1", port: int = 7700, api_key: str | None = None,
                  state_path: str | None = None) -> None:
    import signal

    config = json.loads(Path(config_path).read_text())
    state = Path(state_path) if state_path else Path(config_path).with_name("cluster-state.json")
    router = Router(config, state)
    router.check_health()
    router.start_health_loop()
    server = make_router_server(router, host, port, api_key)
    log.warning("StrataDB %s router on http://%s:%d: %d shards, %d nodes", __version__, host,
                server.server_port, len(router.shards), sum(len(s.nodes) for s in router.shards))

    def stop(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        router.close()
        time.sleep(0)
