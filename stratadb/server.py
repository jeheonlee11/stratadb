"""JSON-over-HTTP server (standard library only).

Endpoints, all JSON in and out::

    GET    /                               web console
    GET    /health
    GET    /metrics                        Prometheus text format
    GET    /collections
    POST   /collections                    {name, dim, metric?, ...options}
    GET    /collections/<name>             stats
    DELETE /collections/<name>
    POST   /collections/<name>/upsert      {ids, vectors, metadata?}
    POST   /collections/<name>/delete      {ids}
    POST   /collections/<name>/get         {ids, include_vectors?}
    POST   /collections/<name>/search      {vector? and/or text?, k?, filter?, ef?, include_vectors?, expand_hops?, graph_boost?}
    POST   /collections/<name>/search_many {vectors, k?, ...same options as search}
    POST   /collections/<name>/scan        {filter?, limit?, offset?}
    POST   /collections/<name>/count       {filter?}
    POST   /collections/<name>/update      {ids, patch, replace?}
    POST   /collections/<name>/delete_where {filter}
    POST   /collections/<name>/link        {edges}
    POST   /collections/<name>/unlink      {edges}
    POST   /collections/<name>/neighbors   {id, hops?, rel?, direction?, limit?, hydrate?}
    POST   /collections/<name>/nodes       {nodes: [{id, props}]}
    POST   /collections/<name>/get_nodes   {ids}
    POST   /collections/<name>/list_nodes  {limit?, offset?, prefix?}
    POST   /collections/<name>/path        {src, dst, max_hops?, rel?, direction?}
    POST   /collections/<name>/pagerank    {top?, seeds?, damping?, directed?}
    POST   /collections/<name>/components
    POST   /collections/<name>/checkpoint
    POST   /collections/<name>/compact
    POST   /collections/<name>/merge       {max_segments?}
    POST   /collections/<name>/tune_ef     {target_recall?, k?, samples?, apply?}
    GET    /collections/<name>/export      streams JSON lines (see Collection.export_jsonl)

With ``api_key`` set, every request except ``/health`` needs
``Authorization: Bearer <key>``.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from importlib import resources
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from . import __version__
from .database import Database
from .filters import FilterError

log = logging.getLogger("stratadb.server")
MAX_BODY = 256 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _require(body: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in body]
    if missing:
        raise ApiError(400, f"missing field(s): {', '.join(missing)}")


def _pick(body: dict, *keys: str) -> dict:
    return {k: body[k] for k in keys if k in body and body[k] is not None}


COLLECTION_OPTIONS = ("m", "ef_construction", "ef_search", "seal_threshold", "sync",
                      "wal_checkpoint_bytes", "compact_deleted_ratio", "background_seal", "exact_search_limit", "text_field", "index_fields", "auto_compact", "build_processes", "build_process_min_records")


class Metrics:
    """Request counters and latency sums, rendered in Prometheus text format."""

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self.requests: dict[tuple[str, str], int] = {}      # (action, status) -> count
        self.seconds: dict[str, float] = {}                 # action -> total seconds
        self.started = time.time()

    def record(self, action: str, status: int, seconds: float) -> None:
        with self._lock:
            key = (action, str(status))
            self.requests[key] = self.requests.get(key, 0) + 1
            self.seconds[action] = self.seconds.get(action, 0.0) + seconds

    def render(self, db: Database) -> str:
        lines = ["# HELP stratadb_up 1 while the server runs.", "# TYPE stratadb_up gauge", "stratadb_up 1",
                 "# HELP stratadb_uptime_seconds Seconds since the server started.",
                 "# TYPE stratadb_uptime_seconds gauge", f"stratadb_uptime_seconds {time.time() - self.started:.0f}",
                 "# HELP stratadb_requests_total Requests by action and HTTP status.",
                 "# TYPE stratadb_requests_total counter"]
        with self._lock:
            for (action, status), n in sorted(self.requests.items()):
                lines.append(f'stratadb_requests_total{{action="{action}",status="{status}"}} {n}')
            lines += ["# HELP stratadb_request_seconds_total Time spent handling requests, by action.",
                      "# TYPE stratadb_request_seconds_total counter"]
            for action, secs in sorted(self.seconds.items()):
                lines.append(f'stratadb_request_seconds_total{{action="{action}"}} {secs:.6f}')
        gauges = [("records", "Live records", "count"), ("edges", "Edges", "edges"), ("nodes", "Nodes with properties", "nodes"),
                  ("wal_bytes", "Bytes in the write-ahead log since the last checkpoint", "wal_bytes"),
                  ("growing_records", "Records in the growing segment", "growing"),
                  ("sealing_records", "Records in segments being indexed", "sealing")]
        stats = []
        for name in db.list_collections():
            try:
                stats.append(db.collection(name).stats())
            except Exception:  # noqa: BLE001 - a broken collection must not break metrics
                continue
        for key, help_text, field in gauges:
            lines += [f"# HELP stratadb_{key} {help_text}.", f"# TYPE stratadb_{key} gauge"]
            for st in stats:
                lines.append(f'stratadb_{key}{{collection="{st["name"]}"}} {st.get(field, 0)}')
        lines += ["# HELP stratadb_sealed_segments Sealed segments per collection.",
                  "# TYPE stratadb_sealed_segments gauge"]
        for st in stats:
            lines.append(f'stratadb_sealed_segments{{collection="{st["name"]}"}} {len(st["sealed_segments"])}')
        return "\n".join(lines) + "\n"


class Api:
    """Routes requests to a Database. Kept separate from HTTP so it's easy to test."""

    def __init__(self, db: Database):
        self.db = db
        self.metrics = Metrics()

    def handle(self, method: str, path: str, body: dict) -> Any:
        parts = [p for p in path.split("?")[0].split("/") if p]
        if parts == ["health"] and method == "GET":
            return {"status": "ok", "version": __version__}
        if not parts or parts[0] != "collections":
            raise ApiError(404, f"no route for {method} {path}")
        if len(parts) == 1:
            if method == "GET":
                return {"collections": self.db.list_collections()}
            if method == "POST":
                _require(body, "name", "dim")
                col = self.db.create_collection(
                    body["name"], int(body["dim"]), body.get("metric", "cosine"),
                    **_pick(body, *COLLECTION_OPTIONS),
                )
                return col.stats()
        name = parts[1]
        if len(parts) == 2:
            if method == "GET":
                return self._col(name).stats()
            if method == "DELETE":
                return {"dropped": self.db.drop_collection(name)}
        if len(parts) == 3 and method == "POST":
            col = self._col(name)
            action = parts[2]
            handler: Callable[[], Any] | None = {
                "upsert": lambda: (_require(body, "ids", "vectors"),
                                   {"upserted": col.upsert(body["ids"], body["vectors"], body.get("metadata"))})[1],
                "delete": lambda: (_require(body, "ids"), {"deleted": col.delete(body["ids"])})[1],
                "get": lambda: (_require(body, "ids"),
                                {"records": col.get(body["ids"], bool(body.get("include_vectors")), body.get("fields"))})[1],
                "search": lambda: {"hits": col.search(
                    body.get("vector"), int(body.get("k", 10)),
                    **_pick(body, "filter", "ef", "include_vectors", "expand_hops", "graph_boost", "text", "fields", "graph_rels"))},
                "search_many": lambda: (_require(body, "vectors"), {"results": col.search_many(
                    body["vectors"], int(body.get("k", 10)),
                    **_pick(body, "filter", "ef", "include_vectors", "expand_hops", "graph_boost", "text"))})[1],
                "scan": lambda: {"records": col.scan(body.get("filter"), body.get("limit"), int(body.get("offset") or 0),
                                                     body.get("fields"))},
                "count": lambda: {"count": col.count(body.get("filter"))},
                "update": lambda: (_require(body, "ids", "patch"), {"updated": col.update_metadata(
                    body["ids"], body["patch"], bool(body.get("replace")))})[1],
                "delete_where": lambda: (_require(body, "filter"), {"deleted": col.delete_where(body["filter"])})[1],
                "link": lambda: (_require(body, "edges"), {"linked": col.link(body["edges"])})[1],
                "unlink": lambda: (_require(body, "edges"), {"unlinked": col.unlink(body["edges"])})[1],
                "neighbors": lambda: (_require(body, "id"), {"neighbors": col.neighbors(
                    body["id"], **_pick(body, "hops", "rel", "direction", "limit", "hydrate", "filter"))})[1],
                "nodes": lambda: (_require(body, "nodes"), {"put": col.put_nodes(body["nodes"])})[1],
                "get_nodes": lambda: (_require(body, "ids"), {"nodes": col.get_nodes(body["ids"])})[1],
                "list_nodes": lambda: col.list_nodes(int(body.get("limit", 100)), int(body.get("offset", 0)),
                                                     body.get("prefix")),
                "path": lambda: (_require(body, "src", "dst"), {"path": col.shortest_path(
                    body["src"], body["dst"], **_pick(body, "max_hops", "rel", "direction"))})[1],
                "pagerank": lambda: {"ranking": col.pagerank(**_pick(body, "top", "seeds", "damping", "directed"))},
                "components": lambda: {"components": col.components()},
                "checkpoint": lambda: {"lsn": col.checkpoint()},
                "compact": lambda: col.compact(),
                "merge": lambda: col.merge(int(body.get("max_segments", 1))),
                "tune_ef": lambda: col.tune_ef(**_pick(body, "target_recall", "k", "samples", "seed", "apply", "queries")),
            }.get(action)
            if handler is not None:
                return handler()
        raise ApiError(404, f"no route for {method} {path}")

    def _col(self, name: str):
        try:
            return self.db.collection(name)
        except KeyError:
            raise ApiError(404, f"no collection named {name!r}")


def make_handler(api: Api, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"StrataDB/{__version__}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # route through logging instead of stderr
            log.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: Any) -> None:
            self._last_status = status
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_console(self) -> None:
            data = resources.files("stratadb").joinpath("console.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_export(self, name: str) -> None:
            """Stream a collection as JSON lines (chunked, so no size limit)."""
            try:
                col = api._col(name)
            except ApiError as e:
                self._send(e.status, {"error": str(e)})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            out = self.wfile

            class Chunked:
                def write(self, text: str) -> None:
                    data = text.encode()
                    out.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")

            col.export_jsonl(Chunked())
            out.write(b"0\r\n\r\n")

        def _dispatch(self, method: str) -> None:
            path = self.path.split("?")[0]
            if method == "GET" and path in ("/", "/console"):
                self._send_console()  # the page is static; its API calls carry the key
                return
            parts = [p for p in path.split("/") if p]
            if method == "GET" and path == "/metrics":
                if api_key and not hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                                      f"Bearer {api_key}".encode()):
                    self._send(401, {"error": "missing or wrong API key"})
                    return
                data = api.metrics.render(api.db).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if method == "GET" and len(parts) == 3 and parts[0] == "collections" and parts[2] == "export":
                if api_key and not hmac.compare_digest(self.headers.get("Authorization", "").encode(),
                                                      f"Bearer {api_key}".encode()):
                    self._send(401, {"error": "missing or wrong API key"})
                    return
                self._send_export(parts[1])
                return
            action = parts[2] if len(parts) == 3 else (parts[0] if parts else "root")
            started = time.perf_counter()
            status = 500
            try:
                if api_key and self.path.split("?")[0] != "/health":
                    supplied = self.headers.get("Authorization", "")
                    if not hmac.compare_digest(supplied.encode(), f"Bearer {api_key}".encode()):
                        raise ApiError(401, "missing or wrong API key")
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
                self._send(200, api.handle(method, self.path, body))
            except ApiError as e:
                self._send(e.status, {"error": str(e)})
            except (ValueError, TypeError, FilterError) as e:
                self._send(400, {"error": str(e)})
            except KeyError as e:
                self._send(404, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - last-resort guard for the server loop
                log.exception("unhandled error")
                self._send(500, {"error": f"internal error: {type(e).__name__}"})
            finally:
                api.metrics.record(action, getattr(self, "_last_status", status), time.perf_counter() - started)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

    return Handler


def make_server(db: Database, host: str = "127.0.0.1", port: int = 7700, api_key: str | None = None):
    server = ThreadingHTTPServer((host, port), make_handler(Api(db), api_key))
    server.daemon_threads = True
    return server


def serve(path: str, host: str = "127.0.0.1", port: int = 7700, api_key: str | None = None) -> None:
    import signal
    import threading

    db = Database(path)
    server = make_server(db, host, port, api_key)
    log.warning("StrataDB %s serving %s on http://%s:%d", __version__, path, host, server.server_port)

    # SIGTERM (docker stop, systemd) and SIGINT both stop the server loop so
    # the database closes cleanly: a final checkpoint, nothing to replay later.
    def stop(signum, frame):
        log.warning("signal %s: shutting down", signal.Signals(signum).name)
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        db.close()
        log.warning("closed %s", path)
