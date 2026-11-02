"""HTTP client with the same method names as the embedded API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np


class StrataError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


def _jsonable(vectors) -> list:
    return np.asarray(vectors, dtype=np.float32).tolist()


class Client:
    def __init__(self, url: str = "http://127.0.0.1:7700", api_key: str | None = None, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", e.reason)
            except (json.JSONDecodeError, AttributeError):
                message = e.reason
            raise StrataError(e.code, message) from None

    def health(self) -> dict:
        return self._request("GET", "/health")

    def list_collections(self) -> list[str]:
        return self._request("GET", "/collections")["collections"]

    def create_collection(self, name: str, dim: int, metric: str = "cosine", **options) -> "RemoteCollection":
        self._request("POST", "/collections", {"name": name, "dim": dim, "metric": metric, **options})
        return RemoteCollection(self, name)

    def collection(self, name: str) -> "RemoteCollection":
        self._request("GET", f"/collections/{name}")
        return RemoteCollection(self, name)

    def drop_collection(self, name: str) -> bool:
        return self._request("DELETE", f"/collections/{name}")["dropped"]


class RemoteCollection:
    def __init__(self, client: Client, name: str):
        self.client = client
        self.name = name

    def _post(self, action: str, body: dict | None = None) -> Any:
        return self.client._request("POST", f"/collections/{self.name}/{action}", body or {})

    def upsert(self, ids, vectors, metadata=None) -> int:
        return self._post("upsert", {"ids": list(ids), "vectors": _jsonable(vectors), "metadata": metadata})["upserted"]

    def delete(self, ids) -> int:
        return self._post("delete", {"ids": list(ids)})["deleted"]

    def get(self, ids, include_vectors: bool = False, fields=None) -> list[dict]:
        return self._post("get", {"ids": list(ids), "include_vectors": include_vectors, "fields": fields})["records"]

    def search(self, vector=None, k: int = 10, filter=None, ef=None, include_vectors=False,
               expand_hops: int = 0, graph_boost: float = 0.0, text: str | None = None, fields=None) -> list[dict]:
        body = {"vector": None if vector is None else _jsonable(vector), "k": k, "filter": filter, "ef": ef,
                "include_vectors": include_vectors, "expand_hops": expand_hops, "graph_boost": graph_boost,
                "text": text, "fields": fields}
        return self._post("search", body)["hits"]

    def search_many(self, vectors, k: int = 10, **kwargs) -> list[list[dict]]:
        body = {"vectors": _jsonable(vectors), "k": k, **kwargs}
        return self._post("search_many", body)["results"]

    def scan(self, filter=None, limit=None, offset: int = 0, fields=None) -> list[dict]:
        return self._post("scan", {"filter": filter, "limit": limit, "offset": offset, "fields": fields})["records"]

    def count(self, filter=None) -> int:
        return self._post("count", {"filter": filter})["count"]

    def update_metadata(self, ids, patch: dict, replace: bool = False) -> int:
        return self._post("update", {"ids": list(ids), "patch": patch, "replace": replace})["updated"]

    def delete_where(self, filter: dict) -> int:
        return self._post("delete_where", {"filter": filter})["deleted"]

    def link(self, edges: list[dict]) -> int:
        return self._post("link", {"edges": edges})["linked"]

    def unlink(self, edges: list[dict]) -> int:
        return self._post("unlink", {"edges": edges})["unlinked"]

    def neighbors(self, rid: str, hops: int = 1, rel=None, direction: str = "both",
                  limit=None, hydrate: bool = False) -> list[dict]:
        body = {"id": rid, "hops": hops, "rel": rel, "direction": direction, "limit": limit, "hydrate": hydrate}
        return self._post("neighbors", body)["neighbors"]

    def put_nodes(self, nodes: list[dict]) -> int:
        return self._post("nodes", {"nodes": nodes})["put"]

    def get_nodes(self, ids) -> dict[str, dict]:
        return self._post("get_nodes", {"ids": list(ids)})["nodes"]

    def list_nodes(self, limit: int = 100, offset: int = 0, prefix: str | None = None) -> dict:
        return self._post("list_nodes", {"limit": limit, "offset": offset, "prefix": prefix})

    def shortest_path(self, src: str, dst: str, max_hops: int = 6, rel=None, direction: str = "both"):
        return self._post("path", {"src": src, "dst": dst, "max_hops": max_hops, "rel": rel,
                                   "direction": direction})["path"]

    def pagerank(self, top=None, seeds=None, damping: float = 0.85, directed: bool = True) -> list:
        return [tuple(x) for x in self._post("pagerank", {"top": top, "seeds": seeds, "damping": damping,
                                                          "directed": directed})["ranking"]]

    def components(self) -> list[list[str]]:
        return self._post("components")["components"]

    def checkpoint(self) -> int:
        return self._post("checkpoint")["lsn"]

    def export_jsonl(self, fp) -> int:
        """Stream the collection's JSON-lines export into ``fp``; returns lines written."""
        req = urllib.request.Request(f"{self.client.url}/collections/{self.name}/export")
        if self.client.api_key:
            req.add_header("Authorization", f"Bearer {self.client.api_key}")
        n = 0
        try:
            with urllib.request.urlopen(req, timeout=self.client.timeout) as resp:
                for line in resp:
                    fp.write(line.decode("utf-8"))
                    n += 1
        except urllib.error.HTTPError as e:
            raise StrataError(e.code, e.reason) from None
        return n

    def merge(self, max_segments: int = 1) -> dict:
        return self._post("merge", {"max_segments": max_segments})

    def compact(self) -> dict:
        return self._post("compact")

    def stats(self) -> dict:
        return self.client._request("GET", f"/collections/{self.name}")
