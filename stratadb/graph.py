"""Typed, weighted edges between records, and properties on nodes.

StrataDB keeps relationships next to the vectors instead of in a separate
graph database. An edge is ``(src, rel, dst)`` with a weight and optional
properties; the same pair can have several edges with different ``rel``
types. Edges may point at ids that don't exist (yet): retrieval simply skips
them, and deleting a record removes every edge touching it.

Nodes can also carry a property dict of their own (``put_node``), which is
what lets a collection hold a whole knowledge graph, not just links between
vector records.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

EdgeKey = tuple[str, str, str]  # (src, rel, dst)


class GraphStore:
    def __init__(self):
        self._out: dict[str, dict[tuple[str, str], tuple[float, dict]]] = {}
        self._in: dict[str, set[tuple[str, str]]] = {}
        self._nodes: dict[str, dict] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._out.values())

    def add(self, src: str, dst: str, rel: str = "related", weight: float = 1.0, props: dict | None = None) -> None:
        self._out.setdefault(src, {})[(rel, dst)] = (float(weight), dict(props or {}))
        self._in.setdefault(dst, set()).add((rel, src))

    def remove(self, src: str, dst: str, rel: str | None = None) -> int:
        """Remove src->dst edges (all relation types when ``rel`` is None)."""
        out = self._out.get(src)
        if not out:
            return 0
        keys = [k for k in out if k[1] == dst and (rel is None or k[0] == rel)]
        for r, d in keys:
            del out[(r, d)]
            inn = self._in.get(d)
            if inn is not None:
                inn.discard((r, src))
                if not inn:
                    del self._in[d]
        if not out:
            del self._out[src]
        return len(keys)

    # ---------------------------------------------------------------- nodes
    def put_node(self, node: str, props: dict | None = None) -> None:
        self._nodes[node] = dict(props or {})

    def get_node(self, node: str) -> dict | None:
        return self._nodes.get(node)

    def has_node(self, node: str) -> bool:
        return node in self._nodes

    def node_ids(self) -> list[str]:
        return list(self._nodes)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def remove_node(self, node: str) -> int:
        """Remove a node's properties and every edge touching it."""
        self._nodes.pop(node, None)
        removed = 0
        for rel, dst in list(self._out.get(node, {})):
            removed += self.remove(node, dst, rel)
        for rel, src in list(self._in.get(node, ())):
            removed += self.remove(src, node, rel)
        return removed

    def degree(self, node: str) -> int:
        return len(self._out.get(node, {})) + len(self._in.get(node, ()))

    def neighbor_ids(self, node: str) -> set[str]:
        """Distinct nodes adjacent in either direction (any relation)."""
        ids = {dst for _, dst in self._out.get(node, {})}
        ids.update(src for _, src in self._in.get(node, ()))
        ids.discard(node)
        return ids

    def get_edge(self, src: str, dst: str, rel: str) -> tuple[float, dict] | None:
        return self._out.get(src, {}).get((rel, dst))

    def all_edges(self):
        for src, out in self._out.items():
            for (rel, dst), (w, props) in out.items():
                yield src, rel, dst, w, props

    def edges(self, node: str, direction: str = "both", rel: str | None = None) -> list[dict]:
        result = []
        if direction in ("out", "both"):
            for (r, dst), (w, props) in self._out.get(node, {}).items():
                if rel is None or r == rel:
                    result.append({"src": node, "dst": dst, "rel": r, "weight": w, "props": props})
        if direction in ("in", "both"):
            for r, src in self._in.get(node, ()):
                if rel is None or r == rel:
                    w, props = self._out[src][(r, node)]
                    result.append({"src": src, "dst": node, "rel": r, "weight": w, "props": props})
        return result

    def _step(self, node: str, direction: str, rel: str | None) -> Iterable[tuple[str, str, float]]:
        if direction in ("out", "both"):
            for (r, dst), (w, _) in self._out.get(node, {}).items():
                if rel is None or r == rel:
                    yield dst, r, w
        if direction in ("in", "both"):
            for r, src in self._in.get(node, ()):
                if rel is None or r == rel:
                    yield src, r, self._out[src][(r, node)][0]

    def neighbors(
        self,
        node: str,
        hops: int = 1,
        rel: str | None = None,
        direction: str = "both",
        limit: int | None = None,
    ) -> list[dict]:
        """Breadth-first expansion. Each result records its hop distance, the
        relation it was reached by and the node it was reached from."""
        seen = {node}
        result: list[dict] = []
        frontier = deque([(node, 0)])
        while frontier:
            current, depth = frontier.popleft()
            if depth >= hops:
                continue
            for nxt, r, w in self._step(current, direction, rel):
                if nxt in seen:
                    continue
                seen.add(nxt)
                result.append({"id": nxt, "hops": depth + 1, "rel": r, "weight": w, "via": current})
                if limit is not None and len(result) >= limit:
                    return result
                frontier.append((nxt, depth + 1))
        return result

    def weighted_neighbors(self, node: str) -> list[tuple[str, float]]:
        """Undirected 1-hop neighbours with their strongest edge weight."""
        best: dict[str, float] = {}
        for nxt, _, w in self._step(node, "both", None):
            if nxt != node and w > best.get(nxt, float("-inf")):
                best[nxt] = w
        return list(best.items())

    # ------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        return {"edges": [list(e) for e in self.all_edges()], "nodes": self._nodes}

    @classmethod
    def from_json(cls, data) -> "GraphStore":
        g = cls()
        if isinstance(data, list):  # snapshots before node properties existed
            data = {"edges": data, "nodes": {}}
        for src, rel, dst, w, props in data["edges"]:
            g.add(src, dst, rel, w, props)
        g._nodes = {k: dict(v) for k, v in data.get("nodes", {}).items()}
        return g
