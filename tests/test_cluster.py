"""The router over real node servers: sharding, replication, failover, parity with one node."""

import io
import json
import threading
import urllib.request

import numpy as np
import pytest

from stratadb import Collection, CollectionConfig, Database
from stratadb.client import Client, StrataError
from stratadb.cluster import Router, make_router_server
from stratadb.server import make_server

KEY = "node-key"


class NodeProc:
    """An in-process ``stratadb serve`` on a random port, restartable on the same port."""

    def __init__(self, path):
        self.path = path
        self.port = 0
        self.db = self.srv = self.thread = None
        self.start()

    def start(self):
        self.db = Database(self.path)
        self.srv = make_server(self.db, port=self.port, api_key=KEY)
        self.port = self.srv.server_port
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.db.close()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def cluster(tmp_path):
    """2 shards x 2 nodes, plus a router on top."""
    nodes = [[NodeProc(tmp_path / f"s{s}n{n}") for n in range(2)] for s in range(2)]
    config = {"shards": [{"nodes": [n.url for n in shard]} for shard in nodes], "api_key": KEY}
    router = Router(config, tmp_path / "cluster-state.json")
    srv = make_router_server(router, port=0, api_key="router-key")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield {"nodes": nodes, "router": router, "url": f"http://127.0.0.1:{srv.server_port}"}
    srv.shutdown()
    srv.server_close()
    router.close()
    for shard in nodes:
        for n in shard:
            try:
                n.stop()
            except Exception:  # noqa: BLE001 - already stopped by a test
                pass


def data(n, dim=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def test_sharded_collection_matches_single_node(cluster, tmp_path):
    client = Client(cluster["url"], api_key="router-key")
    assert client.health()["role"] == "router" and client.health()["nodes_up"] == 4
    col = client.create_collection("docs", dim=16, seal_threshold=64)
    vecs = data(300)
    ids = [f"d{i}" for i in range(300)]
    metas = [{"i": i, "even": i % 2 == 0} for i in range(300)]
    assert col.upsert(ids, vecs, metas) == 300

    # Records are spread over both shards, replicated within each.
    stats = col.stats()
    assert stats["count"] == 300 and stats["sharding"] == "hash"
    per_shard = [s["count"] for s in stats["shards"]]
    assert len(per_shard) == 2 and all(50 < c < 250 for c in per_shard)
    for shard in cluster["nodes"]:
        counts = {n.db["docs"].stats()["count"] for n in shard}
        assert len(counts) == 1

    # Same answers as one embedded collection holding everything.
    single = Collection(tmp_path / "single", CollectionConfig(dim=16, seal_threshold=64))
    single.upsert(ids, vecs, metas)
    single.wait_for_seal()
    for q in vecs[:20]:
        want = [h["id"] for h in single.search(q, 5, ef=200)]
        got = [h["id"] for h in col.search(q, 5, ef=200)]
        assert got == want
        assert [h["id"] for h in col.search(q, 5, ef=200, filter={"even": True})] == \
            [h["id"] for h in single.search(q, 5, ef=200, filter={"even": True})]
    many = col.search_many(vecs[:3], k=4, ef=200)
    assert [[h["id"] for h in r] for r in many] == [[h["id"] for h in single.search(q, 4, ef=200)] for q in vecs[:3]]
    single.close()

    # Id-routed reads and writes.
    got = col.get(["d7", "d8", "nope", "d150"], include_vectors=True)
    assert [g["id"] for g in got] == ["d7", "d8", "d150"] and got[0]["metadata"]["i"] == 7
    assert col.count({"even": True}) == 150
    assert col.update_metadata(["d7", "d8"], {"tag": "x"}) == 2
    assert all(g["metadata"].get("tag") == "x" for g in col.get(["d7", "d8"]))
    assert col.delete(["d7", "d300"]) == 1 and col.count() == 299
    assert col.delete_where({"i": {"$gte": 290}}) == 10 and col.count() == 289
    scanned = col.scan(filter={"i": {"$lt": 20}}, limit=5, offset=2)
    assert len(scanned) == 5 and all(r["metadata"]["i"] < 20 for r in scanned)
    assert len(col.scan(filter={"i": {"$lt": 20}})) == 19

    # Edges: stored on both endpoints' shards, so neighbours resolve either way.
    assert col.link([{"src": "d1", "dst": "d2", "rel": "next"}, {"src": "d2", "dst": "d3", "rel": "next"}]) == 2
    assert [n["id"] for n in col.neighbors("d1")] == ["d2"]
    assert {n["id"] for n in col.neighbors("d2")} == {"d1", "d3"}
    hydrated = col.neighbors("d2", hydrate=True)
    assert all(n["metadata"]["i"] in (1, 3) for n in hydrated)
    assert col.put_nodes([{"id": "d1", "props": {"kind": "a"}}, {"id": "d2", "props": {"kind": "b"}}]) == 2
    assert col.get_nodes(["d1", "d2"]) == {"d1": {"kind": "a"}, "d2": {"kind": "b"}}
    with pytest.raises(StrataError) as e:
        col.pagerank(top=3)
    assert "sharding='none'" in str(e.value)

    # Export concatenates the shards under one header.
    buf = io.StringIO()
    col.export_jsonl(buf)
    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert sum(1 for line in lines if line["type"] == "collection") == 1
    assert sum(1 for line in lines if line["type"] == "record") == 289

    assert client.list_collections() == ["docs"]
    assert client.drop_collection("docs")
    assert client.list_collections() == []


def test_unsharded_collection_supports_whole_graph(cluster):
    client = Client(cluster["url"], api_key="router-key")
    col = client.create_collection("graph", dim=8, sharding="none")
    vecs = data(20, dim=8)
    col.upsert([f"n{i}" for i in range(20)], vecs)
    col.link([{"src": f"n{i}", "dst": f"n{i + 1}", "rel": "next"} for i in range(19)])
    path = col.shortest_path("n0", "n4", max_hops=6)
    assert [e["from"] for e in path] + [path[-1]["to"]] == ["n0", "n1", "n2", "n3", "n4"]
    assert len(col.components()) == 1
    assert col.pagerank(top=1)[0][0] == "n19"
    assert col.stats()["shards"][0]["count"] == 20 and len(col.stats()["shards"]) == 1
    # Shard 1 never saw the data.
    assert cluster["nodes"][1][0].db["graph"].stats()["count"] == 0


def test_replica_failover_and_resync(cluster):
    client = Client(cluster["url"], api_key="router-key")
    router = cluster["router"]
    col = client.create_collection("docs", dim=16, seal_threshold=1000)
    vecs = data(200)
    col.upsert([f"d{i}" for i in range(200)], vecs)

    # Primary of shard 0 goes down: reads continue from its replica.
    s0 = cluster["nodes"][0]
    s0[0].stop()
    q = vecs[5]
    assert col.search(q, 3)[0]["id"] == "d5"
    assert client.health()["status"] == "degraded"
    # Writes to that shard are refused (no promotion), writes to the other shard still work.
    with pytest.raises(StrataError) as e:
        col.upsert([f"d{i}" for i in range(200, 400)], data(200, seed=1))
    assert e.value.status == 503
    s0[0].start()
    router.check_health()
    assert client.health()["status"] == "ok"
    assert col.upsert([f"d{i}" for i in range(200, 400)], data(200, seed=1)) == 200

    # A replica goes down and misses writes: marked stale, then resynced.
    s1 = cluster["nodes"][1]
    s1[1].stop()
    assert col.upsert([f"e{i}" for i in range(100)], data(100, seed=2)) == 100
    assert col.delete(["d5"]) == 1
    assert router.shards[1].nodes[1].up is False
    s1[1].start()
    router.check_health()           # probes it up; resyncs because it was marked stale
    assert router.shards[1].nodes[1].stale is False
    primary_ids = {r["id"] for r in s1[0].db["docs"].scan()}
    replica_ids = {r["id"] for r in s1[1].db["docs"].scan()}
    assert primary_ids == replica_ids and "d5" not in replica_ids and len(primary_ids) > 0
    status = router.status()
    assert all(n["up"] and not n["stale"] for s in status["shards"] for n in s["nodes"])


def test_router_requires_its_own_key(cluster):
    req = urllib.request.Request(cluster["url"] + "/collections")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 401
    assert Client(cluster["url"]).health()["status"] in ("ok", "degraded")   # health is open
