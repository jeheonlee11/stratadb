import json

import numpy as np
import pytest

from stratadb import Collection, CollectionConfig, Database
from stratadb.graph import GraphStore


def test_node_properties_persist_and_cascade(tmp_path):
    with Database(tmp_path / "db") as db:
        g = db.create_graph("kg")
        g.put_nodes([{"id": "Ada", "props": {"type": "person"}}, {"id": "Babbage", "props": {"type": "person"}},
                     {"id": "Engine", "props": {"type": "machine"}}])
        g.link([{"src": "Ada", "dst": "Engine", "rel": "wrote_about", "props": {"year": 1843}},
                {"src": "Babbage", "dst": "Engine", "rel": "designed"}])
        assert g.get_nodes(["Ada", "nope"]) == {"Ada": {"type": "person"}}
        g.put_nodes([{"id": "Ada", "props": {"type": "person", "born": 1815}}])  # replaces props
        assert g.stats()["nodes"] == 3 and g.stats()["edges"] == 2
        assert g.delete(["Engine"]) == 1          # node with no vector still counts as present
        assert g.stats()["edges"] == 0

    with Database(tmp_path / "db") as db:
        g = db.collection("kg")
        assert g.get_nodes(["Ada"])["Ada"]["born"] == 1815
        assert sorted(g.node_ids()) == ["Ada", "Babbage"]


def test_nodes_survive_crash_via_wal(tmp_path):
    col = Collection(tmp_path / "c", CollectionConfig(dim=0))
    col.put_nodes([{"id": "x", "props": {"n": 1}}])
    col.link([{"src": "x", "dst": "y"}])
    col._dir_lock.release()  # simulate the process dying without close()
    again = Collection(tmp_path / "c")
    assert again.get_nodes(["x"]) == {"x": {"n": 1}} and again.edges("x")[0]["dst"] == "y"


def test_graph_only_collection_rejects_vectors(tmp_path):
    g = Collection(tmp_path / "g", CollectionConfig(dim=0))
    with pytest.raises(ValueError, match="graph-only"):
        g.upsert(["a"], np.zeros((1, 0)))
    with pytest.raises(ValueError, match="graph-only"):
        g.search([1.0])
    with pytest.raises(ValueError):
        g.put_nodes([{"props": {}}])


def test_vector_records_and_nodes_share_ids(tmp_path):
    col = Collection(tmp_path / "c", CollectionConfig(dim=4, background_seal=False))
    col.upsert(["a"], np.eye(4)[:1])
    col.put_nodes([{"id": "a", "props": {"kind": "doc"}}])
    col.delete(["a"])
    assert "a" not in col and col.get_nodes(["a"]) == {}


def test_old_edge_only_snapshots_still_load():
    g = GraphStore.from_json([["a", "rel", "b", 1.0, {}]])
    assert g.neighbor_ids("a") == {"b"} and g.node_count == 0
    roundtrip = GraphStore.from_json(json.loads(json.dumps(g.to_json())))
    assert roundtrip.get_edge("a", "b", "rel") == (1.0, {})
