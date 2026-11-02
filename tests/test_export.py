import io

import numpy as np
import pytest

from stratadb import Collection, CollectionConfig, Database
from stratadb.__main__ import main as cli


def test_export_import_roundtrip(tmp_path):
    src = Collection(tmp_path / "src", CollectionConfig(dim=8, seal_threshold=20, background_seal=False,
                                                         build_processes=0, text_field="text"))
    data = np.random.default_rng(0).normal(size=(50, 8))
    src.upsert([f"r{i}" for i in range(50)], data, [{"text": f"item {i}", "n": i} for i in range(50)])
    src.put_nodes([{"id": "r1", "props": {"kind": "doc"}}, {"id": "topic", "props": {"kind": "topic"}}])
    src.link([{"src": "r1", "dst": "topic", "rel": "about", "weight": 0.5, "props": {"by": "test"}}])
    src.delete(["r7"])
    buf = io.StringIO()
    assert src.export_jsonl(buf) == {"records": 49, "nodes": 2, "edges": 1}

    dst = Collection(tmp_path / "dst", CollectionConfig(dim=8, text_field="text", background_seal=False))
    buf.seek(0)
    assert dst.import_jsonl(buf, batch=7) == {"records": 49, "nodes": 2, "edges": 1}
    assert len(dst) == 49 and "r7" not in dst
    q = data[3]
    assert [h["id"] for h in dst.search(q, 5)] == [h["id"] for h in src.search(q, 5)]
    assert dst.search(text="item 12", k=1)[0]["id"] == "r12"
    assert dst.get_nodes(["topic"]) == {"topic": {"kind": "topic"}}
    assert dst.edges("r1")[0] == {"src": "r1", "dst": "topic", "rel": "about", "weight": 0.5, "props": {"by": "test"}}

    other = Collection(tmp_path / "other", CollectionConfig(dim=4))
    buf.seek(0)
    with pytest.raises(ValueError, match="dim=8"):
        other.import_jsonl(buf)


def test_cli_export_import(tmp_path, capsys):
    with Database(tmp_path / "db") as db:
        db.create_collection("a", dim=4).upsert(["x", "y"], np.eye(4)[:2], [{"k": 1}, {"k": 2}])
    cli(["export", "--path", str(tmp_path / "db"), "a", "--out", str(tmp_path / "a.jsonl")])
    assert '"records": 2' in capsys.readouterr().out
    cli(["import", "--path", str(tmp_path / "db2"), "b", "--in", str(tmp_path / "a.jsonl")])
    assert '"records": 2' in capsys.readouterr().out
    with Database(tmp_path / "db2") as db:
        assert db["b"].get(["y"])[0]["metadata"] == {"k": 2}
