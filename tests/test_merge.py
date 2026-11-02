import numpy as np
import pytest

from stratadb import Collection, CollectionConfig


def test_merge_reduces_segments_and_keeps_results(tmp_path):
    data = np.random.default_rng(0).normal(size=(700, 16)).astype(np.float32)
    col = Collection(tmp_path / "c", CollectionConfig(dim=16, seal_threshold=100, background_seal=False,
                                                       build_processes=0, auto_compact=False))
    col.upsert([f"r{i}" for i in range(700)], data, [{"i": i} for i in range(700)])
    col.delete(["r5", "r650"])
    before = [h["id"] for h in col.search(data[42], 5, ef=200)]
    assert len(col.stats()["sealed_segments"]) == 7
    result = col.merge(2)
    assert result["rewritten"] == 7 and result["segments"] == 2
    segs = col.stats()["sealed_segments"]
    assert sorted(s["live"] for s in segs) == [349, 349] and all(s["deleted_ratio"] == 0 for s in segs)
    assert len(col) == 698 and "r5" not in col
    assert [h["id"] for h in col.search(data[42], 5, ef=200)] == before
    assert col.merge(2) == {"rewritten": 0, "segments": 2}      # already merged: no-op
    assert col.merge(1)["segments"] == 1
    col.close()
    assert len(Collection(tmp_path / "c")) == 698
    with pytest.raises(ValueError):
        col.merge(0)


def test_merge_over_http(tmp_path):
    import threading

    from stratadb import Database
    from stratadb.client import Client
    from stratadb.server import make_server

    db = Database(tmp_path / "db")
    srv = make_server(db, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = Client(f"http://127.0.0.1:{srv.server_port}").create_collection(
            "m", dim=4, seal_threshold=10, background_seal=False, build_processes=0)
        c.upsert([f"r{i}" for i in range(40)], np.random.default_rng(1).normal(size=(40, 4)))
        assert c.merge(1)["segments"] == 1
    finally:
        srv.shutdown(); srv.server_close(); db.close()
