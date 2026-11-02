import threading

import numpy as np

from stratadb import Collection, CollectionConfig
from stratadb.segment import SealedSegment


def rand(n, dim=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def make(tmp_path, **kw):
    kw.setdefault("seal_threshold", 200)
    kw.setdefault("background_seal", True)
    return Collection(tmp_path / "c", CollectionConfig(dim=16, **kw))


class SlowBuild:
    """Hold SealedSegment.build until released, so tests can act mid-build."""

    def __init__(self, monkeypatch):
        self.started = threading.Event()
        self.release = threading.Event()
        original = SealedSegment.build.__func__

        def slow(cls, *args, **kwargs):
            self.started.set()
            self.release.wait(10)
            return original(cls, *args, **kwargs)

        monkeypatch.setattr(SealedSegment, "build", classmethod(slow))


def test_writes_and_reads_continue_during_build(tmp_path, monkeypatch):
    gate = SlowBuild(monkeypatch)
    col = make(tmp_path)
    data = rand(260)
    col.upsert([f"r{i}" for i in range(200)], data[:200])  # triggers a background seal
    assert gate.started.wait(5)
    stats = col.stats()
    assert stats["sealing"] == 200 and stats["growing"] == 0

    # Frozen records are still searchable, and writes go to the new growing segment.
    assert col.search(data[17], 1)[0]["id"] == "r17"
    col.upsert([f"r{i}" for i in range(200, 260)], data[200:])
    col.delete(["r3"])                                   # delete a frozen record mid-build
    col.upsert(["r4"], rand(1, seed=99), [{"v": 2}])     # overwrite a frozen record mid-build
    assert "r3" not in col and col.get(["r4"])[0]["metadata"] == {"v": 2}

    gate.release.set()
    col.wait_for_seal()
    stats = col.stats()
    assert stats["sealing"] == 0 and len(stats["sealed_segments"]) == 1
    assert stats["sealed_segments"][0]["live"] == 198     # r3 deleted, r4 moved to growing
    assert len(col) == 259
    assert "r3" not in [h["id"] for h in col.search(data[3], 5)]
    col.close()

    reopened = Collection(tmp_path / "c")
    assert len(reopened) == 259 and "r3" not in reopened
    assert reopened.get(["r4"])[0]["metadata"] == {"v": 2}


def test_crash_during_build_loses_nothing(tmp_path, monkeypatch):
    gate = SlowBuild(monkeypatch)
    col = make(tmp_path)
    data = rand(230)
    col.upsert([f"r{i}" for i in range(200)], data[:200])
    assert gate.started.wait(5)
    col.upsert([f"r{i}" for i in range(200, 230)], data[200:])
    col.checkpoint()  # snapshot taken while the frozen segment is still building
    col.upsert(["late"], rand(1, seed=5))
    # Crash: the build never finishes and close() is never called.
    monkeypatch.undo()
    col._dir_lock.release()
    reopened = Collection(tmp_path / "c", CollectionConfig(dim=16, seal_threshold=200, background_seal=False))
    assert len(reopened) == 231
    assert reopened.search(data[150], 1)[0]["id"] == "r150"
    # A real crash takes the build thread with it. Here it is still parked in
    # the gate, so mark the dead instance closed (no checkpoint on install)
    # and let the thread finish before the test directory is removed.
    col._closed = True
    gate.release.set()
    col._seal_thread.join(10)
    reopened.close()


def test_failed_build_keeps_records_searchable(tmp_path, monkeypatch):
    def boom(cls, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(SealedSegment, "build", classmethod(boom))
    col = make(tmp_path)
    data = rand(200)
    col.upsert([f"r{i}" for i in range(200)], data)
    col.wait_for_seal()
    assert isinstance(col._seal_error, OSError)
    assert col.stats()["sealing"] == 0 and len(col) == 200
    assert col.search(data[9], 1)[0]["id"] == "r9"


def test_many_seals_from_concurrent_writers(tmp_path):
    col = make(tmp_path, seal_threshold=100)
    data = rand(1200)

    def writer(offset):
        for start in range(offset, offset + 300, 50):
            col.upsert([f"r{i}" for i in range(start, start + 50)], data[start:start + 50])

    threads = [threading.Thread(target=writer, args=(o,)) for o in (0, 300, 600, 900)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    col.wait_for_seal()
    assert len(col) == 1200
    stats = col.stats()
    assert sum(s["live"] for s in stats["sealed_segments"]) + stats["growing"] == 1200
    for i in (0, 333, 1199):
        assert col.search(data[i], 1)[0]["id"] == f"r{i}"
    col.close()
    assert len(Collection(tmp_path / "c")) == 1200


def test_exact_planner_matches_hnsw_results(tmp_path):
    data = rand(500)
    exact = Collection(tmp_path / "exact", CollectionConfig(dim=16, seal_threshold=500, background_seal=False))
    graph = Collection(tmp_path / "graph", CollectionConfig(dim=16, seal_threshold=500, background_seal=False,
                                                           exact_search_limit=0))
    for col in (exact, graph):
        col.upsert([f"r{i}" for i in range(500)], data)
    q = rand(1, seed=7)[0]
    a = [h["id"] for h in exact.search(q, 10)]
    b = [h["id"] for h in graph.search(q, 10, ef=200)]
    assert len(set(a) & set(b)) >= 9


def test_compaction_runs_outside_the_lock_and_reconciles(tmp_path, monkeypatch):
    col = make(tmp_path, seal_threshold=100, background_seal=False, auto_compact=False)
    data = rand(200)
    col.upsert([f"r{i}" for i in range(200)], data)
    col.delete([f"r{i}" for i in range(50)])          # segment 1 is now 50% tombstones
    gate = SlowBuild(monkeypatch)
    result = {}
    worker = threading.Thread(target=lambda: result.update(col.compact()))
    worker.start()
    assert gate.started.wait(5)
    # The build is parked: reads and writes must not block on it.
    assert col.search(data[60], 1)[0]["id"] == "r60"
    col.delete(["r70"])                                 # delete a record being rewritten
    col.upsert(["r80"], rand(1, seed=9), [{"v": 2}])    # overwrite one
    gate.release.set()
    worker.join(10)
    assert result["rewritten"] >= 1
    assert "r70" not in col and col.get(["r80"])[0]["metadata"] == {"v": 2}
    assert len(col) == 149
    assert all(s["deleted_ratio"] < 0.3 for s in col.stats()["sealed_segments"])
    col.close()
    reopened = Collection(tmp_path / "c")
    assert len(reopened) == 149 and "r70" not in reopened


def test_auto_compaction_kicks_in(tmp_path):
    col = make(tmp_path, seal_threshold=100, background_seal=False, auto_compact=True)
    col.upsert([f"r{i}" for i in range(200)], rand(200))
    col.delete([f"r{i}" for i in range(40)])
    col.wait_for_compaction()
    segs = col.stats()["sealed_segments"]
    assert all(s["deleted_ratio"] == 0 for s in segs) and len(col) == 160
