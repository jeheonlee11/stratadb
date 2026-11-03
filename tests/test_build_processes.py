"""Worker-process index builds: same results as in-process, and safe fallback."""

import numpy as np

from stratadb import Collection, CollectionConfig
from stratadb.segment import SealedSegment


def test_worker_build_matches_in_process(tmp_path):
    data = np.random.default_rng(0).normal(size=(2500, 16)).astype(np.float32)
    ids = [f"r{i}" for i in range(2500)]
    results = {}
    for procs in (0, 1):
        col = Collection(tmp_path / f"c{procs}", CollectionConfig(
            dim=16, seal_threshold=2500, background_seal=False, build_processes=procs,
            build_process_min_records=2000))
        col.upsert(ids, data)
        col.wait_for_seal()
        assert len(col.stats()["sealed_segments"]) == 1
        results[procs] = [h["id"] for h in col.search(data[42], 5, ef=200)]
        assert col._pool_broken is False
        col.close()
    assert results[0][0] == "r42" and results[1][0] == "r42"
    assert set(results[0]) == set(results[1])


def test_small_segments_stay_in_process(tmp_path):
    col = Collection(tmp_path / "c", CollectionConfig(dim=8, seal_threshold=50, background_seal=False,
                                                       build_processes=2, build_process_min_records=1000))
    col.upsert([f"r{i}" for i in range(50)], np.random.default_rng(1).normal(size=(50, 8)))
    assert col._pool is None  # never started a pool for a 50-record segment
    col.close()


def test_worker_failure_falls_back_in_process(tmp_path, monkeypatch):
    def boom(cls, *a, **k):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(SealedSegment, "build_in_process", classmethod(boom))
    col = Collection(tmp_path / "c", CollectionConfig(dim=8, seal_threshold=100, background_seal=False,
                                                       build_processes=2, build_process_min_records=10))
    data = np.random.default_rng(2).normal(size=(100, 8))
    col.upsert([f"r{i}" for i in range(100)], data)
    assert col._pool_broken is True
    assert len(col.stats()["sealed_segments"]) == 1          # built in-process instead
    assert col.search(data[3], 1)[0]["id"] == "r3"
    col.close()


def test_two_segments_build_concurrently(tmp_path, monkeypatch):
    import threading

    started, release = threading.Semaphore(0), threading.Event()
    def gated(cls, seg_id, path, records, metric, m, efc, pool, sq8=True):
        started.release()
        release.wait(10)
        return SealedSegment.build(seg_id, path, records, metric, m, efc, sq8)  # avoid a real pool here

    monkeypatch.setattr(SealedSegment, "build_in_process", classmethod(gated))
    col = Collection(tmp_path / "c", CollectionConfig(dim=8, seal_threshold=100, build_processes=2,
                                                       build_process_min_records=10, auto_compact=False))
    data = np.random.default_rng(3).normal(size=(250, 8))
    col.upsert([f"r{i}" for i in range(250)], data)      # 2 seals start, both parked; 50 stay growing
    assert started.acquire(timeout=5) and started.acquire(timeout=5)
    assert len(col.frozen) == 2 and col.stats()["sealing"] == 200
    assert col.search(data[20], 1)[0]["id"] == "r20"      # first frozen segment searchable
    assert col.search(data[150], 1)[0]["id"] == "r150"    # second frozen segment searchable
    col.delete(["r20", "r150"])                           # one delete in each in-flight segment
    col.upsert(["r30"], data[30:31], [{"v": 2}])          # overwrite moves r30 to growing
    release.set()
    col.wait_for_seal()
    assert len(col.sealed) == 2 and not col.frozen and len(col) == 248
    assert sum(s["live"] for s in col.stats()["sealed_segments"]) == 197
    assert col.get(["r30"])[0]["metadata"] == {"v": 2}
    assert all(h["id"] not in ("r20", "r150") for h in col.search(data[20], 5))
    col.close()
    assert len(Collection(tmp_path / "c")) == 248
