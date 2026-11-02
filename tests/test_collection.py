import numpy as np
import pytest

from stratadb import Collection, CollectionConfig, Database


def rand(n, dim=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def make(tmp_path, **kw):
    kw.setdefault("seal_threshold", 64)
    # Deterministic, synchronous seals and always use the HNSW path; the
    # background and planner paths have their own tests.
    kw.setdefault("background_seal", False)
    kw.setdefault("exact_search_limit", 0)
    return Collection(tmp_path / "c", CollectionConfig(dim=16, **kw))


def ids(n, prefix="r"):
    return [f"{prefix}{i}" for i in range(n)]


def test_upsert_get_delete(tmp_path):
    col = make(tmp_path)
    vecs = rand(3)
    col.upsert(["a", "b", "c"], vecs, [{"n": 1}, {"n": 2}, {"n": 3}])
    assert len(col) == 3
    got = col.get(["b", "zzz"], include_vectors=True)
    assert got[0]["id"] == "b" and got[0]["metadata"] == {"n": 2}
    assert np.allclose(np.linalg.norm(got[0]["vector"]), 1.0)  # cosine stores unit vectors
    assert col.delete(["b", "nope"]) == 1
    assert "b" not in col and len(col) == 2


def test_overwrite_moves_record(tmp_path):
    col = make(tmp_path, seal_threshold=8)
    col.upsert(ids(8), rand(8))  # seals
    assert col.stats()["growing"] == 0
    col.upsert(["r3"], rand(1, seed=9), [{"v": 2}])
    assert len(col) == 8
    assert col.get(["r3"])[0]["metadata"] == {"v": 2}
    assert col.stats()["sealed_segments"][0]["live"] == 7


def test_duplicate_ids_in_one_batch_keep_last(tmp_path):
    col = make(tmp_path)
    col.upsert(["x", "x"], rand(2), [{"v": 1}, {"v": 2}])
    assert len(col) == 1 and col.get(["x"])[0]["metadata"] == {"v": 2}


def test_validation(tmp_path):
    col = make(tmp_path)
    with pytest.raises(ValueError):
        col.upsert(["a"], rand(1, dim=8))
    with pytest.raises(ValueError):
        col.upsert(["a"], np.array([[np.nan] * 16]))
    with pytest.raises(ValueError):
        col.upsert(["a", "b"], rand(2), [{}])
    with pytest.raises(ValueError):
        col.search(rand(1, dim=8)[0])


def test_search_matches_brute_force_across_segments(tmp_path):
    col = make(tmp_path, seal_threshold=200)
    data = rand(1000)
    col.upsert(ids(1000), data)
    assert len(col.stats()["sealed_segments"]) == 5
    q = rand(1, seed=5)[0]
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    exact = [f"r{i}" for i in np.argsort(-(unit @ (q / np.linalg.norm(q))))[:10]]
    got = [h["id"] for h in col.search(q, 10, ef=200)]
    assert len(set(got) & set(exact)) >= 9
    scores = [h["score"] for h in col.search(q, 10)]
    assert scores == sorted(scores, reverse=True)


def test_filtered_search(tmp_path):
    col = make(tmp_path, seal_threshold=100)
    data = rand(300)
    metas = [{"group": "a" if i % 10 == 0 else "b", "i": i} for i in range(300)]
    col.upsert(ids(300), data, metas)
    hits = col.search(data[0], 50, filter={"group": "a"})
    assert len(hits) == 30 and all(h["metadata"]["group"] == "a" for h in hits)
    hits = col.search(data[0], 5, filter={"i": {"$lt": 3}})
    assert {h["id"] for h in hits} == {"r0", "r1", "r2"}


def test_recovery_without_close(tmp_path):
    col = make(tmp_path, seal_threshold=50)
    col.upsert(ids(120), rand(120))  # two sealed segments + 20 growing
    col.delete(["r5", "r110"])
    col.link([{"src": "r1", "dst": "r2", "rel": "cites"}])
    col.upsert(["r7"], rand(1, seed=3), [{"new": True}])
    expected = col.search(rand(1, seed=8)[0], 5)
    # simulate a crash: no close(), no checkpoint for the last writes
    reopened = Collection(tmp_path / "c")
    assert reopened.replayed_entries > 0
    assert len(reopened) == 118
    assert "r5" not in reopened and "r110" not in reopened
    assert reopened.get(["r7"])[0]["metadata"] == {"new": True}
    assert reopened.edges("r1")[0]["dst"] == "r2"
    assert [h["id"] for h in reopened.search(rand(1, seed=8)[0], 5)] == [h["id"] for h in expected]


def test_recovery_with_torn_wal_tail(tmp_path):
    col = make(tmp_path)
    col.upsert(["a"], rand(1))
    col.checkpoint()
    col.upsert(["b"], rand(1, seed=1))
    col.upsert(["c"], rand(1, seed=2))
    col.wal.close()
    wal = tmp_path / "c" / "wal.log"
    wal.write_bytes(wal.read_bytes()[:-3])
    reopened = Collection(tmp_path / "c")
    assert "a" in reopened and "b" in reopened and "c" not in reopened


def test_close_and_reopen_is_clean(tmp_path):
    col = make(tmp_path, seal_threshold=30)
    col.upsert(ids(70), rand(70))
    col.close()
    reopened = Collection(tmp_path / "c")
    assert reopened.replayed_entries == 0 and len(reopened) == 70
    with pytest.raises(RuntimeError):
        col.upsert(["x"], rand(1))


def test_orphan_segment_is_removed(tmp_path):
    col = make(tmp_path)
    col.upsert(ids(10), rand(10))
    col.close()
    orphan = tmp_path / "c" / "seg-99"
    orphan.mkdir()
    Collection(tmp_path / "c")
    assert not orphan.exists()


def test_compaction_drops_tombstones(tmp_path):
    col = make(tmp_path, seal_threshold=50)
    col.upsert(ids(100), rand(100))
    col.delete(ids(40))
    before = col.stats()["sealed_segments"]
    assert before[0]["deleted_ratio"] > 0.3
    result = col.compact()
    assert result["rewritten"] >= 1
    after = col.stats()
    assert all(s["deleted_ratio"] == 0 for s in after["sealed_segments"])
    assert after["count"] == 60
    reopened_hits = Collection(tmp_path / "c2", CollectionConfig(dim=16))  # unrelated, just ensure no clash
    assert len(reopened_hits) == 0
    col.close()
    assert len(Collection(tmp_path / "c")) == 60


def test_graph_edges_and_cascade(tmp_path):
    col = make(tmp_path)
    col.upsert(["a", "b", "c", "d"], rand(4))
    col.link([
        {"src": "a", "dst": "b", "rel": "knows", "weight": 0.5},
        {"src": "b", "dst": "c", "rel": "knows"},
        {"src": "c", "dst": "d", "rel": "owns"},
    ])
    assert [n["id"] for n in col.neighbors("a", hops=2)] == ["b", "c"]
    assert [n["id"] for n in col.neighbors("b", direction="out")] == ["c"]
    assert col.neighbors("c", rel="owns")[0]["id"] == "d"
    hydrated = col.neighbors("a", hydrate=True)
    assert hydrated[0]["metadata"] == {}
    col.delete(["b"])
    assert col.neighbors("a") == []
    assert col.unlink([{"src": "c", "dst": "d"}]) == 1
    assert col.stats()["edges"] == 0


def test_search_expand_hops(tmp_path):
    col = make(tmp_path)
    data = rand(5)
    col.upsert(ids(5), data)
    col.link([{"src": "r0", "dst": "r3"}])
    hit = col.search(data[0], 1, expand_hops=1)[0]
    assert hit["id"] == "r0" and hit["neighbors"][0]["id"] == "r3"


def test_graph_boost_pulls_in_linked_records(tmp_path):
    rng = np.random.default_rng(1)
    col = make(tmp_path, seal_threshold=1000)
    q = np.zeros(16, np.float32)
    q[0] = 1
    near = [q + 0.05 * rng.normal(size=16) for _ in range(5)]
    far = -q + 0.05 * rng.normal(size=16)  # opposite direction: never in the vector top-k
    noise = rng.normal(size=(60, 16))
    noise[:, 0] = -np.abs(noise[:, 0])
    col.upsert(["n0", "n1", "n2", "n3", "n4", "far"] + ids(60, "z"), np.vstack([near, far, noise]))
    col.link([{"src": f"n{i}", "dst": "far", "rel": "about"} for i in range(5)])
    plain = [h["id"] for h in col.search(q, 6)]
    boosted = col.search(q, 6, graph_boost=0.5)
    assert "far" not in plain
    assert "far" in [h["id"] for h in boosted]
    assert all("graph_score" in h for h in boosted)
    with pytest.raises(ValueError):
        col.search(q, 3, graph_boost=2)


def test_scan_with_filter(tmp_path):
    col = make(tmp_path)
    col.upsert(ids(10), rand(10), [{"even": i % 2 == 0} for i in range(10)])
    assert len(col.scan({"even": True})) == 5
    assert len(col.scan(limit=3)) == 3


def test_database(tmp_path):
    with Database(tmp_path / "db") as db:
        a = db.create_collection("docs", dim=16)
        a.upsert(["x"], rand(1))
        db.create_collection("other", dim=4, metric="l2")
        assert db.list_collections() == ["docs", "other"]
        with pytest.raises(ValueError):
            db.create_collection("../evil", dim=4)
        with pytest.raises(ValueError):
            db.create_collection("docs", dim=16, exist_ok=False)
        assert db.drop_collection("other")
        with pytest.raises(KeyError):
            db.collection("other")
    with Database(tmp_path / "db") as db:
        assert len(db["docs"]) == 1
        with pytest.raises(ValueError):
            db.create_collection("docs", dim=32)  # dim mismatch with stored config
