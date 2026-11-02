import numpy as np
import pytest

from stratadb.hnsw import HNSWIndex


def clustered(n, dim, clusters=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    return (centers[rng.integers(0, clusters, n)] + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)


def recall_at_k(index, queries, k=10, ef=64):
    total = 0
    for q in queries:
        approx = {label for label, _ in index.search(q, k, ef=ef)}
        exact = {label for label, _ in index.brute_force(q, k)}
        total += len(approx & exact)
    return total / (k * len(queries))


@pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
def test_recall_is_high(metric):
    data = clustered(2000, 32)
    idx = HNSWIndex(32, metric=metric, m=12, ef_construction=100)
    idx.add_batch(data)
    queries = clustered(50, 32, seed=1)
    assert recall_at_k(idx, queries, ef=128) >= 0.9


def test_exact_match_is_first():
    data = clustered(500, 16)
    idx = HNSWIndex(16)
    idx.add_batch(data)
    for i in (0, 123, 499):
        label, dist = idx.search(data[i], 1)[0]
        assert label == i and dist == pytest.approx(0, abs=1e-5)


def test_deleted_never_returned():
    data = clustered(300, 8)
    idx = HNSWIndex(8)
    idx.add_batch(data)
    idx.mark_deleted(7)
    assert all(label != 7 for label, _ in idx.search(data[7], 20))
    assert idx.live_count == 299


def test_filtered_search_finds_rare_matches():
    data = clustered(1000, 16)
    idx = HNSWIndex(16)
    idx.add_batch(data)
    allowed = {3, 500, 999}
    hits = idx.search(data[0], 3, allow=lambda label: label in allowed)
    assert {label for label, _ in hits} == allowed


def test_save_load_roundtrip(tmp_path):
    data = clustered(400, 16)
    idx = HNSWIndex(16, metric="l2")
    idx.add_batch(data)
    idx.mark_deleted(5)
    path = tmp_path / "index.npz"
    idx.save(path)
    loaded = HNSWIndex.load(path)
    assert len(loaded) == 400 and loaded.metric == "l2" and loaded.is_deleted(5)
    q = data[42]
    assert loaded.search(q, 5) == idx.search(q, 5)


def test_empty_and_dim_checks():
    idx = HNSWIndex(4)
    assert idx.search([1, 0, 0, 0], 5) == []
    with pytest.raises(ValueError):
        idx.add([1, 2, 3])
