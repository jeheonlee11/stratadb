"""The native (C) HNSW index against the pure-Python one: same results, same files."""

import threading

import numpy as np
import pytest

from stratadb import native
from stratadb.hnsw import NativeHNSWIndex, PyHNSWIndex

pytestmark = pytest.mark.skipif(not native.available(), reason=native.status())


def clustered(n, dim, clusters=20, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    return (centers[rng.integers(0, clusters, n)] + 0.3 * rng.normal(size=(n, dim))).astype(np.float32)


def recall(index, queries, k=10, ef=64):
    total = 0
    for q in queries:
        approx = {label for label, _ in index.search(q, k, ef=ef)}
        exact = {label for label, _ in index.brute_force(q, k)}
        total += len(approx & exact)
    return total / (k * len(queries))


@pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
def test_recall_matches_python_index(metric):
    data = clustered(3000, 32)
    queries = clustered(50, 32, seed=1)
    nat = NativeHNSWIndex(32, metric=metric, m=12, ef_construction=100)
    nat.add_batch(data)
    py = PyHNSWIndex(32, metric=metric, m=12, ef_construction=100)
    py.add_batch(data)
    assert recall(nat, queries, ef=64) >= recall(py, queries, ef=64) - 0.03
    assert recall(nat, queries, ef=128) >= 0.9
    # ef is honoured below the default of 64 too (a small ef must cost recall).
    assert recall(nat, queries, ef=10, k=10) < recall(nat, queries, ef=200, k=10)
    assert recall(nat, queries, ef=10, k=10) == pytest.approx(recall(py, queries, ef=10, k=10), abs=0.1)


@pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
def test_brute_force_matches_numpy(metric):
    data = clustered(500, 16)
    idx = NativeHNSWIndex(16, metric=metric)
    idx.add_batch(data)
    q = clustered(1, 16, seed=3)[0]
    stored = idx.vectors
    qq = q / np.linalg.norm(q) if metric == "cosine" else q
    if metric == "l2":
        exact = ((stored - qq) ** 2).sum(axis=1)
    elif metric == "ip":
        exact = -(stored @ qq)
    else:
        exact = 1 - stored @ qq
    got = idx.brute_force(q, 5)
    assert [label for label, _ in got] == np.argsort(exact)[:5].tolist()
    assert np.allclose([d for _, d in got], np.sort(exact)[:5], atol=1e-5)


def test_exact_match_deletes_and_filters():
    data = clustered(800, 16)
    idx = NativeHNSWIndex(16)
    idx.add_batch(data)
    assert idx.search(data[123], 1)[0][0] == 123
    idx.mark_deleted(123)
    assert idx.is_deleted(123) and idx.live_count == 799 and idx.deleted_labels() == [123]
    assert all(label != 123 for label, _ in idx.search(data[123], 20))
    allowed = {3, 500, 799}
    assert {label for label, _ in idx.search(data[0], 3, allow_labels=allowed)} == allowed
    assert {label for label, _ in idx.search(data[0], 3, allow_labels=np.array(sorted(allowed)))} == allowed
    assert {label for label, _ in idx.search(data[0], 3, allow=lambda l: l in allowed)} == allowed
    assert {label for label, _ in idx.brute_force(data[0], 3, allow_labels=allowed)} == allowed
    # Both restrictions at once: intersection.
    assert [label for label, _ in idx.search(data[0], 5, allow=lambda l: l != 3, allow_labels=allowed)] \
        == [label for label, _ in idx.brute_force(data[0], 5, allow_labels={500, 799})]


def test_empty_and_dim_checks():
    idx = NativeHNSWIndex(4)
    assert idx.search([1, 0, 0, 0], 5) == [] and idx.brute_force([1, 0, 0, 0], 5) == []
    assert len(idx) == 0 and idx.entry_point is None and idx.vectors.shape == (0, 4)
    with pytest.raises(ValueError):
        idx.add([1, 2, 3])
    with pytest.raises(ValueError):
        idx.add_batch(np.zeros((3, 5)))
    idx.add([1, 2, 3, 4])
    with pytest.raises(ValueError):
        idx.search([1, 2, 3], 1)
    assert idx.add_batch(np.zeros((0, 4))) == []


def test_capacity_growth_keeps_results_consistent():
    data = clustered(2500, 8)
    idx = NativeHNSWIndex(8, capacity=1)       # forces many reallocations
    labels = idx.add_batch(data[:1000])
    labels += idx.add_batch(data[1000:])
    assert labels == list(range(2500))
    assert np.allclose(idx.vectors, data / np.linalg.norm(data, axis=1, keepdims=True), atol=1e-6)
    assert idx.search(data[2499], 1)[0][0] == 2499


@pytest.mark.parametrize("writer,reader", [(PyHNSWIndex, NativeHNSWIndex), (NativeHNSWIndex, PyHNSWIndex),
                                           (NativeHNSWIndex, NativeHNSWIndex)])
@pytest.mark.parametrize("split_vectors", [False, True])
def test_files_are_interchangeable(tmp_path, writer, reader, split_vectors):
    data = clustered(600, 16)
    idx = writer(16, metric="l2", m=8, ef_construction=64)
    idx.add_batch(data)
    idx.mark_deleted(5)
    idx.mark_deleted(77)
    path = tmp_path / "index.npz"
    idx.save(path, tmp_path / "vectors.npy" if split_vectors else None)
    loaded = reader.load(path)
    assert len(loaded) == 600 and loaded.metric == "l2" and loaded.m == 8
    assert loaded.is_deleted(5) and loaded.is_deleted(77) and loaded.deleted_labels() == [5, 77]
    assert loaded.entry_point == idx.entry_point and loaded.max_level == idx.max_level
    for q in data[:20]:
        got, want = loaded.search(q, 5, ef=64), idx.search(q, 5, ef=64)
        assert [label for label, _ in got] == [label for label, _ in want]
        assert [d for _, d in got] == pytest.approx([d for _, d in want], rel=1e-5)
    # The graph itself is identical: a second save produces the same arrays.
    again = tmp_path / "again.npz"
    loaded.save(again, tmp_path / "again.npy" if split_vectors else None)
    with np.load(path) as a, np.load(again) as b:
        for key in ("levels", "links", "offsets", "deleted", "params"):
            assert np.array_equal(a[key], b[key]), key


def test_mmap_load_is_read_only_and_shares_the_mapping(tmp_path):
    data = clustered(300, 8)
    idx = NativeHNSWIndex(8)
    idx.add_batch(data)
    idx.save(tmp_path / "index.npz", tmp_path / "vectors.npy")
    loaded = NativeHNSWIndex.load(tmp_path / "index.npz", mmap=True)
    assert loaded._readonly and isinstance(loaded._vectors.base, np.memmap)
    assert loaded.search(data[7], 1)[0][0] == 7
    with pytest.raises(RuntimeError):
        loaded.add(data[0])
    loaded.mark_deleted(7)                     # tombstones still work
    assert loaded.search(data[7], 1)[0][0] != 7


def test_concurrent_searches_are_safe():
    data = clustered(5000, 16)
    idx = NativeHNSWIndex(16)
    idx.add_batch(data)
    expected = [idx.search(q, 5, ef=64) for q in data[:200]]
    errors = []

    def worker():
        try:
            for _ in range(3):
                got = [idx.search(q, 5, ef=64) for q in data[:200]]
                assert got == expected
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_native_is_the_default_when_loaded():
    from stratadb.hnsw import HNSWIndex

    assert HNSWIndex is NativeHNSWIndex
    assert "loaded" in native.status()
