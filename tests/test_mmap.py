import numpy as np
import pytest

from stratadb import Collection, CollectionConfig
from stratadb.hnsw import HNSWIndex


def test_reopened_sealed_segments_are_memory_mapped(tmp_path):
    data = np.random.default_rng(0).normal(size=(300, 16)).astype(np.float32)
    col = Collection(tmp_path / "c", CollectionConfig(dim=16, seal_threshold=100, background_seal=False,
                                                       build_processes=0, auto_compact=False))
    col.upsert([f"r{i}" for i in range(300)], data)
    before = [h["id"] for h in col.search(data[42], 5)]
    col.close()
    again = Collection(tmp_path / "c")
    seg = next(iter(again.sealed.values()))
    assert isinstance(seg.index._vectors.base, np.memmap) and seg.index._readonly
    assert (tmp_path / "c" / f"seg-{seg.seg_id}" / "vectors.npy").exists()
    assert [h["id"] for h in again.search(data[42], 5)] == before
    with pytest.raises(RuntimeError):
        seg.index.add(data[0])
    again.delete(["r42"])                       # tombstones work on a mapped segment
    assert "r42" not in [h["id"] for h in again.search(data[42], 5)]
    again.compact()                             # rewrite reads from the mapped vectors
    assert len(again) == 299 and again.search(data[43], 1)[0]["id"] == "r43"
    again.close()


def test_old_single_file_index_format_still_loads(tmp_path):
    idx = HNSWIndex(8, metric="l2")
    data = np.random.default_rng(1).normal(size=(50, 8)).astype(np.float32)
    idx.add_batch(data)
    idx.save(tmp_path / "index.npz")            # vectors inside the npz, as 0.3.0 wrote them
    loaded = HNSWIndex.load(tmp_path / "index.npz", mmap=True)
    assert not loaded._readonly and loaded.search(data[7], 1)[0][0] == 7
    loaded.add(data[0])                         # still writable
