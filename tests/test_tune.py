import numpy as np
import pytest

from stratadb import Collection, CollectionConfig


def test_tune_ef_finds_smallest_ef_meeting_target(tmp_path):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(20, 32))
    data = (centers[rng.integers(0, 20, 3000)] + 0.3 * rng.normal(size=(3000, 32))).astype(np.float32)
    col = Collection(tmp_path / "c", CollectionConfig(dim=32, seal_threshold=3000, background_seal=False,
                                                       build_processes=0, exact_search_limit=0, m=8, ef_construction=40))
    col.upsert([f"r{i}" for i in range(3000)], data)
    col.seal(wait=True)
    res = col.tune_ef(target_recall=0.9, k=10, samples=30)
    assert res["recall"] >= 0.9 and res["samples"] == 30 and 10 <= res["ef"] <= 1024
    tried = dict(res["tried"])
    # every ef below the chosen one that was tried fell short of the target
    assert all(r < 0.9 for ef, r in tried.items() if ef < res["ef"])
    assert col.config.ef_search == 64            # not applied by default
    col.tune_ef(target_recall=0.9, samples=30, apply=True)
    assert col.config.ef_search == res["ef"]
    with pytest.raises(ValueError):
        col.tune_ef(target_recall=0)
    own = col.tune_ef(target_recall=0.9, queries=data[:20] + 0.1 * rng.normal(size=(20, 32)))
    assert own["samples"] == 20 and own["recall"] >= 0.9
    tiny = Collection(tmp_path / "t", CollectionConfig(dim=4))
    tiny.upsert(["a"], np.eye(4)[:1])
    assert tiny.tune_ef()["samples"] == 0
