"""Metadata indexes must never change results, only speed.

Every test compares an indexed collection against an unindexed twin.
"""

import random

import numpy as np
import pytest

from stratadb import Collection, CollectionConfig
from stratadb.metaindex import MetaIndex


def twins(tmp_path, **kw):
    base = dict(dim=8, seal_threshold=60, background_seal=False, **kw)
    plain = Collection(tmp_path / "plain", CollectionConfig(**base))
    indexed = Collection(tmp_path / "indexed", CollectionConfig(index_fields=["kind", "year", "meta.page"], **base))
    return plain, indexed


def random_meta(rng):
    meta = {"kind": rng.choice(["a", "b", "c", 1, 1.0, True, None, ["list"]]),
            "year": rng.choice([2019, 2020, 2021.5, "2020", None, float("nan")]),
            "meta": {"page": rng.randint(0, 5)}}
    if rng.random() < 0.15:
        del meta["year"]
    return meta


FILTERS = [
    {"kind": "a"}, {"kind": 1}, {"kind": None}, {"kind": ["list"]}, {"kind": {"$in": ["a", "b"]}},
    {"kind": {"$eq": True}}, {"year": {"$gte": 2020}}, {"year": {"$gt": 2020, "$lte": 2021.5}},
    {"year": {"$lt": 2020}}, {"year": "2020"}, {"meta.page": {"$gte": 2, "$lt": 4}},
    {"kind": "b", "year": {"$gte": 2020}}, {"$and": [{"kind": {"$in": ["a", "c"]}}, {"meta.page": 3}]},
    {"$or": [{"kind": "a"}, {"year": 2019}]}, {"kind": {"$ne": "a"}}, {"year": {"$exists": False}},
    {"kind": "a", "$not": {"meta.page": 0}}, {"year": {"$gte": 2020, "$in": [2020, 2019]}},
]


@pytest.mark.parametrize("seed", range(4))
def test_index_matches_full_scan(tmp_path, seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    plain, indexed = twins(tmp_path)
    ids = [f"r{i}" for i in range(150)]
    for _ in range(12):  # inserts, overwrites and deletes across sealed + growing segments
        batch = rng.sample(ids, 20)
        metas = [random_meta(rng) for _ in batch]
        vecs = nrng.normal(size=(20, 8))
        for col in (plain, indexed):
            col.upsert(batch, vecs, metas)
        gone = rng.sample(ids, 5)
        for col in (plain, indexed):
            col.delete(gone)
    q = nrng.normal(size=8)
    for flt in FILTERS:
        assert {r["id"] for r in indexed.scan(flt)} == {r["id"] for r in plain.scan(flt)}, flt
        a = [h["id"] for h in indexed.search(q, 10, filter=flt, ef=400)]
        b = [h["id"] for h in plain.search(q, 10, filter=flt, ef=400)]
        assert a == b, flt
    indexed.close()
    reopened = Collection(tmp_path / "indexed")  # index rebuilt from records on open
    for flt in FILTERS:
        assert {r["id"] for r in reopened.scan(flt)} == {r["id"] for r in plain.scan(flt)}, flt


def test_candidates_narrow_and_fall_back():
    idx = MetaIndex(["kind", "year"])
    for i in range(10):
        idx.add(f"r{i}", {"kind": "a" if i < 3 else "b", "year": 2010 + i})
    assert idx.candidates({"kind": "a"}) == {"r0", "r1", "r2"}
    assert idx.candidates({"year": {"$gte": 2017}}) == {"r7", "r8", "r9"}
    assert idx.candidates({"kind": "b", "year": {"$lt": 2014}}) == {"r3"}
    assert idx.candidates({"$or": [{"kind": "a"}]}) is None      # can't narrow: caller scans
    assert idx.candidates({"other": 1}) is None
    idx.add("r0", {"kind": "b", "year": 2010})                    # overwrite moves r0
    assert idx.candidates({"kind": "a"}) == {"r1", "r2"}
    idx.remove("r1")
    assert idx.candidates({"kind": "a"}) == {"r2"}
