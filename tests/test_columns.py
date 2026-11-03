"""Vectorized filter masks must agree with the predicate on every row."""

import math

import numpy as np
import pytest

from stratadb.columns import ColumnStore
from stratadb.filters import FilterError, compile_filter

VALUES = [1, 1.0, True, False, 0, 2.5, -3, "1", "a", "abc", "ab", "", None, [1, 2], ["a"], [], {"k": 1},
          float("nan"), 10**12, "zz", ("t", 1)]


def awkward_metas(n=600, seed=0):
    rng = np.random.default_rng(seed)
    metas = []
    for _ in range(n):
        meta = {}
        if rng.random() < 0.9:
            meta["x"] = VALUES[rng.integers(len(VALUES))]
        if rng.random() < 0.7:
            meta["nested"] = {"y": VALUES[rng.integers(len(VALUES))]}
        if rng.random() < 0.5:
            meta["num"] = float(rng.normal()) if rng.random() < 0.5 else int(rng.integers(-5, 5))
        meta["lang"] = ["en", "de", "fr"][rng.integers(3)]
        metas.append(meta)
    return metas


FILTERS = [
    {"x": 1}, {"x": 1.0}, {"x": True}, {"x": "1"}, {"x": "a"}, {"x": None}, {"x": [1, 2]}, {"x": []},
    {"x": {"k": 1}}, {"x": float("nan")}, {"x": {"$ne": 1}}, {"x": {"$ne": "a"}}, {"x": {"$ne": [1, 2]}},
    {"x": {"$gt": 0}}, {"x": {"$gte": 1}}, {"x": {"$lt": "b"}}, {"x": {"$lte": "ab"}}, {"x": {"$gt": None}},
    {"x": {"$gt": [0]}}, {"x": {"$in": [1, "a", None]}}, {"x": {"$in": [[1, 2], "zz"]}}, {"x": {"$nin": [1, "a"]}},
    {"x": {"$in": []}}, {"x": {"$nin": []}}, {"x": {"$contains": "a"}}, {"x": {"$contains": 1}},
    {"x": {"$contains": "b"}}, {"x": {"$prefix": "a"}}, {"x": {"$prefix": ""}}, {"x": {"$prefix": 1}},
    {"x": {"$exists": True}}, {"x": {"$exists": False}}, {"x": {"$gte": 0, "$lt": 2}},
    {"nested.y": "abc"}, {"nested.y": {"$gt": 1}}, {"nested.y": {"$exists": True}}, {"nested": {"$exists": True}},
    {"num": {"$gt": 0.5}}, {"num": {"$lte": -1}}, {"num": 0}, {"num": {"$in": [0, 1, 2]}},
    {"lang": "en"}, {"lang": {"$in": ["en", "fr"]}}, {"lang": {"$nin": ["en"]}},
    {"$and": [{"lang": "en"}, {"x": {"$gt": 0}}]}, {"$or": [{"lang": "de"}, {"num": {"$lt": 0}}]},
    {"$not": {"lang": "en"}}, {"$not": {"x": {"$exists": True}}},
    {"$and": [{"$or": [{"lang": "en"}, {"lang": "de"}]}, {"$not": {"x": 1}}]},
    {"missing.field": {"$ne": 3}}, {"missing.field": {"$exists": False}},
]


@pytest.mark.parametrize("expr", FILTERS, ids=[str(f) for f in FILTERS])
def test_mask_matches_predicate(expr):
    metas = awkward_metas()
    store = ColumnStore(metas)
    pred = compile_filter(expr)
    want = np.array([pred(m) for m in metas], dtype=bool)
    got = store.mask(expr)
    assert got.dtype == bool and got.shape == want.shape
    assert np.array_equal(got, want), np.flatnonzero(got != want)[:10]


def test_empty_filter_and_errors():
    store = ColumnStore(awkward_metas(50))
    assert store.mask(None) is None and store.mask({}) is None
    for bad in ({"x": {"$bogus": 1}}, {"$nope": []}, {"$and": []}, {"x": {"$in": 3}}):
        with pytest.raises(FilterError):
            store.mask(bad)


def test_columns_are_built_once_and_lazily():
    store = ColumnStore(awkward_metas(50))
    assert store.fields == []
    store.mask({"lang": "en"})
    assert store.fields == ["lang"]
    col = store.column("lang")
    store.mask({"lang": {"$in": ["de"]}})
    assert store.column("lang") is col and store.fields == ["lang"]


def test_numeric_column_handles_nan_and_bools():
    metas = [{"v": float("nan")}, {"v": True}, {"v": 1}, {"v": 0.0}, {}]
    store = ColumnStore(metas)
    assert store.mask({"v": 1}).tolist() == [False, True, True, False, False]
    assert store.mask({"v": {"$gte": 0}}).tolist() == [False, True, True, True, False]
    assert store.mask({"v": {"$ne": 1}}).tolist() == [True, False, False, True, True]
    assert math.isnan(store.column("v").numeric[0])


def test_incremental_rows_track_appends_and_changes():
    metas = awkward_metas(100)
    store = ColumnStore(metas)
    exprs = [{"lang": "en"}, {"x": {"$gt": 0}}, {"nested.y": {"$exists": True}}, {"x": {"$in": ["a", 1]}}]

    def check():
        for expr in exprs:
            pred = compile_filter(expr)
            assert store.mask(expr).tolist() == [pred(m) for m in metas], expr

    check()
    metas.extend(awkward_metas(300, seed=7))      # appended rows (growing segment puts)
    check()
    for i in (0, 5, 150, 399):                     # overwritten and deleted rows
        metas[i] = {} if i % 2 else {"lang": "fr", "x": "a", "nested": {"y": 2}}
        store.invalidate_row(i)
    check()
    metas.append({"lang": "de", "x": 1})
    metas[3] = {"x": 0.5}
    store.invalidate_row(3)
    check()
    assert len(store.column("lang").codes) == len(metas)
