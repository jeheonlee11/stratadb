import numpy as np
import pytest

from stratadb import Collection, CollectionConfig


def make(tmp_path, **kw):
    cfg = dict(dim=8, seal_threshold=20, background_seal=False, auto_compact=False,
               text_field="text", index_fields=["tag"])
    cfg.update(kw)
    return Collection(tmp_path / "c", CollectionConfig(**cfg))


def seed(col, n=30):
    vecs = np.random.default_rng(0).normal(size=(n, 8))
    col.upsert([f"r{i:02d}" for i in range(n)], vecs,
               [{"tag": "even" if i % 2 == 0 else "odd", "text": f"item {i}", "n": i} for i in range(n)])
    return vecs


def test_update_metadata_merges_replaces_and_reindexes(tmp_path):
    col = make(tmp_path)
    vecs = seed(col)
    assert col.update_metadata(["r03", "r05", "missing"], {"tag": "special", "n": None, "new": 1}) == 2
    got = col.get(["r03"], include_vectors=True)[0]
    assert got["metadata"] == {"tag": "special", "text": "item 3", "new": 1}
    assert np.allclose(got["vector"], vecs[3] / np.linalg.norm(vecs[3]), atol=1e-6)  # vector untouched
    assert {r["id"] for r in col.scan({"tag": "special"})} == {"r03", "r05"}       # metadata index updated
    col.update_metadata(["r04"], {"text": "brand new words"}, replace=True)
    assert col.get(["r04"])[0]["metadata"] == {"text": "brand new words"}
    assert col.search(text="brand new", k=1)[0]["id"] == "r04"                       # BM25 index updated
    assert col.search(vecs[4], 1)[0]["id"] == "r04"                                   # still findable by vector
    col.close()
    again = Collection(tmp_path / "c")
    assert again.get(["r03"])[0]["metadata"]["tag"] == "special"
    with pytest.raises(ValueError):
        again.update_metadata(["r03"], ["not", "a", "dict"])


def test_delete_where_count_and_paging(tmp_path):
    col = make(tmp_path)
    seed(col)
    assert col.count() == 30 and col.count({"tag": "odd"}) == 15
    page1 = col.scan({"tag": "even"}, limit=4)
    page2 = col.scan({"tag": "even"}, limit=4, offset=4)
    assert [r["id"] for r in page1] == ["r00", "r02", "r04", "r06"]
    assert [r["id"] for r in page2] == ["r08", "r10", "r12", "r14"]
    assert [r["id"] for r in col.scan(limit=2, offset=29)] == ["r29"]
    assert col.delete_where({"n": {"$gte": 25}}) == 5
    assert col.count() == 25
    with pytest.raises(ValueError):
        col.delete_where({})
    with pytest.raises(ValueError):
        col.scan(offset=-1)


def test_search_many_matches_single_searches(tmp_path):
    col = make(tmp_path)
    vecs = seed(col)
    batch = col.search_many(vecs[:5], k=3, filter={"tag": "even"})
    assert len(batch) == 5
    for row, q in zip(batch, vecs[:5]):
        assert row == col.search(q, 3, filter={"tag": "even"})
    assert col.search_many(vecs[0], k=1)[0][0]["id"] == "r00"
    with pytest.raises(ValueError):
        col.search_many(np.zeros((2, 2, 8)))


def test_fields_projection(tmp_path):
    from stratadb.collection import project

    assert project({"a": 1, "b": {"c": 2, "d": 3}}, ["a", "b.c", "missing", "b.zz"]) == {"a": 1, "b": {"c": 2}}
    assert project({"a": 1}, None) == {"a": 1}
    col = make(tmp_path)
    vecs = seed(col)
    hit = col.search(vecs[3], 1, fields=["tag"])[0]
    assert hit["metadata"] == {"tag": "odd"} and hit["id"] == "r03"
    assert col.get(["r04"], fields=["n"])[0]["metadata"] == {"n": 4}
    assert col.scan({"tag": "even"}, limit=1, fields=["text"])[0]["metadata"] == {"text": "item 0"}
    assert col.get(["r04"])[0]["metadata"]["text"] == "item 4"   # stored metadata untouched
