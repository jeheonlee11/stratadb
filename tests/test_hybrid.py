import numpy as np
import pytest

from stratadb import Collection, CollectionConfig
from stratadb.text_index import BM25Index, fuse_rrf, tokenize

DOCS = {
    "d1": "Error E1042 means the WAL checksum failed during recovery.",
    "d2": "Recovery replays the log after a crash to restore state.",
    "d3": "Use compaction to reclaim space from deleted vectors.",
    "d4": "The HNSW index trades recall for speed with the ef parameter.",
    "d5": "Checksum failures usually point at a torn write at the log tail.",
}


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("The error E1042 in wal.log is a torn-write") == ["error", "e1042", "wal.log", "torn-write"]


def test_bm25_ranks_rare_terms_highest():
    idx = BM25Index()
    for rid, text in DOCS.items():
        idx.add(rid, text)
    assert idx.search("E1042", 3)[0][1] == "d1"
    top = [rid for _, rid in idx.search("checksum log", 5)]
    assert top[0] == "d5" and set(top) == {"d1", "d2", "d5"}  # d5 is the only doc with both terms
    idx.remove("d1")
    assert all(rid != "d1" for _, rid in idx.search("E1042 checksum", 5))
    idx.add("d5", "replaced text about bananas")  # re-adding replaces old terms
    assert idx.search("checksum", 5) == []


def test_rrf_rewards_agreement():
    fused = fuse_rrf([["a", "b", "c"], ["b", "c", "a"]], k=3)
    assert fused[0][1] == "b"


def make(tmp_path, **kw):
    cfg = CollectionConfig(dim=8, text_field="text", background_seal=False, seal_threshold=3, **kw)
    return Collection(tmp_path / "c", cfg)


def embed(text):  # deterministic toy embedding
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=8)


def test_keyword_and_hybrid_search(tmp_path):
    col = make(tmp_path)
    ids = list(DOCS)
    col.upsert(ids, np.stack([embed(t) for t in DOCS.values()]),
               [{"text": t, "topic": "ops" if i % 2 else "core"} for i, t in enumerate(DOCS.values())])
    hits = col.search(text="E1042", k=3)
    assert hits[0]["id"] == "d1" and hits[0]["keyword_score"] > 0

    # Vector points at d4, keyword at d1: hybrid ranks both, each with its scores.
    hybrid = col.search(embed(DOCS["d4"]), text="E1042", k=5)
    top2 = {h["id"] for h in hybrid[:2]}
    assert top2 == {"d1", "d4"}
    assert all("keyword_score" in h for h in hybrid)

    filtered = col.search(text="checksum log recovery", k=5, filter={"topic": "ops"})
    assert filtered and all(h["metadata"]["topic"] == "ops" for h in filtered)


def test_text_index_follows_writes_and_recovery(tmp_path):
    col = make(tmp_path)
    col.upsert(["a", "b", "c", "d"], np.stack([embed(x) for x in "abcd"]),
               [{"text": "alpha zebra"}, {"text": "beta"}, {"text": "gamma zebra"}, {"text": "delta"}])
    col.upsert(["a"], embed("a2").reshape(1, -1), [{"text": "now about yaks"}])
    col.delete(["c"])
    assert col.search(text="zebra", k=5) == []
    assert col.search(text="yaks", k=5)[0]["id"] == "a"
    col.close()
    reopened = Collection(tmp_path / "c")  # sealed + growing records re-indexed on open
    assert reopened.search(text="yaks", k=5)[0]["id"] == "a"
    assert reopened.search(text="delta", k=5)[0]["id"] == "d"


def test_keyword_search_errors(tmp_path):
    plain = Collection(tmp_path / "p", CollectionConfig(dim=8))
    with pytest.raises(ValueError):
        plain.search(text="anything")
    with pytest.raises(ValueError):
        plain.search()
    col = make(tmp_path)
    with pytest.raises(ValueError):
        col.search(text="x", graph_boost=0.5)
