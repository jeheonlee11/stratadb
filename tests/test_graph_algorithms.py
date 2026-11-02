"""Graph algorithms checked against networkx on random graphs."""

import random

import numpy as np
import pytest

from stratadb import Collection, CollectionConfig
from stratadb.graph import GraphStore

nx = pytest.importorskip("networkx")
# networkx's pure-Python PageRank, so the reference doesn't need scipy.
nx_pagerank = getattr(nx.algorithms.link_analysis.pagerank_alg, "_pagerank_python", None)
needs_ref_pagerank = pytest.mark.skipif(nx_pagerank is None, reason="networkx pure-Python pagerank unavailable")


def random_graph(seed, n=30, m=70):
    rng = random.Random(seed)
    g = GraphStore()
    ref = nx.MultiDiGraph()
    nodes = [f"v{i}" for i in range(n)]
    for v in nodes[: n - 3]:                      # leave a few isolated-by-edges nodes with props
        ref.add_node(v)
    for v in nodes[n - 3:]:
        g.put_node(v, {"lonely": True})
        ref.add_node(v)
    for _ in range(m):
        a, b = rng.sample(nodes[: n - 3], 2)
        rel = rng.choice(["r1", "r2"])
        w = rng.choice([0.5, 1.0, 2.0])
        if g.get_edge(a, b, rel) is None:
            g.add(a, b, rel, w)
            ref.add_edge(a, b, key=rel, weight=w)
    for v in nodes[: n - 3]:
        if not g.degree(v):
            g.put_node(v, {})
    return g, ref


@needs_ref_pagerank
@pytest.mark.parametrize("seed", range(5))
def test_pagerank_matches_networkx(seed):
    g, ref = random_graph(seed)
    ours = g.pagerank()
    theirs = nx_pagerank(ref, weight="weight", tol=1e-12, max_iter=500)
    assert set(ours) == set(theirs)
    for node in theirs:
        assert ours[node] == pytest.approx(theirs[node], abs=1e-6)
    assert sum(ours.values()) == pytest.approx(1.0)


@needs_ref_pagerank
@pytest.mark.parametrize("seed", range(5))
def test_personalized_pagerank_matches_networkx(seed):
    g, ref = random_graph(seed)
    seeds = {"v1": 1.0, "v2": 3.0}
    ours = g.pagerank(personalization=seeds)
    theirs = nx_pagerank(ref, weight="weight", personalization=seeds, tol=1e-12, max_iter=500)
    for node in theirs:
        assert ours[node] == pytest.approx(theirs[node], abs=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_shortest_paths_and_components_match_networkx(seed):
    g, ref = random_graph(seed)
    und = nx.Graph(ref)
    rng = random.Random(seed + 100)
    nodes = sorted(und.nodes)
    for _ in range(25):
        a, b = rng.sample(nodes, 2)
        path = g.shortest_path(a, b, max_hops=50)
        if nx.has_path(und, a, b):
            assert path is not None and len(path) == nx.shortest_path_length(und, a, b)
            assert path[0]["from"] == a and path[-1]["to"] == b
            assert all(s1["to"] == s2["from"] for s1, s2 in zip(path, path[1:]))
        else:
            assert path is None
    ours = sorted(tuple(c) for c in g.components())
    theirs = sorted(tuple(sorted(c)) for c in nx.connected_components(und))
    assert ours == theirs


def test_directed_paths_and_limits():
    g = GraphStore()
    g.add("a", "b", "next")
    g.add("b", "c", "next")
    g.add("c", "d", "other")
    assert [s["to"] for s in g.shortest_path("a", "d")] == ["b", "c", "d"]
    assert g.shortest_path("d", "a", direction="out") is None
    assert g.shortest_path("a", "d", rel="next") is None
    assert g.shortest_path("a", "d", max_hops=2) is None
    assert g.shortest_path("a", "a") == []


def test_collection_api_ranks_seed_neighbourhood_first(tmp_path):
    col = Collection(tmp_path / "g", CollectionConfig(dim=0))
    # two clusters joined by one bridge
    col.link([{"src": f"a{i}", "dst": f"a{j}"} for i in range(5) for j in range(5) if i != j])
    col.link([{"src": f"b{i}", "dst": f"b{j}"} for i in range(5) for j in range(5) if i != j])
    col.link([{"src": "a0", "dst": "b0"}])
    top = [n for n, _ in col.pagerank(top=5, seeds=["a3"])]
    assert set(top) <= {f"a{i}" for i in range(5)} | {"b0"} and "a3" in top
    assert len(col.components()) == 1
    assert [s["to"] for s in col.shortest_path("a3", "b2")] == ["a0", "b0", "b2"]
    with pytest.raises(ValueError):
        col.pagerank(seeds=["nope"])
    assert np.isclose(sum(s for _, s in col.pagerank()), 1.0)
