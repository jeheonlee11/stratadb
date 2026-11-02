"""Recall / latency benchmark for StrataDB against exact (brute-force) search.

    python bench/ann_bench.py --n 20000 --dim 384 --queries 200

Two synthetic datasets:

- ``manifold`` (default): points near a 32-dim subspace of the full space,
  like real text embeddings, which have low intrinsic dimension.
- ``clustered``: Gaussian clusters with isotropic noise in every dimension.
  Neighbours inside a cluster are nearly equidistant, which is close to the
  worst case for any graph index (hnswlib scores about the same on it).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from stratadb import Collection, CollectionConfig


def clustered(n, dim, clusters, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))
    labels = rng.integers(0, clusters, n)
    data = centers[labels] + 0.5 * rng.normal(size=(n, dim))
    return data.astype(np.float32), labels


def manifold(n, dim, seed, intrinsic=32):
    basis = np.random.default_rng(1234).normal(size=(intrinsic, dim))
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, intrinsic)) @ basis + 0.05 * rng.normal(size=(n, dim))
    return data.astype(np.float32), rng.integers(0, 50, n)


def exact_topk(data_unit, queries_unit, k):
    sims = queries_unit @ data_unit.T
    return np.argsort(-sims, axis=1)[:, :k]


def pct(values, p):
    return float(np.percentile(values, p) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seal", type=int, default=10000)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=128)
    ap.add_argument("--efs", default="16,32,64,128,256")
    ap.add_argument("--data", choices=["manifold", "clustered"], default="manifold")
    ap.add_argument("--procs", type=int, default=4, help="build_processes (worker processes for index builds)")
    ap.add_argument("--exact-limit", type=int, default=0,
                    help="segment exact-scan limit (records x dim); 0 forces the HNSW path")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    if args.data == "manifold":
        data, labels = manifold(args.n, args.dim, seed=0)
        queries, _ = manifold(args.queries, args.dim, seed=1)
    else:
        data, labels = clustered(args.n, args.dim, clusters=max(10, args.n // 500), seed=0)
        queries, _ = clustered(args.queries, args.dim, clusters=max(10, args.n // 500), seed=1)
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    qunit = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    truth = exact_topk(unit, qunit, args.k)
    tmp = Path(tempfile.mkdtemp(prefix="strata-bench-"))
    results = {"data": args.data, "build_processes": args.procs, "exact_limit": args.exact_limit, "n": args.n, "dim": args.dim, "queries": args.queries, "k": args.k, "m": args.m,
               "ef_construction": args.efc, "seal_threshold": args.seal}
    try:
        col = Collection(tmp / "bench", CollectionConfig(
            dim=args.dim, m=args.m, ef_construction=args.efc, seal_threshold=args.seal, sync="never",
            exact_search_limit=args.exact_limit, build_processes=args.procs))
        ids = [str(i) for i in range(args.n)]
        metas = [{"cluster": int(c), "even": i % 2 == 0} for i, c in enumerate(labels)]
        t0 = time.perf_counter()
        for start in range(0, args.n, 1000):
            col.upsert(ids[start:start + 1000], data[start:start + 1000], metas[start:start + 1000])
        col.seal()
        col.wait_for_seal()
        results["build_seconds"] = round(time.perf_counter() - t0, 2)
        results["inserts_per_second"] = round(args.n / results["build_seconds"])

        # Brute-force baseline latency (numpy matmul over all vectors).
        lat = []
        for q in qunit:
            t = time.perf_counter()
            np.argsort(-(unit @ q))[: args.k]
            lat.append(time.perf_counter() - t)
        results["brute_force_ms_p50"] = round(pct(lat, 50), 3)

        sweep = []
        for ef in [int(x) for x in args.efs.split(",")]:
            lat, hits = [], 0
            for qi, q in enumerate(queries):
                t = time.perf_counter()
                res = col.search(q, args.k, ef=ef)
                lat.append(time.perf_counter() - t)
                hits += len({int(r["id"]) for r in res} & set(truth[qi].tolist()))
            sweep.append({
                "ef": ef,
                "recall": round(hits / (args.k * args.queries), 4),
                "ms_p50": round(pct(lat, 50), 3),
                "ms_p95": round(pct(lat, 95), 3),
                "qps": round(len(lat) / sum(lat)),
            })
            print(json.dumps(sweep[-1]))
        results["sweep"] = sweep

        # Filtered search: half the records pass the filter.
        even = np.array([i % 2 == 0 for i in range(args.n)])
        even_idx = np.where(even)[0]
        ftruth = even_idx[np.argsort(-(qunit @ unit[even_idx].T), axis=1)[:, : args.k]]
        lat, hits = [], 0
        for qi, q in enumerate(queries):
            t = time.perf_counter()
            res = col.search(q, args.k, ef=128, filter={"even": True})
            lat.append(time.perf_counter() - t)
            hits += len({int(r["id"]) for r in res} & set(ftruth[qi].tolist()))
        results["filtered_50pct"] = {"recall": round(hits / (args.k * args.queries), 4),
                                     "ms_p50": round(pct(lat, 50), 3)}
        print("filtered", json.dumps(results["filtered_50pct"]))

        t0 = time.perf_counter()
        col.close()
        reopened = Collection(tmp / "bench")
        results["reopen_seconds"] = round(time.perf_counter() - t0, 2)
        assert len(reopened) == args.n
        reopened.close()
        results["disk_mb"] = round(sum(f.stat().st_size for f in (tmp / "bench").rglob("*") if f.is_file()) / 1e6, 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(json.dumps({k: v for k, v in results.items() if k != "sweep"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
