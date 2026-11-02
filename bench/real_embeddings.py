"""StrataDB on real text embeddings.

    pip install fastembed
    python bench/real_embeddings.py --n 20000 --queries 200

Corpus: docstring paragraphs from the Python standard library and installed
packages (parsed with ``ast``; nothing is imported), embedded with
BAAI/bge-small-en-v1.5 (384 dims). Queries are held-out paragraphs; ground
truth is exact cosine top-k. Embeddings are cached in ~/.cache/stratadb-bench.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sysconfig
import tempfile
import time
from pathlib import Path

import numpy as np

from stratadb import Collection, CollectionConfig

CACHE = Path.home() / ".cache" / "stratadb-bench"


def docstring_paragraphs(limit: int) -> list[str]:
    roots = [Path(sysconfig.get_paths()["stdlib"]), Path(sysconfig.get_paths()["purelib"])]
    seen, out = set(), []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "test" in path.parts or "tests" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except (SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for para in (ast.get_docstring(node) or "").split("\n\n"):
                    para = " ".join(para.split())
                    if 60 <= len(para) <= 600 and para not in seen and ">>>" not in para:
                        seen.add(para)
                        out.append(para)
                        if len(out) >= limit:
                            return out
    return out


def embeddings(total: int) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"docstrings_bge_small_{total}.npy"
    if path.exists():
        return np.load(path)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from fastembed import TextEmbedding

    texts = docstring_paragraphs(total)
    print(f"embedding {len(texts)} paragraphs (one-time) ...", flush=True)
    vecs = np.array(list(TextEmbedding("BAAI/bge-small-en-v1.5").embed(texts, batch_size=256)), dtype=np.float32)
    np.save(path, vecs)
    return vecs


def pct(values, p):
    return round(float(np.percentile(values, p)) * 1000, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--efs", default="32,64,128")
    args = ap.parse_args()

    vecs = embeddings(args.n + args.queries)
    n = min(args.n, len(vecs) - args.queries)
    data, queries = vecs[:n], vecs[n:n + args.queries]
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    qunit = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    truth = np.argsort(-(qunit @ unit.T), axis=1)[:, : args.k]
    print(f"{n} vectors x {data.shape[1]} dims, {len(queries)} held-out queries")

    tmp = Path(tempfile.mkdtemp(prefix="strata-real-"))
    try:
        col = Collection(tmp / "c", CollectionConfig(dim=data.shape[1], seal_threshold=n, exact_search_limit=0))
        ids = [str(i) for i in range(n)]
        t = time.perf_counter()
        for s in range(0, n, 500):
            col.upsert(ids[s:s + 500], data[s:s + 500])
        ack = time.perf_counter() - t
        col.wait_for_seal()
        print(json.dumps({"writes_per_second": round(n / ack), "index_ready_seconds": round(time.perf_counter() - t, 1)}))

        for ef in [int(x) for x in args.efs.split(",")]:
            lat, hits = [], 0
            for qi, q in enumerate(queries):
                t = time.perf_counter()
                res = col.search(q, args.k, ef=ef)
                lat.append(time.perf_counter() - t)
                hits += len({int(r["id"]) for r in res} & set(truth[qi].tolist()))
            print(json.dumps({"ef": ef, "recall": round(hits / (args.k * len(queries)), 4),
                              "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95)}))
        col.close()
        size = sum(f.stat().st_size for f in (tmp / "c").rglob("*") if f.is_file()) / 1e6
        print(json.dumps({"disk_mb": round(size, 1)}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
