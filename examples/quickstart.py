"""Embedded StrataDB in a dozen lines: insert, filter, search, persist.

    python examples/quickstart.py
"""

import tempfile

import numpy as np

from stratadb import Database

rng = np.random.default_rng(0)
path = tempfile.mkdtemp(prefix="strata-quickstart-")

with Database(path) as db:
    papers = db.create_collection("papers", dim=64)
    vectors = rng.normal(size=(1000, 64))
    papers.upsert(
        ids=[f"paper-{i}" for i in range(1000)],
        vectors=vectors,
        metadata=[{"year": 2015 + i % 10, "venue": ["NeurIPS", "ICML", "ACL"][i % 3]} for i in range(1000)],
    )

    query = vectors[42] + 0.1 * rng.normal(size=64)
    print("top 3:")
    for hit in papers.search(query, k=3):
        print(f"  {hit['id']:<10} score={hit['score']:.3f} {hit['metadata']}")

    print("top 3 from ACL since 2020:")
    for hit in papers.search(query, k=3, filter={"venue": "ACL", "year": {"$gte": 2020}}):
        print(f"  {hit['id']:<10} score={hit['score']:.3f} {hit['metadata']}")

# Everything is on disk: reopen and keep going.
with Database(path) as db:
    print("reopened:", db["papers"].stats()["count"], "records")
