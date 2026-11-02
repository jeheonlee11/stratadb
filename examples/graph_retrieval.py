"""Why edges next to vectors help retrieval.

A tiny knowledge base: short facts about people and the companies they work
at. The question "people employed by a battery manufacturer" matches the
company facts, but the facts about the *people* never mention batteries, only
the company name. Plain vector search finds the companies and stops there;
graph_boost follows the works_at edges from those companies to their people.

Uses a toy bag-of-words embedder so it runs with no model download.

    python examples/graph_retrieval.py
"""

import re
import tempfile

import numpy as np

from stratadb import Database

DOCS = {
    "voltcell": "VoltCell is a battery manufacturer supplying electric car makers.",
    "gridstore": "GridStore is a battery manufacturer for solar farms.",
    "bakery": "Crumb & Co is a bakery chain famous for sourdough.",
    "ana": "Ana joined VoltCell in 2021 as a mechanical engineer.",
    "ben": "Ben leads firmware at GridStore.",
    "cara": "Cara manages three Crumb & Co stores downtown.",
    "dev": "Dev blogs about electric car reviews.",
}
EDGES = [("ana", "voltcell", "works_at"), ("ben", "gridstore", "works_at"), ("cara", "bakery", "works_at")]

vocab = sorted({w for text in DOCS.values() for w in re.findall(r"[a-z]+", text.lower())})


def embed(text: str) -> np.ndarray:
    words = re.findall(r"[a-z]+", text.lower())
    v = np.array([words.count(w) for w in vocab], dtype=np.float32)
    return v + 1e-3  # keep zero vectors out


with Database(tempfile.mkdtemp(prefix="strata-graph-")) as db:
    kb = db.create_collection("kb", dim=len(vocab))
    kb.upsert(list(DOCS), np.stack([embed(t) for t in DOCS.values()]), [{"text": t} for t in DOCS.values()])
    kb.link([{"src": a, "dst": b, "rel": rel} for a, b, rel in EDGES])

    question = "people employed by a battery manufacturer"
    q = embed(question)

    print(f"Q: {question}\n")
    print("vector search only:")
    for h in kb.search(q, k=4):
        print(f"  {h['score']:.2f}  {h['id']:<9} {h['metadata']['text']}")

    print("\nvector search + graph_boost=0.4:")
    for h in kb.search(q, k=4, graph_boost=0.4):
        print(f"  {h['score']:.2f}  {h['id']:<9} (vector {h['vector_score']:.2f}, graph {h['graph_score']:.2f})"
              f"  {h['metadata']['text']}")

    print("\nneighbours of voltcell:", [n["id"] for n in kb.neighbors("voltcell")])
