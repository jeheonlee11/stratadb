# StrataDB

An embedded, **graph-native vector database** in pure Python + numpy.

StrataDB stores vectors, JSON metadata **and typed, weighted edges between
records** in one place, so retrieval can follow relationships instead of
stopping at the nearest neighbours. It is built from the ground up: its own
HNSW index, write-ahead log, segment lifecycle, crash recovery, filter
engine, HTTP server and client. The only dependency is numpy.

```python
from stratadb import Database

db = Database("./data")
kb = db.create_collection("kb", dim=384, text_field="text")

kb.upsert(ids, vectors, metadata)                         # durable once it returns
kb.link([{"src": "ana", "dst": "voltcell", "rel": "works_at"}])

kb.search(q, k=5)                                         # nearest neighbours
kb.search(q, k=5, filter={"year": {"$gte": 2020}})        # with metadata filters
kb.search(q, k=5, expand_hops=1)                          # attach each hit's neighbourhood
kb.search(q, k=5, graph_boost=0.4)                        # re-rank by spreading activation over edges
kb.search(q, text="E1042 checksum", k=5)                  # hybrid: vector + BM25 keyword, fused
kb.search_many(queries, k=5)                               # many queries, one call
```

## Why graph-native

Vector search answers "what text looks like my question?". Many questions
need one more step: *who works at* the companies that match, *which papers
cite* the ones that match. In a plain vector store that step means a second
database and glue code.

In StrataDB, `graph_boost` runs spreading activation over the edges: records
that clearly match pass activation along their edges, and linked records,
including ones the vector search missed entirely, move up. From
[`examples/graph_retrieval.py`](examples/graph_retrieval.py):

```
Q: people employed by a battery manufacturer

vector search only:
  0.61  gridstore GridStore is a battery manufacturer for solar farms.
  0.58  voltcell  VoltCell is a battery manufacturer supplying electric car makers.
  0.21  ana       Ana joined VoltCell in 2021 as a mechanical engineer.
  0.19  bakery    Crumb & Co is a bakery chain famous for sourdough.

vector search + graph_boost=0.4:
  0.49  ana       (vector 0.21, graph 0.91)  Ana joined VoltCell in 2021 as a mechanical engineer.
  0.40  ben       (vector 0.00, graph 1.00)  Ben leads firmware at GridStore.
  0.37  gridstore (vector 0.61, graph 0.00)  GridStore is a battery manufacturer for solar farms.
  0.35  voltcell  (vector 0.58, graph 0.00)  VoltCell is a battery manufacturer supplying electric car makers.
```

Ben's description shares no words with the question; he is found only
through the `works_at` edge.

Only candidates that score above the pool's background level spread
activation, so a weak, barely related match can't promote its neighbours.

## Features

- **HNSW index from scratch**: layered graph, heuristic neighbour selection,
  cosine / inner product / L2. Recall matches hnswlib within about one point
  at the same parameters (see [Benchmarks](#benchmarks)).
- **Durable writes**: CRC-framed write-ahead log with raw float32 vectors;
  torn or corrupt tails are detected and truncated on open.
- **Segments**: writes land in a brute-force growing segment, which seals
  into an HNSW-indexed segment **in a background worker process** while
  staying searchable. Several segments build at once, so bulk loads scale
  with cores: 4 workers index 20k x 384 vectors 4x faster than one. Deletes are tombstones; segments heavy with them are compacted
  automatically in the background (the rebuild runs outside the lock, so
  reads and writes never wait for it).
- **Crash recovery**: generation-numbered snapshots plus an atomically
  swapped manifest; opening a collection replays the WAL. Tested by
  reopening without `close()`, mid-build crashes and truncated logs.
- **Query planner**: small segments and selective filters use an exact scan
  (faster than a Python graph walk at that size, and exact); large ones use
  HNSW with the filter applied during the walk.
- **Metadata filters**: `$eq $ne $gt $gte $lt $lte $in $nin $contains $prefix
  $exists`, dotted paths, `$and $or $not`.
- **Secondary indexes** (`index_fields`): hash indexes for equality / `$in`
  and sorted arrays for numeric ranges narrow filtered searches before any
  record is checked; when the indexes answer the whole filter, the per-record
  check is skipped entirely. Up to 60x faster filtered search.
- **Graph layer**: typed, weighted edges with properties, node properties,
  N-hop neighbours, `expand_hops` and `graph_boost` in search, cascade on
  delete. Graph-only collections (`dim=0`) hold a knowledge graph with no
  vectors at all.
- **Graph algorithms**: shortest paths (typed / directed), weighted PageRank
  and personalized PageRank (verified against networkx to 1e-6), connected
  components.
- **Hybrid search**: a BM25 keyword index over one metadata field, fused with
  vector results by reciprocal rank fusion. Catches exact tokens (error codes,
  function names, part numbers) that embeddings blur.
- **Server, client and console**: standard-library JSON/HTTP server with
  optional bearer-token auth, a client with the same method names as the
  embedded API, and a built-in web console (collections, filtered scans,
  keyword search, "more like this", graph explorer).
- **Online backups**: a point-in-time copy of a live database. **JSONL
  export/import** of records, nodes and edges for moving data between
  versions or tools.

## Install

```bash
pip install -e .            # Python 3.10+, numpy is the only dependency
pip install -e ".[dev]"     # + pytest
```

## Use it embedded

```python
import numpy as np
from stratadb import Database

with Database("./data") as db:
    papers = db.create_collection("papers", dim=64)            # metric="cosine" | "ip" | "l2"
    papers.upsert(["p1", "p2"], np.random.rand(2, 64), [{"year": 2021}, {"year": 2024}])
    papers.link([{"src": "p2", "dst": "p1", "rel": "cites", "weight": 1.0}])

    hits = papers.search(np.random.rand(64), k=5, filter={"year": {"$gte": 2022}}, expand_hops=1)
    papers.neighbors("p2", hops=2, rel="cites", hydrate=True)
    papers.update_metadata(["p2"], {"year": 2025, "draft": None})  # merge; None removes a key
    papers.count({"year": {"$gte": 2022}})
    papers.scan({"year": {"$gte": 2020}}, limit=50, offset=50)   # id order, paged
    papers.delete_where({"year": {"$lt": 2015}})
    papers.delete(["p1"])                                       # also removes p1's edges

    # Hybrid search: index a metadata field for BM25 keyword matching
    docs = db.create_collection("docs", dim=64, text_field="text")
    docs.upsert(["d1"], np.random.rand(1, 64), [{"text": "Error E1042: WAL checksum failed"}])
    docs.search(text="E1042", k=5)                             # keyword only
    docs.search(np.random.rand(64), text="checksum", k=5)      # fused with the vector ranking

    # A knowledge graph with no vectors: nodes with properties, typed edges
    kg = db.create_graph("kg")
    kg.put_nodes([{"id": "Ada", "props": {"type": "person"}}, {"id": "Engine", "props": {"type": "machine"}}])
    kg.link([{"src": "Ada", "dst": "Engine", "rel": "wrote_about"}])
    kg.neighbors("Ada", hydrate=True)
    kg.shortest_path("Ada", "Engine")                          # [{"from", "to", "rel", "weight"}, ...]
    kg.pagerank(top=10, seeds=["Ada"])                         # personalized PageRank around Ada
    kg.components()
```

## Run it as a server

```bash
stratadb serve --path ./data --port 7700 --api-key dev-key
```

```python
from stratadb.client import Client

client = Client("http://127.0.0.1:7700", api_key="dev-key")
notes = client.create_collection("notes", dim=384)
notes.upsert(ids, vectors, metadata)
notes.search(query, k=5, graph_boost=0.3)
```

| Method | Path | Body |
|---|---|---|
| GET | `/health` | |
| GET / POST | `/collections` | `{name, dim, metric?, ...options}` |
| GET / DELETE | `/collections/<name>` | |
| POST | `/collections/<name>/upsert` | `{ids, vectors, metadata?}` |
| POST | `/collections/<name>/search_many` | `{vectors, k?, ...same options}` |
| POST | `/collections/<name>/search` | `{vector?, text?, k?, filter?, ef?, expand_hops?, graph_boost?, include_vectors?}` |
| POST | `/collections/<name>/get` · `delete` · `scan` · `count` | `{ids}` / `{ids}` / `{filter?, limit?, offset?}` / `{filter?}` |
| POST | `/collections/<name>/update` · `delete_where` | `{ids, patch, replace?}` / `{filter}` |
| POST | `/collections/<name>/link` · `unlink` · `neighbors` | `{edges}` / `{edges}` / `{id, hops?, rel?, direction?}` |
| POST | `/collections/<name>/nodes` · `get_nodes` · `list_nodes` | `{nodes: [{id, props}]}` / `{ids}` / `{limit?, offset?, prefix?}` |
| POST | `/collections/<name>/path` · `pagerank` · `components` | `{src, dst, max_hops?}` / `{top?, seeds?}` / |
| POST | `/collections/<name>/checkpoint` · `compact` | |

Open `http://127.0.0.1:7700/` for the **web console**: browse collections and their stats,
run filtered scans and keyword searches, find records similar to any stored record (using its
own vector, so no embedding model is needed), explore the graph around a node, and rank
nodes by PageRank or list connected components.

Maintenance from the shell: `stratadb stats|compact|checkpoint --path ./data [collection]`, and
`stratadb backup --path ./data --out ./backup` for a consistent online copy, and
`stratadb export|import --path ./data <collection> --out|--in file.jsonl` for portable dumps.

## Configuration

Set per collection at creation (`create_collection(name, dim, **options)`):

| option | default | meaning |
|---|---|---|
| `metric` | `cosine` | `cosine`, `ip` or `l2` |
| `m` / `ef_construction` | 16 / 128 | HNSW graph degree and build-time beam width |
| `ef_search` | 64 | default query beam width (override per query with `ef=`) |
| `seal_threshold` | 16384 | growing-segment size that triggers a seal |
| `background_seal` | `True` | build indexes in the background instead of blocking the write |
| `build_processes` | 2 | worker processes for index builds (and how many build at once); 0 = in-process |
| `build_process_min_records` | 2000 | smaller segments build in-process (a worker costs ~1 s to start) |

Worker processes start with `spawn`, which re-imports your main script, so a
script that writes to StrataDB needs the standard `if __name__ == "__main__":`
guard (as with any use of `multiprocessing`). In a REPL or notebook, where there
is no script to re-import, StrataDB notices the failed start and builds
in-process instead.
| `exact_search_limit` | 3,000,000 | segments with records x dim at or under this are scanned exactly |
| `sync` | `flush` | WAL durability: `always` (fsync), `flush`, `never` |
| `wal_checkpoint_bytes` | 64 MB | checkpoint when the WAL grows past this |
| `compact_deleted_ratio` | 0.3 | tombstone ratio that makes a segment a compaction candidate |
| `auto_compact` | `True` | compact in the background when a segment reaches that ratio |
| `text_field` | none | metadata field to index for keyword / hybrid search |
| `index_fields` | none | metadata fields (dotted paths allowed) with secondary indexes |

## Benchmarks

All numbers: Apple Silicon laptop, Python 3.11, single process.

**Real text embeddings.** 20,000 paragraphs of Python docstrings embedded
with `BAAI/bge-small-en-v1.5` (384 dims), 200 held-out paragraphs as
queries, recall against exact cosine top-10
([`bench/real_embeddings.py`](bench/real_embeddings.py)):

| search | recall@10 | p50 | p95 |
|---|---|---|---|
| HNSW, ef=32 | 0.978 | 0.48 ms | 0.77 ms |
| HNSW, ef=64 | 0.990 | 0.78 ms | 1.15 ms |
| HNSW, ef=128 | **0.999** | 1.53 ms | 2.10 ms |
| exact scan (planner's small-segment path) | 1.000 | 6.8 ms | 8.6 ms |

- Writes are acknowledged at about **260,000 vectors/s** (WAL append + growing
  segment). Index builds run in worker processes, several at a time
  (`seal_threshold=5000`, so four segments):

  | `build_processes` | 20k x 384 indexed in |
  |---|---|
  | 0 (in-process) | 30.0 s |
  | 2 | 15.3 s |
  | 4 | **7.6 s** |
- On disk: 33 MB for 20k x 384 vectors with index. Reopening takes under 0.1 s.

**Scaling to 100k vectors** (embedding-like synthetic data, 384 dims,
`bench/ann_bench.py --data manifold --n 100000`):

| segments | recall@10 (ef=64 / 128) | p50 (ef=64 / 128) | build | disk |
|---|---|---|---|---|
| 7 x 16k (`seal_threshold=16384`, default) | 0.987 / **0.999** | 8.8 ms / 15.6 ms | 225 s | 169 MB |
| 2 x 50k (`seal_threshold=50000`) | 0.934 / 0.986 | **2.7 ms / 4.7 ms** | 256 s | 169 MB |

A query walks every segment's graph, so fewer, larger segments are about 3x
faster per query; smaller segments give higher recall at the same `ef` and
cheaper, more frequent background builds. Pick `seal_threshold` for the
workload: large for read-heavy, small for write-heavy.

**Filtered search with secondary indexes** (same 20k vectors, `index_fields=["src", "year"]`):

| filter | no index | indexed |
|---|---|---|
| equality, 0.5% of records match | 4.25 ms | **0.07 ms** |
| numeric range, 40% match | 13.2 ms | **3.75 ms** |
| equality + range, 0.2% match | 10.4 ms | **0.59 ms** |

A differential test runs every filter shape against indexed and unindexed
twins (with tricky values: `1` vs `1.0` vs `True`, `None`, lists, NaN, numeric
strings) and requires identical results.

**Index quality vs. a reference implementation.** Same vectors and
parameters (`M=16, ef_construction=128`, 20k x 384) through StrataDB's HNSW
and [hnswlib](https://github.com/nmslib/hnswlib) (C++):

| data | ef | StrataDB recall@10 | hnswlib recall@10 |
|---|---|---|---|
| embedding-like (low intrinsic dimension) | 64 | 0.959 | 0.950 |
| embedding-like | 128 | 0.993 | 0.991 |
| hard: isotropic noise inside clusters | 128 | 0.687 | 0.707 |

Recall matches within about a point. Builds are about 25-40x slower than
hnswlib, the cost of pure Python, which is why they run in the background.

Reproduce the synthetic runs with `python bench/ann_bench.py --data manifold`
(or `--data clustered`).

## How it works

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the write path, WAL
format, segment lifecycle, query planner, HNSW details, the checkpoint and
recovery protocol, and the spreading-activation re-ranker.

## Tests

```bash
pytest -q
```

The 128 tests cover HNSW recall on three metrics, filters, WAL corruption and
torn tails, crash recovery without `close()`, a crash in the middle of a
background index build, concurrent writers, compaction, hybrid search, the
graph layer and the HTTP API, plus a **crash-recovery fuzzer**: random
operation histories, the WAL cut at random byte offsets, and a check that
the reopened collection equals the state after some prefix of the
operations. (It found a real bug: batches that crossed the seal threshold
weren't atomic.)

## Limits

- One process per collection directory (enforced with a lock file).
- Growing segments, metadata, the graph and the HNSW link structure are held in
  memory; sealed segments' vectors are memory-mapped from disk, so RAM for a
  large read-mostly collection is well below its on-disk size.
- Index builds are pure Python, roughly 700 vectors/second per core at 384
  dims; they run in background worker processes, so they don't block writes
  and scale with cores.

## License

MIT
