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
kb = db.create_collection("kb", dim=384)

kb.upsert(ids, vectors, metadata)                         # durable once it returns
kb.link([{"src": "ana", "dst": "voltcell", "rel": "works_at"}])

kb.search(q, k=5)                                         # nearest neighbours
kb.search(q, k=5, filter={"year": {"$gte": 2020}})        # with metadata filters
kb.search(q, k=5, expand_hops=1)                          # attach each hit's neighbourhood
kb.search(q, k=5, graph_boost=0.4)                        # re-rank by spreading activation over edges
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
  into an HNSW-indexed segment **on a background thread** while staying
  searchable. Deletes are tombstones; `compact()` rewrites heavy segments.
- **Crash recovery**: generation-numbered snapshots plus an atomically
  swapped manifest; opening a collection replays the WAL. Tested by
  reopening without `close()`, mid-build crashes and truncated logs.
- **Query planner**: small segments and selective filters use an exact scan
  (faster than a Python graph walk at that size, and exact); large ones use
  HNSW with the filter applied during the walk.
- **Metadata filters**: `$eq $ne $gt $gte $lt $lte $in $nin $contains $prefix
  $exists`, dotted paths, `$and $or $not`.
- **Graph layer**: typed, weighted edges with properties, N-hop neighbours,
  `expand_hops` and `graph_boost` in search, cascade on delete.
- **Server + client**: standard-library JSON/HTTP server with optional
  bearer-token auth; a client with the same method names as the embedded API.

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
    papers.delete(["p1"])                                       # also removes p1's edges
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
| POST | `/collections/<name>/search` | `{vector, k?, filter?, ef?, expand_hops?, graph_boost?, include_vectors?}` |
| POST | `/collections/<name>/get` · `delete` · `scan` | `{ids}` / `{ids}` / `{filter?, limit?}` |
| POST | `/collections/<name>/link` · `unlink` · `neighbors` | `{edges}` / `{edges}` / `{id, hops?, rel?, direction?}` |
| POST | `/collections/<name>/checkpoint` · `compact` | |

Maintenance from the shell: `stratadb stats|compact|checkpoint --path ./data [collection]`.

## Configuration

Set per collection at creation (`create_collection(name, dim, **options)`):

| option | default | meaning |
|---|---|---|
| `metric` | `cosine` | `cosine`, `ip` or `l2` |
| `m` / `ef_construction` | 16 / 128 | HNSW graph degree and build-time beam width |
| `ef_search` | 64 | default query beam width (override per query with `ef=`) |
| `seal_threshold` | 16384 | growing-segment size that triggers a seal |
| `background_seal` | `True` | build indexes on a worker thread |
| `exact_search_limit` | 3,000,000 | segments with records x dim at or under this are scanned exactly |
| `sync` | `flush` | WAL durability: `always` (fsync), `flush`, `never` |
| `wal_checkpoint_bytes` | 64 MB | checkpoint when the WAL grows past this |
| `compact_deleted_ratio` | 0.3 | tombstone ratio that makes a segment a compaction candidate |

## Benchmarks

BENCHMARKS_PLACEHOLDER

## How it works

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the write path, WAL
format, segment lifecycle, query planner, HNSW details, the checkpoint and
recovery protocol, and the spreading-activation re-ranker.

## Tests

```bash
pytest -q
```

The 67 tests cover HNSW recall on three metrics, filters, WAL corruption and
torn tails, crash recovery without `close()`, a crash in the middle of a
background index build, concurrent writers, compaction, the graph layer and
the HTTP API.

## Limits

- One process per collection directory (enforced with a lock file).
- Each collection is held in memory.
- Index builds are pure Python, roughly 1,000 vectors/second at 384 dims
  (in the background, so they don't block writes).

## License

MIT
