# Changelog

## Unreleased

- `GET /metrics` in Prometheus text format: per-collection gauges (records,
  edges, nodes, WAL bytes, segments) and request/latency counters by action.
- `stratadb serve` shuts down cleanly on SIGTERM as well as Ctrl-C: final
  checkpoint, nothing to replay on the next start.
- Console: Maintenance tab (checkpoint, compact, merge, tune ef).

## 0.6.0

- `neighbors(filter=...)` keeps only neighbours whose metadata or node
  properties match; `graph_rels=[...]` limits which relation types
  `graph_boost` follows.
- `tune_ef(target_recall, queries=None)`: binary-searches the smallest `ef`
  that meets a recall target against exact search, using your own query
  vectors or synthetic ones pushed away from the stored data (a stored vector
  used as-is is far too easy a query).
- `fields=[...]` on `search`, `get` and `scan` returns only those metadata
  keys (dotted paths allowed); embedded, HTTP and client.

## 0.5.0

- `GET /collections/<name>/export` streams a JSON-lines dump; `RemoteCollection.export_jsonl`;
  an Export button in the console.
- Compaction and merge build their output segments concurrently in worker
  processes (merge(2) of 20k vectors: 14 s, the time of one build).
- `merge(max_segments)` rewrites sealed segments into fewer, larger ones for
  faster queries after a bulk load (`stratadb merge`, `POST .../merge`).

## 0.4.0

- JSONL export/import (`export_jsonl` / `import_jsonl`, `stratadb export|import`).
- Sealed segments' vectors are memory-mapped on reopen (stored in a separate
  `vectors.npy`); older single-file segments still load.
- Console: PageRank ranking and connected components in the Graph tab.

## 0.3.0

- **Parallel index builds**: segments build in worker processes
  (`build_processes`, default 2), several at a time, so bulk loads use
  multiple cores. Falls back to in-process builds where workers can't start.
- `search_many`: batch queries in one call, embedded and over HTTP.
- `update_metadata` (merge or replace, without resending vectors),
  `delete_where`, `count`, and id-ordered `scan` paging with `offset`.
- **Background auto-compaction**: segments past `compact_deleted_ratio` are
  rewritten on a worker thread; compaction no longer holds the lock while
  rebuilding, and reconciles deletes/overwrites made during the build.
- **Graph algorithms**: `shortest_path`, `pagerank` (incl. personalized),
  `components`, over HTTP too. Checked against networkx.
- **Secondary indexes** on metadata fields (`index_fields`): hash and
  numeric-range indexes narrow filtered search; exact plans skip the
  per-record filter check. 60x faster for selective equality filters.

## 0.2.0

- **Hybrid search**: BM25 keyword index over a metadata field (`text_field`),
  fused with vector results by reciprocal rank fusion.
- **Node properties and graph-only collections** (`dim=0`,
  `Database.create_graph`): a collection can hold a full knowledge graph.
- **Atomic batches**: each upsert is one WAL entry; found by the new
  crash-recovery fuzzer.
- **Online backups**: `Collection.backup`, `Database.backup`, `stratadb backup`.
- **Web console** served at `/`: stats, filtered scans, keyword search,
  "more like this", graph explorer. New `list_nodes` endpoint.
- **Process lock**: a second process opening the same collection fails fast.
- `graph_boost` only lets candidates above the pool's background similarity
  spread activation, so weak matches can't promote their neighbours.

## 0.1.0

- HNSW index from scratch (cosine / inner product / L2), heuristic neighbour
  selection, filtered search, npz persistence.
- CRC-framed write-ahead log with torn-tail recovery.
- Growing / frozen / sealed segment lifecycle with background index builds,
  tombstones and compaction.
- Generation-numbered checkpoints and manifest-based crash recovery.
- Query planner: exact scan for small segments and selective filters.
- Metadata filter language.
- Typed, weighted edges; `expand_hops` and `graph_boost` (spreading
  activation) in search.
- JSON/HTTP server, Python client, CLI.
