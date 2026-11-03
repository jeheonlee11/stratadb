# Changelog

## 0.9.0

- **8-bit scalar quantization** (`CollectionConfig.sq8`, default on). Sealed
  segments keep uint8 codes of their vectors (per-dimension min/max,
  `codes.npy` beside `vectors.npy`, memory-mapped on reopen). The native
  search walks the graph on the codes, a quarter of the memory traffic, then
  re-ranks the candidates it found with the float32 vectors, so recall is
  unchanged and returned distances are exact. 50k x 384, ef=64: 0.74 ms ->
  0.41 ms per segment. The numpy index ignores the codes.
- **Parallel segment search** (`CollectionConfig.search_threads`, default 4):
  a query searches sealed segments past the exact-scan size on several threads
  at once with the native index.
- `NativeHNSWIndex.quantize()`, `.quantized`, `.codes`.

## 0.8.0

- **Native HNSW index.** The index is now also written in C
  (`stratadb/native/hnsw.c`, loaded through ctypes; no Python or numpy C API).
  Same algorithm and the same `index.npz` / `vectors.npy` files as the numpy
  version, so existing segments load unchanged and either implementation can
  read what the other wrote. `pip install` compiles it when a C compiler is
  present and skips it otherwise; `python -m stratadb.native` reports which
  index is in use, `python -m stratadb.native build` compiles it later, and
  `STRATADB_NATIVE=0` forces the numpy index. `stats()["index"]` and
  `GET /health` report `native` or `python`.
- Segment builds use the native index in one call per segment and run on
  worker threads instead of worker processes when it is loaded (the C code
  releases the GIL), so nothing is pickled and small segments no longer wait
  for a process to start.
- **Vectorized filters on sealed segments.** The first filter on a field turns
  it into a numpy column (float64 for numbers, integer codes for strings and
  other values, presence and "other" masks); every operator then evaluates
  over the whole segment in a few array operations, with per-row fallback only
  for values the arrays can't decide (lists under `$eq`, odd comparisons).
  Semantics are identical to the predicate, checked row for row by a
  differential test on awkward values. Filtered search on 100k x 384 vectors
  (no metadata index): 24.7 ms -> 1.4 ms.
- Growing (and frozen) segments use the same columns, kept in step with
  puts and deletes, and scan without copying the vector buffer when most rows
  qualify: a 16k x 384 growing segment answers in 0.6 ms unfiltered and
  0.7 ms with a 50% filter (was 3.8 ms / 2.9 ms).
- `HNSWIndex.search(allow_labels=...)` takes the allowed labels as a
  collection or array, which the native index turns into a byte mask;
  segments use it instead of a Python callback per visited node.

## 0.7.0

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
