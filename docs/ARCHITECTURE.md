# StrataDB architecture

StrataDB is an embedded vector database written in Python and numpy, with
the HNSW index also implemented in C. This document explains how data moves
through it and why it is built this way.

```
            upsert / delete / link
                     │
                     ▼
   ┌────────────── WAL (wal.log) ──────────────┐  every mutation, CRC-framed
   │                                           │
   ▼                                           ▼
growing segment ──seal──▶ frozen segment ──build──▶ sealed segment (HNSW)
 (brute force)            (still searchable)          seg-N/ on disk
   │                                                     │
   └─────────── checkpoint: manifest.json ◀──────────────┘
                 growing-G.npz/json, graph-G.json

          edge store (graph.py): typed, weighted edges between ids
```

## Records and collections

A collection is one vector space: fixed dimension and metric (`cosine`, `ip`
or `l2`). Each record has a string id, a vector and a JSON metadata object.
Cosine collections store unit-normalized vectors, so cosine distance is
`1 - dot`.

A `Database` is just a directory of collections. Nothing is shared between
collections; each has its own WAL, segments and lock.

## Write path

1. Validate the batch (shape, finite values, metadata length). Duplicate ids
   inside one batch keep the last occurrence.
2. Append **one** WAL entry for the whole batch. The header (ids, metadata)
   is JSON; vectors follow as raw float32 bytes, so a 768-dim batch is not
   inflated into JSON text. One entry per batch makes every batch atomic: a
   crash keeps all of it or none of it.
3. Apply in memory: remove the id from wherever it currently lives (growing,
   frozen or a sealed segment's tombstones) and put it in the growing segment.
4. If the growing segment is now over `seal_threshold`, seal threshold-sized
   chunks one after another (a big batch may overshoot the threshold; it is
   never split across WAL entries).

The WAL frame is `u32 length | u32 crc32 | payload`. On open, StrataDB scans
the log and truncates at the first frame that is short or fails its CRC: a
crash mid-append loses only the write that was never acknowledged.

`sync` controls durability per collection: `always` (fsync every append),
`flush` (default: write to the OS, survives a process crash) or `never`.

## Segments

**Growing.** Recent writes. Searched by brute force (one matrix product),
which is fast for a few thousand vectors and needs no index maintenance.

**Frozen.** When the growing segment reaches `seal_threshold`, it is frozen
and a new growing segment takes its place. A worker thread builds the HNSW
index while the frozen segment keeps answering queries by brute force. Writes
are never blocked by an index build.

Several segments can be in this state at once (`build_processes`, default
2), each on its own CPU core; a bulk load that seals four segments builds
them four-wide. With the native index the builds run on worker **threads**:
the C code releases the GIL for the whole build, so threads already use every
core and the vectors never leave the process. With the numpy index each build
runs in a worker **process** instead, since Python code would otherwise share
the main process's GIL. Those workers start with `spawn`, never `fork`: the
main process runs background threads holding locks, and a forked child would
inherit those locks in a held state. Segments under
`build_process_min_records` build in-process, since starting a worker process
costs about a second. If a worker can't start at all (spawn re-imports
`__main__`, which fails under a REPL, a notebook or `python -`), the
collection notes it and builds in-process from then on; nothing is lost
either way.

**Sealed.** The built segment is written to `seg-N/` (`index.npz` for the
graph structure, `vectors.npy` for the vectors, `records.json` for ids and
metadata) via a temporary directory and an atomic rename, then swapped in. On
reopen the vectors are memory-mapped read-only rather than copied: opening a
collection costs no vector I/O up front, pages come in from disk as searches
touch them, and the OS can drop them under memory pressure. Segments written
by 0.3.0 and earlier (vectors inside `index.npz`) still load, in memory. Deletes and
overwrites that hit the frozen segment during the build are reconciled at
swap time by tombstoning those ids in the new segment. A frozen record is
located by its *future* segment id, so installing the built segment changes
no locator entries: the id simply starts resolving to the sealed segment.

Sealed segments are immutable. Deleting a record adds a tombstone; the graph
still routes through tombstoned nodes but never returns them. `compact()`
rewrites segments whose tombstone ratio passes `compact_deleted_ratio` and
merges undersized ones. It works like sealing: the live records are
snapshotted under the lock, the replacement segments are built outside it,
and at install time every record whose locator no longer points at a
rewritten segment (deleted or overwritten during the build) is tombstoned in
the new one. With `auto_compact` (default) a delete or overwrite that pushes a
segment past the ratio starts this on a worker thread.

## Search

For each query, every segment returns its local top-k and the results are
merged. With the native index, sealed segments past the exact-scan size are
searched on `search_threads` threads at once (the C walk runs without the
GIL), so a collection of several segments answers in about the time of its
slowest one. Each sealed segment picks a plan:

| situation | plan |
|---|---|
| live records x dim <= `exact_search_limit` | exact scan (matrix product + argpartition) |
| filter matches few records | exact scan over the matches |
| otherwise | HNSW, with the filter checked during the walk; `ef` widens until k allowed hits are found |

**Filters become column masks.** A filter is compiled once into a predicate
over a metadata dict (`filters.py`), which the planner still uses for
single-record checks. For selecting records out of a segment, though, calling
that predicate once per record is the dominant cost on a large segment, so
segments keep columns (`columns.py`): the first filter that mentions a field
turns it into a float64 array for its numeric values, an int32 array of codes
into the field's distinct non-numeric values (strings, None, ...), and two
bit masks (field present; value is something the arrays can't compare, such
as a list). Every operator then runs as a few numpy comparisons, with
`$contains`/`$prefix` evaluated once per distinct value and spread through the
codes, and only the rows the arrays can't decide handed to the predicate.
Python's equality rules are kept exactly (`1 == 1.0 == True`, NaN equals
nothing, a missing field fails everything but `$ne`/`$nin`/`$exists: False`,
incompatible comparisons never match), and `tests/test_columns.py` checks
every operator against the predicate row for row on awkward values. Sealed
segments never change, so their columns are built once; growing segments
append to theirs and re-derive rows that were overwritten or deleted.
Filtered search on 100k x 384 vectors without a metadata index went from
24.7 ms to 1.4 ms.

The exact-scan threshold exists because a numpy matrix product over a small
segment is both fast and exact: below a couple of million floats it ties or
beats the graph walk even with the native index (at 384 dimensions the two
cross at about 10k vectors), and it never misses a neighbour.

## HNSW

`hnsw.py` is a from-scratch implementation of Malkov & Yashunin (2016):
layered proximity graph, greedy descent through the upper layers, best-first
search with a dynamic candidate list (`ef`) on layer 0, and the paper's
neighbour-selection heuristic (keep a candidate only if it is closer to the
new node than to any neighbour already kept). Layer 0 allows `2*M` links.

The index exists twice:

- `PyHNSWIndex` (numpy). Neighbour selection computes all
  candidate-to-candidate distances in one matrix product and keeps a running
  "distance to nearest kept neighbour" vector, instead of one numpy call per
  pair.
- `NativeHNSWIndex` (`native/hnsw.c`, loaded through ctypes). The same
  algorithm, step for step, in C: vectors in one flat array, level-0 links in
  a fixed-width table of `2*M` slots per node, upper-level links in a pool of
  `M`-slot rows, a per-thread epoch-stamped visited array (no clearing between
  queries, and concurrent searches never share state), and 8-way unrolled
  distance loops the compiler vectorizes. There is no Python or numpy C API
  involved, so a single `cc -O3 -shared` builds it and ctypes releases the
  GIL for every call.

**Quantized traversal.** Search time is bound by memory latency: each
candidate's vector is a random row of a buffer far larger than cache. The
native index therefore keeps, per sealed segment, an 8-bit code of every
vector (`codes.npy`; per-dimension min and max over the segment, 255 steps
between) and walks the graph on those. A code row is a quarter the size of
the float row, and the distance is a float-by-uint8 dot product plus two
per-query constants (`dot(q, x) = sum(q*qmin) + sum(q*qscale*code)`; for L2
the decoded squared norm of each row is stored too). The codes only choose
the candidates: the `ef` nodes the walk ends with are re-scored with the
float32 vectors before the top k are returned, so the distances a caller
sees are exact and recall is indistinguishable from the float walk (the
candidate set differs in a few places where codes misorder near-ties, but a
near-tie is by definition not what decides recall). Inserts always use the
floats; adding to a quantized index drops its codes. On reopen both
`codes.npy` and `vectors.npy` are memory-mapped, and a query touches only
the codes plus `ef` float rows, so the hot working set shrinks about 4x.

`HNSWIndex` is whichever one is in use (`python -m stratadb.native` says
which; `STRATADB_NATIVE=0` forces numpy). Both write and read the same
`index.npz` (levels, CSR-packed links, deleted labels, entry point) and the
same `vectors.npy`, so a segment built by one loads in the other, and a
differential test checks that a round trip through either leaves the graph
byte-for-byte identical.

Measured against hnswlib (C++) with the same `M=16, ef_construction=128` on
20k x 384 vectors, recall@10 is within about one point on both easy and hard
data for either implementation. The numpy build is 25-40x slower than hnswlib.
The C build is about 2x faster than hnswlib's pip wheel on Apple Silicon and
answers queries in half the time, but only because hnswlib's SIMD kernels are
x86-only; on x86 hnswlib should win. Search time is bound by memory latency,
not arithmetic: a distance costs ~95 ns on a 3k-vector index that sits in
cache and ~420 ns on a 50k-vector one (384 dims), and software prefetching
measured neutral here.

## Checkpoints and recovery

`checkpoint()`:

1. Bump the checkpoint generation `G`.
2. Write `growing-G.npz/json` (growing **and** frozen records, since a frozen
   segment isn't on disk yet) and `graph-G.json`.
3. Atomically replace `manifest.json`: WAL position, generation, sealed
   segment ids with their tombstones, snapshot file names.
4. Empty the WAL, then delete the previous generation's snapshot files.

Opening a collection loads the manifest, the sealed segments and the
snapshots, deletes anything the manifest doesn't reference (half-built
segments, stale snapshots), and replays WAL entries newer than the
checkpoint. Every operation is idempotent (upsert overwrites, delete of a
missing id is a no-op, link overwrites), so replaying an entry that was
already captured is harmless.

Snapshot names carry the generation rather than the WAL position: two
checkpoints with no writes in between would otherwise reuse a file name and
overwrite the snapshot the live manifest still points to.

Checkpoints happen when a segment is sealed or compacted, when the WAL
exceeds `wal_checkpoint_bytes`, and on `close()`.

## Graph layer

Edges are `(src, rel, dst)` with a float weight and a properties dict, stored
in an adjacency map with a reverse index. Nodes can also carry a property
dict of their own. Both are logged in the WAL and snapshotted with each
checkpoint. An edge may reference an id that doesn't exist (yet); deleting an
id removes its vector record, its node properties and every edge touching it.

A collection created with `dim=0` is graph-only: it has no vectors at all,
just nodes and edges. That is what LightRAG-style systems need for their
entity/relation graph, and it shares the same WAL, checkpoint and recovery
machinery as vector collections.

Search can use the graph in two ways.

**`expand_hops=N`** attaches each hit's neighbourhood (breadth-first, with
hop count, relation and the node it was reached from).

**`graph_boost=α`** re-ranks with spreading activation:

1. Take a candidate pool of `max(4k, 32)` nearest records.
2. A candidate's *strength* is how far its similarity sits above the pool
   mean. Candidates at or below the mean spread nothing, so a weak match
   can't promote its neighbours.
3. Each candidate splits its strength across its edges by weight; neighbours
   accumulate activation, scaled so a record linked only to the strongest
   candidate gets 1.0.
4. Linked records outside the pool are pulled in (if they pass the filter).
5. Final score: `(1 - α) * similarity + α * activation`.

This is what lets a question about "battery manufacturers' employees" return
the people linked to battery companies even when the people's own text never
mentions batteries (`examples/graph_retrieval.py`).

## Graph algorithms

- `shortest_path`: breadth-first, fewest hops, optionally restricted to one
  relation type and edge direction, bounded by `max_hops`.
- `pagerank`: power iteration over a weighted edge list held in numpy arrays
  (`bincount` does the sparse matrix-vector product). Dangling nodes
  redistribute along the personalization vector, the same definition
  networkx uses; personalized PageRank restarts walks at given seeds, which
  scores nodes by graph proximity to them.
- `components`: weakly connected components by iterative DFS.

## Secondary indexes

`index_fields=[...]` keeps, per field, a hash map from value to ids (equality
and `$in`) and a lazily rebuilt sorted array of numeric values (ranges). For a
filter, `MetaIndex.plan` intersects whatever the indexes can answer into a
candidate set and reports whether that set is *exact*: every condition was
answered by an index (no `$or`, `$ne`, unindexed field, unhashable values,
NaN). Segments then check only the candidates, and skip the Python predicate
entirely when the plan is exact. Equality uses dict lookup, so it agrees with
Python `==` (`1 == 1.0 == True`); range indexes hold only numbers, matching the
filter engine, where comparing a string with a number never matches.

Like the BM25 index, these are derived state rebuilt from the records on open.

## Keyword and hybrid search

A collection created with `text_field="..."` keeps a BM25 index (`k1=1.2`,
`b=0.75`) over that metadata field. The tokenizer keeps identifiers like
`e1042`, `wal.log` and `torn-write` whole, because exact tokens are where
keyword search beats embeddings. The index is derived state: it is updated on
every write and rebuilt from the records when the collection opens, so it
needs no files of its own.

`search(vector, text=...)` runs both searches over a candidate pool of
`max(4k, 32)` and merges the two rankings with reciprocal rank fusion
(`score = sum of 1 / (60 + rank)`), which needs no score calibration between
cosine similarity and BM25. Metadata filters apply to both sides.

## Backups

`backup(dest)` waits for any background seal, checkpoints, and copies exactly
the files the manifest references while holding the lock. The copy is a
point-in-time snapshot with an empty WAL that opens like any collection;
writes resume as soon as the copy finishes.

## Testing strategy

Beyond unit tests for each module, two kinds of tests check behaviour against
an independent model:

- **Crash-recovery fuzzing** (`tests/test_recovery_fuzz.py`): random
  histories of upserts, deletes, edge and node writes, seals and checkpoints
  are mirrored in a plain-Python model. The WAL of a copy is cut at a random
  byte offset, the copy is reopened, and its state must equal the model after
  some prefix of the operations no older than the last checkpoint. This found
  that batches crossing the seal threshold weren't atomic (fixed by logging
  one entry per batch).
- **Recall against exact search** for every metric, and against hnswlib for
  index quality (see the README benchmarks).

## Concurrency

One re-entrant lock per collection guards all state. Reads and writes are
serialized per collection; the HNSW build runs outside the lock and takes it
only to install the finished segment. The HTTP server is a
`ThreadingHTTPServer`, one thread per connection.

## Cluster

`cluster.py` is a router, not a new storage engine: the nodes are ordinary
`stratadb serve` processes, and the router is the only component that knows
there is more than one. It keeps one small piece of state of its own (which
collections exist and how each is sharded, in `cluster-state.json`) and
otherwise forwards.

- **Placement** is static hash sharding: shard = `crc32(id) % shards`. Static
  because the alternative, consistent hashing with migration, is a lot of
  machinery for a system whose shard count is chosen once. `sharding: "none"`
  pins a collection to shard 0 for whole-graph algorithms.
- **Replication** is synchronous within a shard: a write goes to every live
  node of the shard in parallel and is acknowledged once the primary has it.
  A replica that fails the write is marked stale. The alternative (async
  replication with a log position) is faster on writes but needs a catch-up
  protocol; export-import resync of a stale replica is slower but needs
  nothing new, and replicas are rarely down.
- **Failover** is read-only: when a primary is unreachable, reads go to an
  in-sync replica and writes to that shard are refused. Promoting a replica
  automatically would need a consensus on who is primary (otherwise two
  routers, or a router and a returning primary, can diverge), and that is a
  different project. The honest guarantee: no two nodes ever accept
  conflicting writes.
- **Search** fans out to one node per shard and merges the per-shard top-k by
  distance; since every shard returns its exact local top-k (within recall),
  the merge is exact. Edges are written to the shards of both endpoints so
  one-hop neighbourhoods are complete anywhere; spreading activation runs
  per shard because its candidate pool is per shard.
- **Health** is a loop probing `/health`; a node that comes back and is stale
  is re-synced from its primary by dropping and re-importing each collection.

## Limits

- Single process per collection directory: a `LOCK` file (flock / msvcrt)
  makes a second process fail fast instead of corrupting the WAL. Scale-out is
  by sharding across processes through the router, not by sharing a
  directory.
- Cluster mode has no automatic primary election and no re-sharding.
- Everything except sealed vectors is held in memory.
- The server has bearer-token auth only; put it behind TLS if it leaves
  localhost. The web console page itself is served without auth; every API
  call it makes needs the key.
