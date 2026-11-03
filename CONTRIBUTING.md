# Contributing

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" pyflakes     # also compiles stratadb/native/hnsw.c if cc is available
python -m stratadb.native            # "native HNSW loaded from ..." or why not
```

After editing `hnsw.c`, rebuild with `python -m stratadb.native build` (or
`pip install -e .` again). Compile with warnings on at least once:
`cc -O3 -std=c11 -Wall -Wextra -fPIC -shared -o /dev/null stratadb/native/hnsw.c`.

## Checks

```bash
python -m pyflakes stratadb tests bench examples   # lint (CI runs this first)
pytest -q                                          # ~20 s, no network
STRATADB_NATIVE=0 pytest -q                        # the numpy index must pass too (~30 s)
python examples/quickstart.py && python examples/graph_retrieval.py
```

CI runs all three on Linux for Python 3.10–3.12 on every push. macOS jobs run
on demand (workflow_dispatch); day-to-day development happens on macOS.

## Where things live

| file | what |
|---|---|
| `stratadb/hnsw.py` | the HNSW index in numpy (`PyHNSWIndex`) and the ctypes wrapper for the C one (`NativeHNSWIndex`); persistence |
| `stratadb/native/hnsw.c` | the C index: same algorithm, same files; `native/__init__.py` loads and builds it |
| `stratadb/segment.py` | growing / sealed segments, on-disk segment format, index build entry point |
| `stratadb/columns.py` | per-field numpy columns that evaluate filters over a whole segment at once |
| `stratadb/wal.py` | write-ahead log framing and recovery |
| `stratadb/collection.py` | the write path, checkpoints, recovery, search planner, graph re-ranking, maintenance |
| `stratadb/graph.py` | edges, node properties, BFS, shortest path, PageRank, components |
| `stratadb/filters.py`, `metaindex.py`, `text_index.py` | metadata filters, secondary indexes, BM25 |
| `stratadb/server.py`, `client.py`, `console.html` | HTTP API, client, web console |
| `stratadb/cluster.py` | the router: shards, replicas, fan-out, failover, resync |
| `docs/ARCHITECTURE.md` | the design and the reasons behind it; read this before changing the write path |

## Tests worth knowing about

- `tests/test_recovery_fuzz.py` runs random operation histories, cuts the WAL at
  random offsets and checks recovery against a Python model. Run it with more
  seeds after touching the WAL, checkpoints or segment lifecycle:
  `pytest tests/test_recovery_fuzz.py -k seed` with the range widened locally.
- `tests/test_background_seal.py` and `tests/test_build_processes.py` park index
  builds mid-flight to test concurrency. If you change locking, loop them:
  `for i in $(seq 20); do pytest -q tests/test_background_seal.py; done`.
- `tests/test_metaindex.py` compares indexed and unindexed collections on every
  filter shape; `tests/test_graph_algorithms.py` compares against networkx.
- `tests/test_cluster.py` runs four node servers and a router in one process
  and checks sharded results against a single embedded collection, then kills
  and restarts nodes; `tests/test_cluster_cli.py` does it with real processes.
- `tests/test_native.py` compares the C index with the numpy one (recall,
  exact search, filters, deletes, file interchange in both directions, mmap,
  concurrent searches); `tests/test_columns.py` compares column masks with the
  predicate on every operator. Both skip or shrink automatically when the C
  library isn't built, so run them with it built.

## Benchmarks

```bash
python bench/ann_bench.py --data manifold --n 20000      # synthetic, recall vs ef, build time
python bench/real_embeddings.py --n 20000                # real text embeddings (needs fastembed)
```

Numbers in the README come from these scripts on an Apple Silicon laptop.
Quote the script and the parameters with any number you add.

## Conventions

- One idea per commit, with the measured effect in the message when there is one.
- Every behaviour change ships with a test; every performance change with a number.
- Keep the only runtime dependency numpy. The C library is optional: every
  feature must work, and every test pass, with `STRATADB_NATIVE=0`.
- The two index implementations must stay interchangeable: same results
  within float rounding, same files. A change to one is a change to both.
- The `CHANGELOG.md` "Unreleased" section is updated in the same commit as the change.
