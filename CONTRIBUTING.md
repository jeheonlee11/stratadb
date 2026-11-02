# Contributing

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" pyflakes
```

## Checks

```bash
python -m pyflakes stratadb tests bench examples   # lint (CI runs this first)
pytest -q                                          # ~20 s, no network
python examples/quickstart.py && python examples/graph_retrieval.py
```

CI runs all three on Linux for Python 3.10–3.12 on every push. macOS jobs run
on demand (workflow_dispatch); day-to-day development happens on macOS.

## Where things live

| file | what |
|---|---|
| `stratadb/hnsw.py` | the HNSW index: build, search, filtered search, persistence |
| `stratadb/segment.py` | growing / sealed segments, on-disk segment format, index build entry point |
| `stratadb/wal.py` | write-ahead log framing and recovery |
| `stratadb/collection.py` | the write path, checkpoints, recovery, search planner, graph re-ranking, maintenance |
| `stratadb/graph.py` | edges, node properties, BFS, shortest path, PageRank, components |
| `stratadb/filters.py`, `metaindex.py`, `text_index.py` | metadata filters, secondary indexes, BM25 |
| `stratadb/server.py`, `client.py`, `console.html` | HTTP API, client, web console |
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
- Keep the only runtime dependency numpy.
- The `CHANGELOG.md` "Unreleased" section is updated in the same commit as the change.
