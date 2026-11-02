"""Crash-recovery fuzzing.

Run a random mix of upserts, deletes, edge and node writes, checkpoints and
seals while tracking the expected state in a plain-Python model. Then copy
the collection directory, cut its WAL at a random byte offset (a crash that
tore the last write), reopen the copy, and require the recovered state to
equal the model after some prefix of the operations: never a state that
mixes writes, and never older than the last checkpoint.
"""

import random
import shutil

import numpy as np
import pytest

from stratadb import Collection, CollectionConfig


def state_of(col: Collection):
    records = {r["id"]: r["metadata"] for r in col.scan()}
    edges = {(e[0], e[1], e[2]) for e in col.graph.all_edges()}
    nodes = {n: dict(col.graph.get_node(n)) for n in col.graph.node_ids()}
    return records, edges, nodes


class Model:
    def __init__(self):
        self.records, self.edges, self.nodes = {}, set(), {}

    def snapshot(self):
        return dict(self.records), set(self.edges), {k: dict(v) for k, v in self.nodes.items()}

    def delete(self, rid):
        self.records.pop(rid, None)
        self.nodes.pop(rid, None)
        self.edges = {e for e in self.edges if rid not in (e[0], e[2])}


@pytest.mark.parametrize("seed", range(8))
def test_random_crashes_recover_a_consistent_prefix(tmp_path, seed):
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    col = Collection(tmp_path / "c", CollectionConfig(dim=8, seal_threshold=25, background_seal=False))
    model = Model()
    history = [model.snapshot()]   # history[i] = expected state after i operations
    checkpoint_at = 0               # operations covered by the latest checkpoint
    ids = [f"r{i}" for i in range(40)]

    for _ in range(150):
        op = rng.random()
        if op < 0.45:
            batch = rng.sample(ids, rng.randint(1, 4))
            metas = [{"v": rng.randint(0, 99)} for _ in batch]
            col.upsert(batch, nrng.normal(size=(len(batch), 8)), metas)
            for rid, meta in zip(batch, metas):
                model.records[rid] = meta
        elif op < 0.6:
            batch = rng.sample(ids, rng.randint(1, 3))
            col.delete(batch)
            for rid in batch:
                model.delete(rid)
        elif op < 0.75:
            a, b = rng.sample(ids, 2)
            rel = rng.choice(["x", "y"])
            col.link([{"src": a, "dst": b, "rel": rel}])
            model.edges.add((a, rel, b))
        elif op < 0.82:
            a, b = rng.sample(ids, 2)
            col.unlink([{"src": a, "dst": b}])
            model.edges = {e for e in model.edges if not (e[0] == a and e[2] == b)}
        elif op < 0.9:
            rid = rng.choice(ids)
            props = {"p": rng.randint(0, 9)}
            col.put_nodes([{"id": rid, "props": props}])
            model.nodes[rid] = props
        else:
            col.checkpoint()
            history.append(model.snapshot())
            checkpoint_at = len(history) - 1
            continue
        history.append(model.snapshot())
        # Sealing checkpoints internally; track it so the lower bound stays right.
        if col.stats()["checkpoint_lsn"] == col.stats()["last_lsn"]:
            checkpoint_at = len(history) - 1

    assert state_of(col) == history[-1]
    col.wait_for_compaction()  # don't copy the directory mid-checkpoint

    wal_size = (tmp_path / "c" / "wal.log").stat().st_size
    for trial in range(6):
        crash_dir = tmp_path / f"crash{trial}"
        shutil.copytree(tmp_path / "c", crash_dir)
        (crash_dir / "LOCK").unlink()
        cut = rng.randint(0, wal_size)
        with open(crash_dir / "wal.log", "r+b") as fh:
            fh.truncate(cut)
        recovered = state_of(Collection(crash_dir))
        assert any(recovered == history[i] for i in range(checkpoint_at, len(history))), (
            f"seed {seed} trial {trial}: recovered state matches no prefix after the last checkpoint"
        )
        if cut == wal_size:
            assert recovered == history[-1]
