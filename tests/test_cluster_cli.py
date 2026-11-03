"""`stratadb serve` x2 + `stratadb cluster` as real processes, driven through the client."""

import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from stratadb.client import Client


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url, seconds=15):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            Client(url).health()
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    raise RuntimeError(f"{url} did not come up")


@pytest.mark.skipif(sys.platform == "win32", reason="process signals differ")
def test_cluster_over_processes(tmp_path):
    env = dict(os.environ, STRATADB_API_KEY="k")
    ports = [free_port() for _ in range(3)]
    procs = [subprocess.Popen([sys.executable, "-m", "stratadb", "serve", "--path", str(tmp_path / f"n{i}"),
                               "--port", str(p)], env=env) for i, p in enumerate(ports[:2])]
    config = tmp_path / "cluster.json"
    config.write_text(json.dumps({"shards": [{"nodes": [f"http://127.0.0.1:{ports[0]}"]},
                                             {"nodes": [f"http://127.0.0.1:{ports[1]}"]}], "api_key": "k"}))
    try:
        for p in ports[:2]:
            wait_for(f"http://127.0.0.1:{p}")
        procs.append(subprocess.Popen([sys.executable, "-m", "stratadb", "cluster", str(config),
                                       "--port", str(ports[2])], env=env))
        url = f"http://127.0.0.1:{ports[2]}"
        wait_for(url)
        client = Client(url, api_key="k")
        assert client.health()["role"] == "router" and client.health()["nodes_up"] == 2
        col = client.create_collection("c", dim=8)
        vecs = np.random.default_rng(0).normal(size=(100, 8))
        assert col.upsert([f"r{i}" for i in range(100)], vecs) == 100
        assert col.search(vecs[7], 1)[0]["id"] == "r7"
        status = client.cluster_status()
        assert len(status["shards"]) == 2 and status["collections"]["c"]["sharding"] == "hash"
        assert (tmp_path / "cluster-state.json").exists()
        counts = [Client(f"http://127.0.0.1:{p}", api_key="k").collection("c").stats()["count"] for p in ports[:2]]
        assert sum(counts) == 100 and all(c > 0 for c in counts)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=20)
