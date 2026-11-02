import threading

import numpy as np
import pytest

from stratadb import Database
from stratadb.__main__ import main as cli_main
from stratadb.client import Client, StrataError
from stratadb.server import make_server


@pytest.fixture
def server(tmp_path):
    db = Database(tmp_path / "db")
    srv = make_server(db, port=0, api_key="secret")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()
    db.close()


def test_end_to_end_over_http(server):
    client = Client(server, api_key="secret")
    assert client.health()["status"] == "ok"
    col = client.create_collection("docs", dim=8, seal_threshold=16)
    vecs = np.random.default_rng(0).normal(size=(40, 8))
    assert col.upsert([f"d{i}" for i in range(40)], vecs, [{"i": i} for i in range(40)]) == 40
    assert client.list_collections() == ["docs"]

    hits = col.search(vecs[3], k=3)
    assert hits[0]["id"] == "d3" and hits[0]["metadata"] == {"i": 3}
    assert all(h["metadata"]["i"] < 5 for h in col.search(vecs[3], k=3, filter={"i": {"$lt": 5}}))

    assert col.link([{"src": "d1", "dst": "d2", "rel": "next"}]) == 1
    assert col.neighbors("d1")[0]["id"] == "d2"
    assert col.search(vecs[1], k=1, expand_hops=1)[0]["neighbors"][0]["id"] == "d2"
    assert col.get(["d1"], include_vectors=True)[0]["vector"]

    assert col.delete(["d1"]) == 1
    assert col.stats()["count"] == 39
    assert col.checkpoint() >= 1
    assert "segments" in col.compact()
    assert col.scan({"i": 7})[0]["id"] == "d7"
    assert client.drop_collection("docs") is True


def test_errors_map_to_status_codes(server):
    client = Client(server, api_key="secret")
    with pytest.raises(StrataError) as e:
        client.collection("missing")
    assert e.value.status == 404
    col = client.create_collection("c", dim=4)
    with pytest.raises(StrataError) as e:
        col.upsert(["a"], [[1, 2, 3]])
    assert e.value.status == 400
    with pytest.raises(StrataError) as e:
        col.search([1, 0, 0, 0], filter={"x": {"$bogus": 1}})
    assert e.value.status == 400
    with pytest.raises(StrataError) as e:
        client._request("POST", "/collections/c/upsert", {"ids": ["a"]})
    assert e.value.status == 400 and "vectors" in str(e.value)


def test_api_key_required(server):
    assert Client(server).health()["status"] == "ok"  # health is open
    with pytest.raises(StrataError) as e:
        Client(server).list_collections()
    assert e.value.status == 401
    with pytest.raises(StrataError):
        Client(server, api_key="wrong").list_collections()


def test_cli_stats_and_compact(tmp_path, capsys):
    with Database(tmp_path / "db") as db:
        db.create_collection("a", dim=4).upsert(["x"], [[1, 0, 0, 0]])
    cli_main(["stats", "--path", str(tmp_path / "db")])
    assert '"count": 1' in capsys.readouterr().out
    cli_main(["compact", "--path", str(tmp_path / "db"), "a"])
    assert '"a"' in capsys.readouterr().out
