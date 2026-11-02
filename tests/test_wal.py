import numpy as np
import pytest

from stratadb.wal import WriteAheadLog


def test_append_and_replay(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.log")
    vecs = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert wal.append("upsert", {"ids": ["a", "b"]}, vecs) == 1
    assert wal.append("delete", {"ids": ["a"]}) == 2
    entries = list(wal.replay())
    assert [e.op for e in entries] == ["upsert", "delete"]
    assert entries[0].data == {"ids": ["a", "b"]}
    np.testing.assert_array_equal(entries[0].vectors, vecs)
    assert entries[1].vectors is None
    assert [e.lsn for e in wal.replay(after_lsn=1)] == [2]


def test_reopen_continues_lsn(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.log")
    wal.append("x")
    wal.append("y")
    wal.close()
    wal = WriteAheadLog(tmp_path / "wal.log")
    assert wal.last_lsn == 2
    assert wal.append("z") == 3


def test_torn_tail_is_truncated(tmp_path):
    path = tmp_path / "wal.log"
    wal = WriteAheadLog(path)
    wal.append("keep", {"n": 1})
    wal.append("torn", {"n": 2})
    wal.close()
    data = path.read_bytes()
    path.write_bytes(data[:-5])  # simulate a crash mid-write
    wal = WriteAheadLog(path)
    assert [e.op for e in wal.replay()] == ["keep"]
    assert wal.append("next") == 2


def test_corrupt_frame_stops_replay(tmp_path):
    path = tmp_path / "wal.log"
    wal = WriteAheadLog(path)
    wal.append("a")
    first_end = path.stat().st_size
    wal.append("b")
    wal.append("c")
    wal.close()
    data = bytearray(path.read_bytes())
    data[first_end + 12] ^= 0xFF  # flip a byte inside the second frame
    path.write_bytes(bytes(data))
    wal = WriteAheadLog(path)
    assert [e.op for e in wal.replay()] == ["a"]


def test_reset_keeps_lsn_monotonic(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.log", sync="always")
    wal.append("a")
    wal.append("b")
    wal.reset()
    assert list(wal.replay()) == []
    assert wal.append("c") == 3


def test_bad_sync_mode(tmp_path):
    with pytest.raises(ValueError):
        WriteAheadLog(tmp_path / "w.log", sync="sometimes")
