"""Write-ahead log.

Every mutation is appended here before it touches in-memory state, so a crash
loses nothing that was acknowledged. On disk each entry is one frame::

    u32 payload_len | u32 crc32(payload) | payload

and the payload is::

    u32 header_len | header JSON (utf-8) | float32 vector bytes (optional)

Vectors travel as raw float32 rather than JSON numbers: a 768-dim batch of
100 vectors is 300 KB raw versus ~1.5 MB as JSON text.

Replay stops at the first torn or corrupt frame (a crash mid-append) and
truncates the file there, so the log always ends on a clean boundary.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

_FRAME = struct.Struct("<II")
_U32 = struct.Struct("<I")

SYNC_MODES = ("always", "flush", "never")


@dataclass
class WalEntry:
    lsn: int
    op: str
    data: dict = field(default_factory=dict)
    vectors: np.ndarray | None = None


def encode(entry: WalEntry) -> bytes:
    header = {"lsn": entry.lsn, "op": entry.op, "data": entry.data}
    vec_bytes = b""
    if entry.vectors is not None:
        vecs = np.ascontiguousarray(entry.vectors, dtype=np.float32)
        header["shape"] = list(vecs.shape)
        vec_bytes = vecs.tobytes()
    hbytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    payload = _U32.pack(len(hbytes)) + hbytes + vec_bytes
    return _FRAME.pack(len(payload), zlib.crc32(payload)) + payload


def decode(payload: bytes) -> WalEntry:
    (hlen,) = _U32.unpack_from(payload, 0)
    header = json.loads(payload[4: 4 + hlen])
    vectors = None
    if "shape" in header:
        vectors = np.frombuffer(payload[4 + hlen:], dtype=np.float32).reshape(header["shape"]).copy()
    return WalEntry(lsn=header["lsn"], op=header["op"], data=header["data"], vectors=vectors)


class WriteAheadLog:
    def __init__(self, path: str | Path, sync: str = "flush"):
        if sync not in SYNC_MODES:
            raise ValueError(f"sync must be one of {SYNC_MODES}")
        self.path = Path(path)
        self.sync = sync
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_lsn = 0
        # Scan once to find the tail LSN and chop any torn frame left by a crash.
        for entry in self._scan(truncate=True):
            self.last_lsn = entry.lsn
        self._fh = open(self.path, "ab")

    def _scan(self, truncate: bool = False) -> Iterator[WalEntry]:
        if not self.path.exists():
            return
        good_end = 0
        with open(self.path, "rb") as fh:
            while True:
                head = fh.read(_FRAME.size)
                if len(head) < _FRAME.size:
                    break
                length, crc = _FRAME.unpack(head)
                payload = fh.read(length)
                if len(payload) < length or zlib.crc32(payload) != crc:
                    break
                try:
                    entry = decode(payload)
                except (ValueError, KeyError, json.JSONDecodeError):
                    break
                good_end = fh.tell()
                yield entry
        if truncate and self.path.stat().st_size > good_end:
            with open(self.path, "r+b") as fh:
                fh.truncate(good_end)

    def append(self, op: str, data: dict | None = None, vectors: np.ndarray | None = None) -> int:
        self.last_lsn += 1
        self._fh.write(encode(WalEntry(self.last_lsn, op, data or {}, vectors)))
        if self.sync != "never":
            self._fh.flush()
            if self.sync == "always":
                os.fsync(self._fh.fileno())
        return self.last_lsn

    def replay(self, after_lsn: int = 0) -> Iterator[WalEntry]:
        self._fh.flush()
        for entry in self._scan():
            if entry.lsn > after_lsn:
                yield entry

    def reset(self, next_lsn: int | None = None) -> None:
        """Drop all entries (called after a checkpoint made them redundant).

        LSNs keep counting up from ``next_lsn`` (default: the current tail) so a
        checkpoint's LSN never collides with a later entry.
        """
        self._fh.close()
        with open(self.path, "wb") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        if next_lsn is not None:
            self.last_lsn = next_lsn
        self._fh = open(self.path, "ab")

    def size_bytes(self) -> int:
        self._fh.flush()
        return self.path.stat().st_size if self.path.exists() else 0

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
