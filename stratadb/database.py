"""A database is a directory of collections."""

from __future__ import annotations

import re
import shutil
import threading
from pathlib import Path

from .collection import Collection, CollectionConfig

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _check_name(name: str) -> str:
    if not _NAME.match(name) or ".." in name:
        raise ValueError(f"invalid collection name {name!r} (letters, digits, _ . - ; max 128)")
    return name


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, Collection] = {}
        self._lock = threading.Lock()

    def create_collection(self, name: str, dim: int, metric: str = "cosine", exist_ok: bool = True, **options):
        _check_name(name)
        with self._lock:
            if name in self._collections:
                if not exist_ok:
                    raise ValueError(f"collection {name!r} already exists")
                col = self._collections[name]
                if (col.config.dim, col.config.metric) != (dim, metric):
                    raise ValueError(
                        f"collection {name!r} exists with dim={col.config.dim} metric={col.config.metric}"
                    )
                return col
            path = self.path / name
            if path.exists() and not exist_ok:
                raise ValueError(f"collection {name!r} already exists")
            col = Collection(path, CollectionConfig(dim=dim, metric=metric, **options))
            self._collections[name] = col
            return col

    def collection(self, name: str) -> Collection:
        _check_name(name)
        with self._lock:
            if name not in self._collections:
                if not (self.path / name / "config.json").exists():
                    raise KeyError(f"no collection named {name!r}")
                self._collections[name] = Collection(self.path / name)
            return self._collections[name]

    __getitem__ = collection

    def list_collections(self) -> list[str]:
        return sorted(p.name for p in self.path.iterdir() if (p / "config.json").exists())

    def drop_collection(self, name: str) -> bool:
        _check_name(name)
        with self._lock:
            col = self._collections.pop(name, None)
            if col is not None:
                col.wal.close()
                col._closed = True
            path = self.path / name
            if path.exists():
                shutil.rmtree(path)
                return True
            return False

    def close(self) -> None:
        with self._lock:
            for col in self._collections.values():
                col.close()
            self._collections.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
