"""Vectorized filter evaluation over a sealed segment's metadata.

A sealed segment's metadata never changes, so the first filter that touches a
field turns that field into a column: a float64 array for the numeric values
(NaN elsewhere), integer codes into a table of the distinct non-numeric
hashable values (strings, None, ...), and bit masks saying which rows have the
field at all and which hold something else (lists, dicts). From then on a
filter on that field is a handful of numpy comparisons instead of one Python
predicate call per record, which is what made filtered search on a large
segment cost tens of milliseconds.

Semantics match ``stratadb.filters`` exactly (``1 == 1.0 == True``, a missing
field fails every test except ``$ne``/``$nin``/``$exists: False``, comparing
incompatible types never matches); rows the vectorized path can't decide, such
as list values under ``$eq``, go through the ordinary predicate one by one.
A differential test checks every operator against the predicate on awkward
values.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .filters import _FIELD_OPS, FilterError, _field_predicate, _get, _MISSING, compile_filter


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


class _Column:
    """One field over the rows of a segment. Arrays carry spare capacity so a
    growing segment can append rows cheaply; ``n`` is the live row count."""

    __slots__ = ("field", "n", "_cap", "_present", "_numeric", "_codes", "uniques", "code_of", "_other", "dirty")

    def __init__(self, field: str, metas: list[dict]):
        self.field = field
        self.n = 0
        self.dirty: set[int] = set()  # rows changed in place since the last sync
        self.uniques: list = []
        self.code_of: dict = {}
        self._alloc(max(len(metas), 16))
        for meta in metas:
            self._set(self.n, meta)
            self.n += 1

    def _alloc(self, cap: int) -> None:
        self._cap = cap
        self._present = np.zeros(cap, dtype=bool)
        self._numeric = np.full(cap, np.nan, dtype=np.float64)
        self._codes = np.full(cap, -1, dtype=np.int32)
        self._other = np.zeros(cap, dtype=bool)   # unhashable values, and NaN

    def _grow(self) -> None:
        old = (self._present, self._numeric, self._codes, self._other)
        n = self.n
        self._alloc(self._cap * 2)
        self._present[:n], self._numeric[:n], self._codes[:n], self._other[:n] = (a[:n] for a in old)

    @property
    def present(self) -> np.ndarray:
        return self._present[: self.n]

    @property
    def numeric(self) -> np.ndarray:
        return self._numeric[: self.n]

    @property
    def codes(self) -> np.ndarray:
        return self._codes[: self.n]

    @property
    def other(self) -> np.ndarray:
        return self._other[: self.n]

    def _set(self, i: int, meta: dict) -> None:
        v = _get(meta, self.field)
        self._present[i] = v is not _MISSING
        self._numeric[i] = np.nan
        self._codes[i] = -1
        self._other[i] = False
        if v is _MISSING:
            return
        if _is_number(v):
            self._numeric[i] = float(v)
            return
        try:
            code = self.code_of.get(v)
            if code is None:
                code = self.code_of[v] = len(self.uniques)
                self.uniques.append(v)
        except TypeError:  # unhashable: list, dict
            self._other[i] = True
            return
        if isinstance(v, float):  # NaN: hashable but equal to nothing
            self._other[i] = True
        else:
            self._codes[i] = code

    def append(self, meta: dict) -> None:
        if self.n == self._cap:
            self._grow()
        self._set(self.n, meta)
        self.n += 1

    def set(self, i: int, meta: dict) -> None:
        self._set(i, meta)

    # ---------------------------------------------------------------- helpers
    def _map_uniques(self, fn) -> np.ndarray:
        """Evaluate ``fn`` on each distinct value and spread the result over the rows."""
        if not self.uniques:
            return np.zeros(self.n, dtype=bool)
        table = np.fromiter((bool(fn(u)) for u in self.uniques), dtype=bool, count=len(self.uniques))
        out = np.zeros(self.n, dtype=bool)
        codes = self.codes
        has = codes >= 0
        out[has] = table[codes[has]]
        return out

    def _fallback(self, metas: list[dict], rows: np.ndarray, cond: Any) -> np.ndarray:
        """Decide ``rows`` (a bool mask) with the ordinary predicate."""
        out = np.zeros(self.n, dtype=bool)
        idx = np.flatnonzero(rows)
        if len(idx):
            pred = _field_predicate(self.field, cond)
            out[idx] = [pred(metas[i]) for i in idx.tolist()]
        return out

    # ------------------------------------------------------------------- ops
    def equals(self, arg: Any, metas: list[dict]) -> np.ndarray:
        if _is_number(arg):
            return self.numeric == float(arg)
        if isinstance(arg, float):  # NaN
            return np.zeros(self.n, dtype=bool)
        try:
            code = self.code_of.get(arg)
        except TypeError:
            # A list or dict argument can only equal list/dict values.
            return self._fallback(metas, self.other, {"$eq": arg})
        if code is None:
            return np.zeros(self.n, dtype=bool)
        return self.codes == code

    def compare(self, op: str, arg: Any, metas: list[dict]) -> np.ndarray:
        fn = {"$gt": np.greater, "$gte": np.greater_equal, "$lt": np.less, "$lte": np.less_equal}[op]
        if _is_number(arg):
            with np.errstate(invalid="ignore"):
                return fn(self.numeric, float(arg))
        if isinstance(arg, str):
            def test(u, arg=arg):
                return isinstance(u, str) and fn(u, arg)
            return self._map_uniques(test)
        # Anything else (lists, None, NaN): let Python's own rules decide, row by row.
        return self._fallback(metas, self.present, {op: arg})

    def contains(self, arg: Any, metas: list[dict]) -> np.ndarray:
        def test(u, arg=arg):  # hashable containers: str, tuple, frozenset
            try:
                return isinstance(u, (str, tuple, frozenset, set, list)) and arg in u
            except TypeError:
                return False
        out = self._map_uniques(test)
        if self.other.any():
            out |= self._fallback(metas, self.other, {"$contains": arg})
        return out

    def prefix(self, arg: Any, metas: list[dict]) -> np.ndarray:
        if not isinstance(arg, str):
            return self._fallback(metas, self.present, {"$prefix": arg})
        return self._map_uniques(lambda u, arg=arg: isinstance(u, str) and u.startswith(arg))


class ColumnStore:
    """Per-field columns over a list of metadata dicts, built lazily.

    The store shares the caller's ``metas`` list. Rows appended to that list
    are picked up on the next ``mask``; a row changed in place must be
    reported with ``invalidate_row`` (a set insert on the write path, the
    column update deferred to the next ``mask``). Sealed segments never do
    either; growing segments do both.
    """

    def __init__(self, metas: list[dict]):
        self.metas = metas
        self._cols: dict[str, _Column] = {}

    def invalidate_row(self, i: int) -> None:
        for col in self._cols.values():
            col.dirty.add(i)

    def _sync(self, col: _Column) -> None:
        metas = self.metas
        if col.dirty:
            for i in col.dirty:
                if i < col.n:
                    col.set(i, metas[i])
            col.dirty.clear()
        while col.n < len(metas):
            col.append(metas[col.n])

    def column(self, field: str) -> _Column:
        col = self._cols.get(field)
        if col is None:
            col = self._cols[field] = _Column(field, self.metas)
        else:
            self._sync(col)
        return col

    @property
    def fields(self) -> list[str]:
        return list(self._cols)

    def mask(self, expr: dict | None) -> np.ndarray | None:
        """Boolean mask of the rows matching ``expr``; None when it matches everything."""
        if not expr:
            return None
        compile_filter(expr)  # validate, raising FilterError exactly as the predicate path does
        return self._expr(expr)

    def _expr(self, expr: dict) -> np.ndarray:
        parts = []
        for key, value in expr.items():
            if key == "$and":
                parts.append(np.logical_and.reduce([self._expr(v) if v else self._all() for v in value]))
            elif key == "$or":
                parts.append(np.logical_or.reduce([self._expr(v) if v else self._all() for v in value]))
            elif key == "$not":
                parts.append(~(self._expr(value) if value else self._all()))
            elif key.startswith("$"):
                raise FilterError(f"unknown top-level operator {key}")
            else:
                parts.append(self._field(key, value))
        return parts[0] if len(parts) == 1 else np.logical_and.reduce(parts)

    def _all(self) -> np.ndarray:
        return np.ones(len(self.metas), dtype=bool)

    def _field(self, field: str, cond: Any) -> np.ndarray:
        col = self.column(field)
        if not isinstance(cond, dict) or not any(k.startswith("$") for k in cond):
            return col.equals(cond, self.metas)
        parts = []
        for op, arg in cond.items():
            if op not in _FIELD_OPS:
                raise FilterError(f"unknown operator {op} on field {field!r}")
            if op == "$exists":
                parts.append(col.present if arg else ~col.present)
            elif op == "$eq":
                parts.append(col.equals(arg, self.metas))
            elif op == "$ne":
                parts.append(~col.equals(arg, self.metas))
            elif op == "$in":
                hits = [col.equals(a, self.metas) for a in arg]
                parts.append(np.logical_or.reduce(hits) if hits else np.zeros(col.n, dtype=bool))
            elif op == "$nin":
                hits = [col.equals(a, self.metas) for a in arg]
                parts.append(~np.logical_or.reduce(hits) if hits else self._all())
            elif op in ("$gt", "$gte", "$lt", "$lte"):
                parts.append(col.compare(op, arg, self.metas))
            elif op == "$contains":
                parts.append(col.contains(arg, self.metas))
            elif op == "$prefix":
                parts.append(col.prefix(arg, self.metas))
        return parts[0] if len(parts) == 1 else np.logical_and.reduce(parts)
