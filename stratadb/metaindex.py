"""Secondary indexes on metadata fields.

``candidates(filter)`` narrows a filter to a superset of the matching ids
using equality (hash) and numeric range (sorted array) indexes. The caller
still checks the full predicate on each candidate, so an index can only make
filtering faster, never change its result. Conditions the indexes can't
answer (``$or``, ``$not``, ``$ne``, ``$contains``, ...) just don't narrow.

Like the BM25 index, this is derived state, rebuilt from the records on open.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

_MISSING = object()


def _get(meta: dict, field: str) -> Any:
    value: Any = meta
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


class _FieldIndex:
    def __init__(self):
        self.eq: dict[Any, set[str]] = {}
        self.unhashable: set[str] = set()   # always candidates for equality lookups
        self.numbers: dict[str, float] = {}
        self._sorted: tuple[list[float], list[str]] | None = None

    def add(self, rid: str, value: Any) -> None:
        if value is _MISSING:
            return
        try:
            self.eq.setdefault(value, set()).add(rid)
        except TypeError:
            self.unhashable.add(rid)
        if _is_number(value):
            self.numbers[rid] = value
            self._sorted = None

    def remove(self, rid: str, value: Any) -> None:
        if value is _MISSING:
            return
        try:
            ids = self.eq.get(value)
            if ids is not None:
                ids.discard(rid)
                if not ids:
                    del self.eq[value]
        except TypeError:
            self.unhashable.discard(rid)
        if self.numbers.pop(rid, None) is not None:
            self._sorted = None

    def equal(self, values) -> set[str] | None:
        out = set(self.unhashable)
        for v in values:
            try:
                out |= self.eq.get(v, set())
            except TypeError:
                return None  # unhashable query value: can't use the hash index
        return out

    def range(self, lo=None, lo_inc=True, hi=None, hi_inc=True) -> set[str]:
        if self._sorted is None:
            pairs = sorted((v, rid) for rid, v in self.numbers.items())
            self._sorted = ([v for v, _ in pairs], [rid for _, rid in pairs])
        values, rids = self._sorted
        start = 0 if lo is None else (bisect.bisect_left(values, lo) if lo_inc else bisect.bisect_right(values, lo))
        end = len(values) if hi is None else (bisect.bisect_right(values, hi) if hi_inc else bisect.bisect_left(values, hi))
        return set(rids[start:end])


class MetaIndex:
    def __init__(self, fields: list[str]):
        self.fields = list(fields)
        self._idx = {f: _FieldIndex() for f in self.fields}
        self._values: dict[str, dict[str, Any]] = {}  # rid -> {field: indexed value}

    def add(self, rid: str, meta: dict) -> None:
        self.remove(rid)
        stored = {}
        for f in self.fields:
            v = _get(meta, f)
            if v is not _MISSING:
                self._idx[f].add(rid, v)
                stored[f] = v
        self._values[rid] = stored

    def remove(self, rid: str) -> None:
        stored = self._values.pop(rid, None)
        if stored:
            for f, v in stored.items():
                self._idx[f].remove(rid, v)

    def candidates(self, expr: dict | None) -> set[str] | None:
        """A superset of the ids matching ``expr``, or None if no index applies."""
        return self.plan(expr)[0]

    def plan(self, expr: dict | None) -> tuple[set[str] | None, bool]:
        """``(candidates, exact)``. ``exact`` means the indexes answered every
        condition, so the candidates are precisely the matches and the caller
        may skip re-checking the filter on them."""
        if not expr or not isinstance(expr, dict):
            return None, False
        parts: list[set[str]] = []
        exact = True
        for key, cond in expr.items():
            if key == "$and" and isinstance(cond, list):
                for sub in cond:
                    c, sub_exact = self.plan(sub)
                    if c is None:
                        exact = False
                    else:
                        parts.append(c)
                        exact = exact and sub_exact
            elif key in self._idx:
                c, field_exact = self._field_plan(self._idx[key], cond)
                if c is None:
                    exact = False
                else:
                    parts.append(c)
                    exact = exact and field_exact
            else:
                exact = False
        if not parts:
            return None, False
        parts.sort(key=len)
        result = set(parts[0])
        for p in parts[1:]:
            result &= p
            if not result:
                break
        return result, exact

    @staticmethod
    def _field_plan(idx: _FieldIndex, cond: Any) -> tuple[set[str] | None, bool]:
        def equality(values):
            c = idx.equal(values)
            ok = c is not None and not idx.unhashable and not any(isinstance(v, float) and math.isnan(v) for v in values)
            return c, ok

        if not isinstance(cond, dict) or not any(k.startswith("$") for k in cond):
            return equality([cond])
        found: list[set[str]] = []
        exact = True
        lo = hi = None
        lo_inc = hi_inc = True
        for op, arg in cond.items():
            if op == "$eq":
                c, ok = equality([arg])
            elif op == "$in" and isinstance(arg, (list, tuple, set)):
                c, ok = equality(list(arg))
            elif op in ("$gt", "$gte") and _is_number(arg):
                if lo is None or arg > lo or (arg == lo and op == "$gt"):
                    lo, lo_inc = arg, op == "$gte"
                continue
            elif op in ("$lt", "$lte") and _is_number(arg):
                if hi is None or arg < hi or (arg == hi and op == "$lt"):
                    hi, hi_inc = arg, op == "$lte"
                continue
            else:
                exact = False
                continue
            if c is None:
                exact = False
            else:
                found.append(c)
                exact = exact and ok
        if lo is not None or hi is not None:
            found.append(idx.range(lo, lo_inc, hi, hi_inc))
        if not found:
            return None, False
        result = set(found[0])
        for f in found[1:]:
            result &= f
        return result, exact
