"""Metadata filter expressions.

Filters are plain JSON so they travel unchanged over the HTTP API::

    {"lang": "en"}                                  # equality
    {"year": {"$gte": 2020, "$lt": 2025}}           # comparisons
    {"tags": {"$contains": "rag"}}                  # list membership
    {"source": {"$in": ["a.md", "b.md"]}}
    {"$or": [{"lang": "en"}, {"lang": "de"}]}
    {"$not": {"draft": True}}
    {"author": {"$exists": True}}

Field names may use dots to reach into nested dicts (``"meta.page"``).
``compile_filter`` turns an expression into a predicate over a metadata dict.
"""

from __future__ import annotations

from typing import Any, Callable

Predicate = Callable[[dict], bool]
_MISSING = object()


class FilterError(ValueError):
    pass


def _get(meta: dict, field: str) -> Any:
    value: Any = meta
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _compare(op: str, left: Any, right: Any) -> bool:
    if left is _MISSING:
        return op == "$ne" or op == "$nin"
    try:
        if op == "$eq":
            return left == right
        if op == "$ne":
            return left != right
        if op == "$gt":
            return left > right
        if op == "$gte":
            return left >= right
        if op == "$lt":
            return left < right
        if op == "$lte":
            return left <= right
        if op == "$in":
            return left in right
        if op == "$nin":
            return left not in right
        if op == "$contains":
            return isinstance(left, (list, tuple, set, str)) and right in left
        if op == "$prefix":
            return isinstance(left, str) and left.startswith(right)
    except TypeError:
        # Comparing incompatible types (e.g. str > int) never matches.
        return False
    raise FilterError(f"unknown operator {op}")


_FIELD_OPS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$contains", "$prefix", "$exists"}


def _field_predicate(field: str, cond: Any) -> Predicate:
    if not isinstance(cond, dict) or not any(k.startswith("$") for k in cond):
        return lambda meta: _get(meta, field) == cond

    checks = []
    for op, arg in cond.items():
        if op not in _FIELD_OPS:
            raise FilterError(f"unknown operator {op} on field {field!r}")
        if op in ("$in", "$nin") and not isinstance(arg, (list, tuple, set)):
            raise FilterError(f"{op} needs a list")
        if op == "$exists":
            want = bool(arg)
            checks.append(lambda meta, want=want: (_get(meta, field) is not _MISSING) == want)
        else:
            checks.append(lambda meta, op=op, arg=arg: _compare(op, _get(meta, field), arg))
    return lambda meta: all(check(meta) for check in checks)


def compile_filter(expr: dict | None) -> Predicate | None:
    """Compile a filter expression. ``None`` / ``{}`` means "match everything"."""
    if not expr:
        return None
    if not isinstance(expr, dict):
        raise FilterError("filter must be a JSON object")

    parts: list[Predicate] = []
    for key, value in expr.items():
        if key in ("$and", "$or"):
            if not isinstance(value, list) or not value:
                raise FilterError(f"{key} needs a non-empty list")
            subs = [compile_filter(v) or (lambda meta: True) for v in value]
            if key == "$and":
                parts.append(lambda meta, subs=subs: all(s(meta) for s in subs))
            else:
                parts.append(lambda meta, subs=subs: any(s(meta) for s in subs))
        elif key == "$not":
            sub = compile_filter(value) or (lambda meta: True)
            parts.append(lambda meta, sub=sub: not sub(meta))
        elif key.startswith("$"):
            raise FilterError(f"unknown top-level operator {key}")
        else:
            parts.append(_field_predicate(key, value))

    if len(parts) == 1:
        return parts[0]
    return lambda meta: all(p(meta) for p in parts)
