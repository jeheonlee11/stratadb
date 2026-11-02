import pytest

from stratadb.filters import FilterError, compile_filter

DOC = {"lang": "en", "year": 2023, "tags": ["rag", "graph"], "meta": {"page": 4}, "title": "Graph RAG"}


def match(expr, doc=DOC):
    pred = compile_filter(expr)
    return True if pred is None else pred(doc)


@pytest.mark.parametrize(
    "expr,expected",
    [
        (None, True),
        ({}, True),
        ({"lang": "en"}, True),
        ({"lang": "de"}, False),
        ({"year": {"$gte": 2020, "$lt": 2025}}, True),
        ({"year": {"$gt": 2023}}, False),
        ({"tags": {"$contains": "rag"}}, True),
        ({"tags": {"$contains": "sql"}}, False),
        ({"lang": {"$in": ["en", "fr"]}}, True),
        ({"lang": {"$nin": ["en"]}}, False),
        ({"meta.page": 4}, True),
        ({"meta.page": {"$lte": 3}}, False),
        ({"title": {"$prefix": "Graph"}}, True),
        ({"author": {"$exists": False}}, True),
        ({"author": {"$exists": True}}, False),
        ({"author": {"$ne": "x"}}, True),
        ({"$or": [{"lang": "de"}, {"year": 2023}]}, True),
        ({"$and": [{"lang": "en"}, {"year": 1999}]}, False),
        ({"$not": {"lang": "en"}}, False),
        ({"lang": "en", "year": 2023}, True),
        ({"year": {"$gt": "abc"}}, False),  # incompatible types never match
    ],
)
def test_filters(expr, expected):
    assert match(expr) is expected


@pytest.mark.parametrize(
    "bad",
    [{"$xor": []}, {"year": {"$between": 1}}, {"lang": {"$in": "en"}}, {"$or": []}, ["not", "a", "dict"]],
)
def test_bad_filters_raise(bad):
    with pytest.raises(FilterError):
        compile_filter(bad)
