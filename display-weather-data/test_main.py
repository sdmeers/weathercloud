"""Tests for display-weather-data.

They run the function through functions-framework's real Flask app (not a bare
call to the handler), because the /robots.txt fix depends on that app's routing.
Firestore is replaced by an in-memory fake that records every query and its
size, so the tests can assert what a request would be billed.
"""

import datetime
import html
import operator
import pathlib
import re
from typing import ClassVar
from urllib.parse import parse_qs, urlparse

import functions_framework
import pytest
from google.cloud import firestore

SOURCE = str(pathlib.Path(__file__).with_name("main.py"))
PER_PAGE = 96
N_DOCS = 250  # two full pages plus a partial one


class FakeDoc:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)

    def get(self, field):
        return self._data[field]


class FakeQuery:
    OPS: ClassVar = {"<": operator.lt, ">": operator.gt}

    def __init__(self, store, filters=(), order=None, limit=None):
        self.store, self.filters, self.order, self._limit = store, filters, order, limit

    def where(self, *, filter):
        return FakeQuery(self.store, self.filters + (filter,), self.order, self._limit)

    def order_by(self, field, direction):
        return FakeQuery(self.store, self.filters, (field, direction), self._limit)

    def limit(self, n):
        return FakeQuery(self.store, self.filters, self.order, n)

    def offset(self, n):
        raise AssertionError(
            "offset() bills every skipped document; it must not be used"
        )

    def count(self):
        raise AssertionError("count() is not needed for cursor pagination")

    def stream(self):
        assert self._limit is not None, "unbounded query"
        docs = self.store.docs
        for f in self.filters:
            docs = [
                d for d in docs if self.OPS[f.op_string](d.get(f.field_path), f.value)
            ]
        field, direction = self.order
        docs = sorted(
            docs,
            key=lambda d: d.get(field),
            reverse=direction == firestore.Query.DESCENDING,
        )
        result = docs[: self._limit]
        self.store.reads.append(len(result))
        return iter(result)


class FakeStore:
    def __init__(self, n_docs):
        start = datetime.datetime(2025, 5, 1, tzinfo=datetime.UTC)
        self.docs = [
            # Non-zero microseconds so the tests catch any cursor truncation.
            FakeDoc(
                {
                    "timestamp_UTC": start
                    + datetime.timedelta(minutes=15 * i, microseconds=123457 + i),
                    "temperature": i,
                }
            )
            for i in range(n_docs)
        ]
        self.reads = []
        self.clients_created = 0


@pytest.fixture
def store(monkeypatch):
    store = FakeStore(N_DOCS)

    class FakeClient:
        def __init__(self, database):
            assert database == "weatherdata"
            store.clients_created += 1

        def collection(self, name):
            assert name == "weather-readings"
            return FakeQuery(store)

    monkeypatch.setattr(firestore, "Client", FakeClient)
    return store


@pytest.fixture
def client():
    app = functions_framework.create_app(target="display_weather_data", source=SOURCE)
    return app.test_client()


def temperatures(resp):
    return [
        int(t)
        for t in re.findall(
            r"<tr>\s*<td>[^<]*</td>\s*<td>(-?\d+)</td>", resp.get_data(as_text=True)
        )
    ]


def links(resp):
    found = re.findall(
        r'<a href="([^"]*)" rel="nofollow"[^>]*>([^<]+)</a>',
        resp.get_data(as_text=True),
    )
    return {
        label: parse_qs(urlparse(html.unescape(href)).query) for href, label in found
    }


def follow(client, query):
    return client.get("/", query_string={k: v[0] for k, v in query.items()})


def test_robots_txt_disallows_all_without_touching_firestore(client, store):
    resp = client.get("/robots.txt")
    assert resp.status_code == 200
    assert resp.mimetype == "text/plain"
    assert resp.get_data(as_text=True) == "User-agent: *\nDisallow: /\n"
    assert store.clients_created == 0


@pytest.mark.parametrize("page", ["1", "500", "1203", "garbage"])
def test_legacy_page_urls_are_gone_without_touching_firestore(client, store, page):
    resp = client.get("/", query_string={"page": page})
    assert resp.status_code == 410
    assert store.clients_created == 0


def test_invalid_cursor_is_rejected_without_a_query(client, store):
    resp = client.get("/", query_string={"before": "not-a-date"})
    assert resp.status_code == 400
    assert store.reads == []


def test_first_page_is_newest_and_bounded(client, store):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["X-Robots-Tag"] == "noindex, nofollow"
    assert '<meta name="robots" content="noindex, nofollow">' in resp.get_data(
        as_text=True
    )
    assert temperatures(resp) == list(range(N_DOCS - 1, N_DOCS - 1 - PER_PAGE, -1))
    assert set(links(resp)) == {"Next Page", "Last Page"}
    assert store.reads == [PER_PAGE + 1]


def test_walking_next_then_previous_covers_every_reading_exactly_once(client, store):
    resp = client.get("/")
    seen = temperatures(resp)
    while "Next Page" in links(resp):
        resp = follow(client, links(resp)["Next Page"])
        seen += temperatures(resp)
    assert seen == list(range(N_DOCS - 1, -1, -1))
    assert "Previous Page" in links(resp)

    # Walk back to the newest page; each page must match exactly (no dup/skip).
    back = [temperatures(resp)]
    while "Previous Page" in links(resp):
        resp = follow(client, links(resp)["Previous Page"])
        back.append(temperatures(resp))
    assert back[-1] == list(range(N_DOCS - 1, N_DOCS - 1 - PER_PAGE, -1))
    assert [t for page in reversed(back) for t in page] == seen

    # Whatever page was requested, no query read more than one page + 1.
    assert max(store.reads) <= PER_PAGE + 1


def test_last_page_is_oldest_readings(client, store):
    resp = client.get("/", query_string={"oldest": "1"})
    assert temperatures(resp) == list(range(PER_PAGE - 1, -1, -1))
    assert set(links(resp)) == {"First Page", "Previous Page"}


@pytest.mark.parametrize(
    "param,value",
    [("before", "2000-01-01T00:00:00+00:00"), ("after", "2100-01-01T00:00:00+00:00")],
)
def test_cursor_past_either_end_renders_empty_page(client, store, param, value):
    resp = client.get("/", query_string={param: value})
    assert resp.status_code == 200
    assert temperatures(resp) == []
    assert set(links(resp)) == {"First Page"}
