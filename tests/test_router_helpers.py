from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from routers import search as router
from routers.search import (
    _api_error,
    _build_os_filters,
    _image_mime,
    _parse_filters,
    _resolve_embeddings,
    _valid_scalar,
    _validate_enum,
    _validate_filter_value,
)
from services.embedding_store import ImageEmbedding, QueryEmbedding
from tests.conftest import GIF_BYTES, JPEG_BYTES, PNG_BYTES, WEBP_BYTES


def _code(exc_info) -> str:
    return exc_info.value.detail["code"]


@pytest.mark.parametrize("data, mime", [
    (PNG_BYTES, "image/png"),
    (JPEG_BYTES, "image/jpeg"),
    (GIF_BYTES, "image/gif"),
    (b"GIF87a" + b"\x00" * 8, "image/gif"),
    (WEBP_BYTES, "image/webp"),
    (b"RIFF\x00\x00\x00\x00WAVE", None),
    (b"hello world", None),
    (b"", None),
])
def test_image_mime(data, mime):
    assert _image_mime(data) == mime


def test_api_error():
    exc = _api_error(418, "TEAPOT", "short and stout")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 418
    assert exc.detail == {"code": "TEAPOT", "message": "short and stout"}


def test_validate_enum():
    _validate_enum("web", frozenset({"web", "mobile"}), "channel", "UNSUPPORTED_CHANNEL")
    with pytest.raises(HTTPException) as exc:
        _validate_enum("tv", frozenset({"web", "mobile"}), "channel", "UNSUPPORTED_CHANNEL")
    assert exc.value.status_code == 400
    assert _code(exc) == "UNSUPPORTED_CHANNEL"
    assert exc.value.detail["message"] == "Unsupported channel 'tv'. Allowed: mobile, web"


class TestParseFilters:
    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_empty(self, raw):
        assert _parse_filters(raw) == {}

    def test_object(self):
        assert _parse_filters('{"currency": "SAR"}') == {"currency": "SAR"}

    @pytest.mark.parametrize("raw", ["{bad json", '["a"]', '"text"', "42"])
    def test_invalid(self, raw):
        with pytest.raises(HTTPException) as exc:
            _parse_filters(raw)
        assert exc.value.status_code == 400
        assert _code(exc) == "INVALID_FILTER"


class TestFilterValues:
    @pytest.mark.parametrize("key, value, ok", [
        ("price", 10, True),
        ("price", 9.5, True),
        ("price", True, False),  # bools are not numbers here
        ("price", "10", False),
        ("contract_indicator", True, True),
        ("contract_indicator", 1, False),
        ("currency", "SAR", True),
        ("currency", "", False),
        ("currency", 5, False),
    ])
    def test_valid_scalar(self, key, value, ok):
        assert _valid_scalar(key, value) is ok

    @pytest.mark.parametrize("key, value", [
        ("price", 100),
        ("price", [10, 20]),
        ("price", {"gte": 50, "lte": 150}),
        ("moq", {"gt": 1}),
        ("supplier_count", {"lt": 3.5}),
        ("contract_indicator", False),
        ("currency", "SAR"),
        ("category_ids", ["c1", "c2"]),
    ])
    def test_accepts(self, key, value):
        _validate_filter_value(key, value)

    @pytest.mark.parametrize("key, value", [
        ("price", {}),
        ("price", {"between": 5}),
        ("price", {"gte": "5"}),
        ("currency", {"gte": "A"}),  # ranges only on numeric fields
        ("price", []),
        ("price", [1, "2"]),
        ("currency", [""]),
        ("contract_indicator", "true"),
        ("currency", None),
    ])
    def test_rejects(self, key, value):
        with pytest.raises(HTTPException) as exc:
            _validate_filter_value(key, value)
        assert exc.value.status_code == 400
        assert _code(exc) == "INVALID_FILTER_VALUE"


class TestBuildOsFilters:
    def test_builds_term_terms_and_range_clauses(self):
        clauses = _build_os_filters({
            "product_type": "Valve",
            "category_ids": ["c1", "c2"],
            "price": {"gte": 10, "lte": 100},
            "contract_indicator": True,
        })
        assert clauses == [
            {"term": {"product_type.keyword": "Valve"}},
            {"terms": {"category_ids": ["c1", "c2"]}},
            {"range": {"price": {"gte": 10, "lte": 100}}},
            {"term": {"contract_indicator": True}},
        ]

    def test_empty(self):
        assert _build_os_filters({}) == []

    def test_unknown_key(self):
        with pytest.raises(HTTPException) as exc:
            _build_os_filters({"colour": "red"})
        assert _code(exc) == "INVALID_FILTER_KEY"
        assert "colour" in exc.value.detail["message"]

    def test_invalid_value(self):
        with pytest.raises(HTTPException) as exc:
            _build_os_filters({"price": "cheap"})
        assert _code(exc) == "INVALID_FILTER_VALUE"


class TestResolveEmbeddings:
    @pytest.fixture
    def store(self, monkeypatch):
        q = QueryEmbedding("qk", "valve", [0.1], False)
        i = ImageEmbedding("ik", b"img", "image/png", [0.2], False)
        mocks = {
            "query_embedding": AsyncMock(return_value=q),
            "load_query": AsyncMock(return_value=q),
            "image_embedding": AsyncMock(return_value=i),
            "load_image": AsyncMock(return_value=i),
        }
        for name, mock in mocks.items():
            monkeypatch.setattr(router.embedding_store, name, mock)
        return mocks

    async def test_raw_inputs_are_embedded(self, store):
        q, i = await _resolve_embeddings("valve", "ignored-id", (b"img", "image/png"), "ignored-id")
        assert q.key == "qk" and i.key == "ik"
        store["query_embedding"].assert_awaited_once_with("valve")
        store["image_embedding"].assert_awaited_once_with(b"img", "image/png")
        store["load_query"].assert_not_awaited()
        store["load_image"].assert_not_awaited()

    async def test_ids_are_loaded(self, store):
        await _resolve_embeddings(None, "qid", None, "iid")
        store["load_query"].assert_awaited_once_with("qid")
        store["load_image"].assert_awaited_once_with("iid")

    async def test_nothing_to_resolve(self, store):
        assert await _resolve_embeddings(None, None, None, None) == [None, None]
