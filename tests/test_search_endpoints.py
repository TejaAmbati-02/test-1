"""Endpoint tests for /v1/search, /v1/search/suggestions and /indices."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from routers import search as router
from routers.search import MAX_IMAGE_BYTES
from services.embedding_store import (
    EmbeddingIdNotFound,
    EmbeddingStoreUnavailable,
    ImageEmbedding,
    QueryEmbedding,
)
from services.opensearch import IndexNotFoundError, OpenSearchQueryError
from services.search import search_service
from tests.conftest import GIF_BYTES, JPEG_BYTES, PNG_BYTES, WEBP_BYTES, make_search_response

BASE = {"channel": "web", "language": "en"}
QUERY_ID = "asmo:emb:query:" + "a" * 64
IMAGE_ID = "asmo:emb:image:" + "b" * 64


@pytest.fixture
def search_mocks(monkeypatch):
    """Mock the search service and embedding store used by the router."""
    store = router.embedding_store
    mocks = {
        "search_v1": AsyncMock(side_effect=lambda **kw: make_search_response(request_id=kw["request_id"])),
        "filter_browse": AsyncMock(side_effect=lambda **kw: make_search_response(request_id=kw["request_id"])),
        "query_embedding": AsyncMock(side_effect=lambda q: QueryEmbedding("qkey", q, [0.1, 0.2], False)),
        "load_query": AsyncMock(side_effect=lambda k: QueryEmbedding(k, "stored query", [0.5], True)),
        "image_embedding": AsyncMock(side_effect=lambda d, m: ImageEmbedding("ikey", d, m, [0.3], False)),
        "load_image": AsyncMock(side_effect=lambda k: ImageEmbedding(k, b"img", "image/png", [0.6], True)),
    }
    for name in ("search_v1", "filter_browse"):
        monkeypatch.setattr(search_service, name, mocks[name])
    for name in ("query_embedding", "load_query", "image_embedding", "load_image"):
        monkeypatch.setattr(store, name, mocks[name])
    return mocks


def _error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["code"] == code
    assert set(body) == {"code", "message"}
    return body


# --------------------------------------------------------------------------
# POST /v1/search — success paths
# --------------------------------------------------------------------------


class TestSearchModes:
    def test_text_search(self, client, search_mocks):
        resp = client.post("/v1/search", data={**BASE, "search_query": "  ball valve  "})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_results"] == 1
        assert body["products"][0]["sku_id"] == "SKU-1"
        assert body["query_embedding_id"] == "qkey"
        assert body["image_embedding_id"] is None
        assert body["request_id"]  # set by the request-logging middleware

        search_mocks["query_embedding"].assert_awaited_once_with("ball valve")
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["query"] == "ball valve"
        assert kw["query_vector"] == [0.1, 0.2]
        assert kw["image_vector"] is None
        assert kw["index"] == "asmo-search-api-index"
        assert kw["vector_field"] == "content_vector"
        assert kw["language"] == "en"
        assert kw["filters"] == []
        assert kw["sort_by"] is None
        assert kw["sort_order"] == "desc"
        assert (kw["page"], kw["page_size"]) == (1, 60)
        assert kw["retrieval_k"] == 100
        assert kw["ef_search"] == 500
        assert kw["image_min_score"] == 0.6
        assert kw["request_id"] == body["request_id"]
        search_mocks["filter_browse"].assert_not_awaited()

    @pytest.mark.parametrize("data, mime", [
        (PNG_BYTES, "image/png"),
        (JPEG_BYTES, "image/jpeg"),
        (GIF_BYTES, "image/gif"),
        (WEBP_BYTES, "image/webp"),
    ])
    def test_image_search(self, client, search_mocks, data, mime):
        resp = client.post("/v1/search", data=BASE, files={"image": ("photo", data, mime)})
        assert resp.status_code == 200
        assert resp.json()["image_embedding_id"] == "ikey"
        assert resp.json()["query_embedding_id"] is None
        search_mocks["image_embedding"].assert_awaited_once_with(data, mime)
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["image_vector"] == [0.3]
        assert kw["query"] is None and kw["query_vector"] is None

    def test_image_mime_is_detected_from_bytes_not_header(self, client, search_mocks):
        resp = client.post("/v1/search", data=BASE, files={"image": ("x.bin", PNG_BYTES, "application/octet-stream")})
        assert resp.status_code == 200
        search_mocks["image_embedding"].assert_awaited_once_with(PNG_BYTES, "image/png")

    def test_declared_type_may_differ_from_actual(self, client, search_mocks):
        resp = client.post("/v1/search", data=BASE, files={"image": ("x.jpg", PNG_BYTES, "image/jpeg")})
        assert resp.status_code == 200
        search_mocks["image_embedding"].assert_awaited_once_with(PNG_BYTES, "image/png")

    def test_image_part_without_filename(self, client, search_mocks):
        resp = client.post("/v1/search", data=BASE, files={"image": (None, PNG_BYTES)})
        assert resp.status_code == 200
        search_mocks["image_embedding"].assert_awaited_once_with(PNG_BYTES, "image/png")

    def test_text_plus_image(self, client, search_mocks):
        resp = client.post("/v1/search", data={**BASE, "search_query": "valve"}, files={"image": ("a.png", PNG_BYTES, "image/png")})
        assert resp.status_code == 200
        assert resp.json()["query_embedding_id"] == "qkey"
        assert resp.json()["image_embedding_id"] == "ikey"
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["query"] == "valve" and kw["image_vector"] == [0.3]

    def test_empty_image_part_is_ignored(self, client, search_mocks):
        resp = client.post("/v1/search", data={**BASE, "search_query": "valve"}, files={"image": ("a.png", b"", "image/png")})
        assert resp.status_code == 200
        search_mocks["image_embedding"].assert_not_awaited()

    def test_filter_browse(self, client, search_mocks):
        filters = {"product_type": "Valve", "price": {"lte": 500}}
        resp = client.post("/v1/search", data={
            **BASE, "filters": json.dumps(filters), "sort_by": "price", "sort_order": "asc",
            "page": "2", "page_size": "25",
        })
        assert resp.status_code == 200
        assert resp.json()["query_embedding_id"] is None
        kw = search_mocks["filter_browse"].call_args.kwargs
        assert kw["filters"] == [
            {"term": {"product_type.keyword": "Valve"}},
            {"range": {"price": {"lte": 500}}},
        ]
        assert kw["sort_by"] == "price" and kw["sort_order"] == "asc"
        assert (kw["page"], kw["page_size"]) == (2, 25)
        search_mocks["search_v1"].assert_not_awaited()
        search_mocks["query_embedding"].assert_not_awaited()

    def test_text_search_with_filters(self, client, search_mocks):
        resp = client.post("/v1/search", data={**BASE, "search_query": "valve", "filters": '{"currency": ["SAR", "USD"]}'})
        assert resp.status_code == 200
        assert search_mocks["search_v1"].call_args.kwargs["filters"] == [{"terms": {"currency": ["SAR", "USD"]}}]

    def test_relevance_sort_means_default_order(self, client, search_mocks):
        client.post("/v1/search", data={**BASE, "search_query": "valve", "sort_by": "relevance"})
        assert search_mocks["search_v1"].call_args.kwargs["sort_by"] is None

    def test_mobile_arabic_accepted(self, client, search_mocks):
        resp = client.post("/v1/search", data={"channel": "mobile", "language": "ar", "search_query": "صمام"})
        assert resp.status_code == 200
        assert search_mocks["search_v1"].call_args.kwargs["language"] == "ar"

    def test_empty_text_fields_are_treated_as_absent(self, client, search_mocks):
        resp = client.post("/v1/search", data={**BASE, "search_query": "valve", "sort_by": "", "filters": ""})
        assert resp.status_code == 200
        assert search_mocks["search_v1"].call_args.kwargs["sort_by"] is None


class TestEmbeddingIds:
    def test_query_embedding_id_reuses_stored_query(self, client, search_mocks):
        resp = client.post("/v1/search", data={**BASE, "query_embedding_id": QUERY_ID})
        assert resp.status_code == 200
        assert resp.json()["query_embedding_id"] == QUERY_ID
        search_mocks["load_query"].assert_awaited_once_with(QUERY_ID)
        search_mocks["query_embedding"].assert_not_awaited()
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["query"] == "stored query" and kw["query_vector"] == [0.5]

    def test_image_embedding_id_reuses_stored_image(self, client, search_mocks):
        resp = client.post("/v1/search", data={**BASE, "image_embedding_id": IMAGE_ID})
        assert resp.status_code == 200
        assert resp.json()["image_embedding_id"] == IMAGE_ID
        search_mocks["load_image"].assert_awaited_once_with(IMAGE_ID)
        assert search_mocks["search_v1"].call_args.kwargs["image_vector"] == [0.6]

    def test_raw_query_wins_over_id(self, client, search_mocks):
        client.post("/v1/search", data={**BASE, "search_query": "valve", "query_embedding_id": QUERY_ID})
        search_mocks["load_query"].assert_not_awaited()
        search_mocks["query_embedding"].assert_awaited_once_with("valve")

    def test_raw_image_wins_over_id(self, client, search_mocks):
        client.post("/v1/search", data={**BASE, "image_embedding_id": IMAGE_ID}, files={"image": ("a.png", PNG_BYTES, "image/png")})
        search_mocks["load_image"].assert_not_awaited()
        search_mocks["image_embedding"].assert_awaited_once()

    def test_unknown_embedding_id(self, client, search_mocks):
        search_mocks["load_query"].side_effect = EmbeddingIdNotFound("query embedding id 'x' was not found or has expired")
        resp = client.post("/v1/search", data={**BASE, "query_embedding_id": QUERY_ID})
        body = _error(resp, 400, "EMBEDDING_ID_NOT_FOUND")
        assert "expired" in body["message"]

    def test_embedding_store_unavailable(self, client, search_mocks):
        search_mocks["load_image"].side_effect = EmbeddingStoreUnavailable("Redis is disabled")
        resp = client.post("/v1/search", data={**BASE, "image_embedding_id": IMAGE_ID})
        body = _error(resp, 503, "EMBEDDING_STORE_UNAVAILABLE")
        assert "Redis is disabled" in body["message"]


# --------------------------------------------------------------------------
# POST /v1/search — validation errors
# --------------------------------------------------------------------------


class TestSearchValidation:
    @pytest.mark.parametrize("field", ["channel", "language"])
    def test_missing_required_field(self, client, search_mocks, field):
        data = {**BASE, "search_query": "valve"}
        del data[field]
        body = _error(client.post("/v1/search", data=data), 400, "VALIDATION_ERROR")
        assert body["message"].startswith(f"{field}:")

    @pytest.mark.parametrize("field, value, code", [
        ("channel", "tv", "UNSUPPORTED_CHANNEL"),
        ("language", "fr", "UNSUPPORTED_LANGUAGE"),
        ("sort_by", "name", "INVALID_SORT_BY"),
        ("sort_order", "up", "INVALID_SORT_ORDER"),
    ])
    def test_unsupported_enum_values(self, client, search_mocks, field, value, code):
        resp = client.post("/v1/search", data={**BASE, "search_query": "valve", field: value})
        _error(resp, 400, code)
        search_mocks["search_v1"].assert_not_awaited()

    @pytest.mark.parametrize("field, value", [
        ("page", "0"), ("page", "abc"), ("page_size", "0"), ("page_size", "201"),
    ])
    def test_invalid_pagination(self, client, search_mocks, field, value):
        body = _error(client.post("/v1/search", data={**BASE, "search_query": "v", field: value}), 400, "VALIDATION_ERROR")
        assert body["message"].startswith(f"{field}:")

    def test_max_page_size_allowed(self, client, search_mocks):
        assert client.post("/v1/search", data={**BASE, "search_query": "v", "page_size": "200"}).status_code == 200

    @pytest.mark.parametrize("data", [
        BASE,
        {**BASE, "search_query": "   "},
        {**BASE, "filters": "{}"},
    ])
    def test_empty_request(self, client, search_mocks, data):
        _error(client.post("/v1/search", data=data), 400, "EMPTY_REQUEST")
        search_mocks["filter_browse"].assert_not_awaited()

    @pytest.mark.parametrize("filters, code", [
        ("{not json", "INVALID_FILTER"),
        ('["price"]', "INVALID_FILTER"),
        ('{"colour": "red"}', "INVALID_FILTER_KEY"),
        ('{"price": "cheap"}', "INVALID_FILTER_VALUE"),
        ('{"contract_indicator": "yes"}', "INVALID_FILTER_VALUE"),
    ])
    def test_invalid_filters(self, client, search_mocks, filters, code):
        _error(client.post("/v1/search", data={**BASE, "filters": filters}), 400, code)

    def test_unsupported_declared_image_type(self, client, search_mocks):
        resp = client.post("/v1/search", data=BASE, files={"image": ("a.txt", PNG_BYTES, "text/plain")})
        body = _error(resp, 400, "UNSUPPORTED_IMAGE_TYPE")
        assert "text/plain" in body["message"]

    def test_bytes_that_are_not_an_image(self, client, search_mocks):
        resp = client.post("/v1/search", data=BASE, files={"image": ("a.png", b"definitely not a png", "image/png")})
        _error(resp, 400, "UNSUPPORTED_IMAGE_TYPE")
        search_mocks["image_embedding"].assert_not_awaited()

    def test_text_part_without_filename_is_not_an_image(self, client, search_mocks):
        resp = client.post("/v1/search", data=BASE, files={"image": (None, "plain text ✓".encode("utf-8"))})
        _error(resp, 400, "UNSUPPORTED_IMAGE_TYPE")

    def test_oversized_image(self, client, search_mocks):
        big = PNG_BYTES + b"\x00" * MAX_IMAGE_BYTES
        resp = client.post("/v1/search", data=BASE, files={"image": ("big.png", big, "image/png")})
        _error(resp, 413, "IMAGE_TOO_LARGE")
        search_mocks["image_embedding"].assert_not_awaited()

    def test_image_exactly_at_limit_is_accepted(self, client, search_mocks):
        exact = PNG_BYTES + b"\x00" * (MAX_IMAGE_BYTES - len(PNG_BYTES))
        resp = client.post("/v1/search", data=BASE, files={"image": ("ok.png", exact, "image/png")})
        assert resp.status_code == 200

    def test_huge_part_without_filename_is_413(self, client, search_mocks):
        huge = PNG_BYTES + b"\x00" * (2 * MAX_IMAGE_BYTES + 1)
        resp = client.post("/v1/search", data=BASE, files={"image": (None, huge)})
        _error(resp, 413, "IMAGE_TOO_LARGE")

    def test_malformed_multipart_body(self, client, search_mocks):
        resp = client.post("/v1/search", content=b"garbage", headers={"content-type": "multipart/form-data"})
        _error(resp, 400, "VALIDATION_ERROR")


# --------------------------------------------------------------------------
# POST /v1/search — backend error mapping
# --------------------------------------------------------------------------


class TestSearchErrors:
    @pytest.mark.parametrize("error, status, code", [
        (IndexNotFoundError("Index 'x' not found"), 404, "INDEX_NOT_FOUND"),
        (OpenSearchQueryError("bad query"), 400, "QUERY_ERROR"),
        (RuntimeError("kaboom"), 500, "INTERNAL_ERROR"),
    ])
    def test_search_v1_errors(self, client, search_mocks, error, status, code):
        search_mocks["search_v1"].side_effect = error
        body = _error(client.post("/v1/search", data={**BASE, "search_query": "v"}), status, code)
        assert body["message"] == str(error)

    @pytest.mark.parametrize("error, status, code", [
        (IndexNotFoundError("missing"), 404, "INDEX_NOT_FOUND"),
        (OpenSearchQueryError("bad"), 400, "QUERY_ERROR"),
        (ValueError("boom"), 500, "INTERNAL_ERROR"),
    ])
    def test_filter_browse_errors(self, client, search_mocks, error, status, code):
        search_mocks["filter_browse"].side_effect = error
        _error(client.post("/v1/search", data={**BASE, "filters": '{"currency": "SAR"}'}), status, code)

    def test_embedding_failure_is_500(self, client, search_mocks):
        search_mocks["query_embedding"].side_effect = RuntimeError("OCI down")
        body = _error(client.post("/v1/search", data={**BASE, "search_query": "v"}), 500, "INTERNAL_ERROR")
        assert body["message"] == "OCI down"


# --------------------------------------------------------------------------
# GET /v1/search/suggestions
# --------------------------------------------------------------------------


@pytest.fixture
def suggestions_mock(monkeypatch):
    from models.response import SuggestionsResponse

    mock = AsyncMock(return_value=SuggestionsResponse(
        query="val",
        query_suggestions=[{"text": "Valve", "destination_url": "/search?q=Valve"}],
        product_suggestions=[{"text": "Valve", "destination_url": "/p/1"}],
    ))
    monkeypatch.setattr(search_service, "get_suggestions", mock)
    return mock


SUGGEST = {"q": "val", "channel": "web", "language": "en"}


class TestSuggestions:
    def test_success(self, client, suggestions_mock):
        resp = client.get("/v1/search/suggestions", params={**SUGGEST, "query_limit": 3, "product_limit": 7})
        assert resp.status_code == 200
        assert resp.json()["query_suggestions"][0]["text"] == "Valve"
        suggestions_mock.assert_awaited_once_with(
            index="asmo-search-api-index", query="val", language="en", query_limit=3, product_limit=7,
        )

    def test_default_limits(self, client, suggestions_mock):
        client.get("/v1/search/suggestions", params=SUGGEST)
        kw = suggestions_mock.call_args.kwargs
        assert kw["query_limit"] == 5 and kw["product_limit"] == 5

    @pytest.mark.parametrize("field, value, code", [
        ("channel", "tv", "UNSUPPORTED_CHANNEL"),
        ("language", "fr", "UNSUPPORTED_LANGUAGE"),
    ])
    def test_unsupported_enums(self, client, suggestions_mock, field, value, code):
        _error(client.get("/v1/search/suggestions", params={**SUGGEST, field: value}), 400, code)
        suggestions_mock.assert_not_awaited()

    @pytest.mark.parametrize("params", [
        {"channel": "web", "language": "en"},
        {**SUGGEST, "q": ""},
        {"q": "val", "language": "en"},
        {**SUGGEST, "query_limit": 0},
        {**SUGGEST, "product_limit": 51},
    ])
    def test_validation_errors(self, client, suggestions_mock, params):
        body = _error(client.get("/v1/search/suggestions", params=params), 400, "VALIDATION_ERROR")
        assert body["message"].startswith("query.")

    @pytest.mark.parametrize("error, status, code", [
        (IndexNotFoundError("missing"), 404, "INDEX_NOT_FOUND"),
        (OpenSearchQueryError("bad"), 400, "QUERY_ERROR"),
        (RuntimeError("boom"), 500, "INTERNAL_ERROR"),
    ])
    def test_backend_errors(self, client, suggestions_mock, error, status, code):
        suggestions_mock.side_effect = error
        _error(client.get("/v1/search/suggestions", params=SUGGEST), status, code)


# --------------------------------------------------------------------------
# GET /indices
# --------------------------------------------------------------------------


class TestIndices:
    def test_list_indices(self, client, monkeypatch):
        monkeypatch.setattr(search_service._os, "list_indices", AsyncMock(return_value=[
            {"name": "products", "doc_count": 10, "size_bytes": 2048, "status": "open"},
        ]))
        resp = client.get("/indices")
        assert resp.status_code == 200
        assert resp.json() == {
            "indices": [{"name": "products", "doc_count": 10, "size_bytes": 2048, "status": "open"}],
            "total": 1,
        }

    def test_list_indices_failure(self, client, monkeypatch):
        monkeypatch.setattr(search_service._os, "list_indices", AsyncMock(side_effect=ConnectionError("down")))
        body = _error(client.get("/indices"), 500, "INTERNAL_ERROR")
        assert body["message"] == "down"
