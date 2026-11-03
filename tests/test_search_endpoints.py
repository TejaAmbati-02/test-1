"""Endpoint tests for /v1/search, /v1/search/suggestions and /indices."""
from __future__ import annotations

import json
import uuid
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from routers import search as router
from routers.search import MAX_IMAGE_BYTES, NOT_AN_IMAGE_PART, UNSUPPORTED_IMAGE
from services.embedding_store import EmbeddingStoreUnavailable, RequestIdNotFound, SearchContext
from services.image_processor import InvalidImageError
from services.opensearch import IndexNotFoundError, OpenSearchQueryError
from services.search import search_service
from tests.conftest import GIF_BYTES, JPEG_BYTES, PNG_BYTES, WEBP_BYTES, make_search_response

BASE = {"channel": "web", "language": "en"}
PREVIOUS_ID = "0f8fad5b-d9cb-469f-a165-70867728950e"
PROCESSED = (b"processed-image", "image/jpeg")

# The endpoint only accepts multipart/form-data. httpx sends a plain ``data=``
# dict as urlencoded, so text-only requests add this empty, ignored text part
# to force a multipart body.
_FORCE_MULTIPART = {"_": (None, b"")}


def _post(client, data: dict | None = None, files: dict | None = None):
    return client.post("/v1/search", data=data or {}, files=files or _FORCE_MULTIPART)


def _real_image(fmt: str, size: tuple[int, int] = (8, 6)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format=fmt)
    return buf.getvalue()


def _resolve(request_id, query, image, previous_request_id):
    """Stand-in for EmbeddingStore.resolve."""
    if previous_request_id and not query and image is None:
        return SearchContext(previous_request_id, "stored query", [0.5], None, reused=True)
    return SearchContext(
        request_id,
        query,
        [0.1, 0.2] if query else None,
        [0.3] if image is not None else None,
        reused=False,
    )


@pytest.fixture
def search_mocks(monkeypatch):
    """Mock the search service, the embedding store and the image processor used by the router."""
    mocks = {
        "search_v1": AsyncMock(side_effect=lambda **kw: make_search_response(request_id=kw["request_id"])),
        "filter_browse": AsyncMock(side_effect=lambda **kw: make_search_response(request_id=kw["request_id"])),
        "resolve": AsyncMock(side_effect=_resolve),
        "process": Mock(return_value=PROCESSED),
    }
    for name in ("search_v1", "filter_browse"):
        monkeypatch.setattr(search_service, name, mocks[name])
    monkeypatch.setattr(router.embedding_store, "resolve", mocks["resolve"])
    monkeypatch.setattr(router.image_processor, "process", mocks["process"])
    return mocks


def _error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["code"] == code
    assert set(body) == {"code", "message"}
    return body


def _resolve_args(search_mocks) -> tuple:
    """(request_id, query, image, previous_request_id) passed to embedding_store.resolve."""
    search_mocks["resolve"].assert_awaited_once()
    return search_mocks["resolve"].call_args.args


# --------------------------------------------------------------------------
# POST /v1/search — success paths
# --------------------------------------------------------------------------


class TestSearchModes:
    def test_text_search(self, client, search_mocks):
        resp = _post(client, {**BASE, "search_query": "  ball valve  "})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total_results"] == 1
        assert body["products"][0]["sku_id"] == "SKU-1"
        assert body["request_id"]  # set by the request-logging middleware
        assert "query_embedding_id" not in body and "image_embedding_id" not in body

        request_id, query, image, previous = _resolve_args(search_mocks)
        assert request_id == body["request_id"]
        uuid.UUID(request_id)
        assert (query, image, previous) == ("ball valve", None, None)

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
        search_mocks["process"].assert_not_called()

    def test_each_request_gets_a_new_request_id(self, client, search_mocks):
        first = _post(client, {**BASE, "search_query": "valve"}).json()["request_id"]
        second = _post(client, {**BASE, "search_query": "valve"}).json()["request_id"]
        assert first != second

    @pytest.mark.parametrize("data, mime", [
        (PNG_BYTES, "image/png"),
        (JPEG_BYTES, "image/jpeg"),
        (WEBP_BYTES, "image/webp"),
    ])
    def test_image_search(self, client, search_mocks, data, mime):
        resp = _post(client, BASE, files={"image": ("photo", data, mime)})
        assert resp.status_code == 200, resp.text
        # The raw upload is preprocessed; the processed bytes go to the embedder.
        search_mocks["process"].assert_called_once_with(data)
        _, query, image, _ = _resolve_args(search_mocks)
        assert query is None
        assert image == PROCESSED
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["image_vector"] == [0.3]
        assert kw["query"] is None and kw["query_vector"] is None

    def test_declared_type_may_differ_from_actual(self, client, search_mocks):
        resp = _post(client, BASE, files={"image": ("x.jpg", PNG_BYTES, "image/jpeg")})
        assert resp.status_code == 200, resp.text
        search_mocks["process"].assert_called_once_with(PNG_BYTES)

    def test_content_type_parameters_are_ignored(self, client, search_mocks):
        resp = _post(client, BASE, files={"image": ("x.png", PNG_BYTES, "image/PNG; charset=binary")})
        assert resp.status_code == 200, resp.text

    def test_text_plus_image(self, client, search_mocks):
        resp = _post(client, {**BASE, "search_query": "valve"}, files={"image": ("a.png", PNG_BYTES, "image/png")})
        assert resp.status_code == 200, resp.text
        _, query, image, _ = _resolve_args(search_mocks)
        assert (query, image) == ("valve", PROCESSED)
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["query"] == "valve" and kw["image_vector"] == [0.3]

    def test_empty_image_part_is_ignored(self, client, search_mocks):
        resp = _post(client, {**BASE, "search_query": "valve"}, files={"image": ("a.png", b"", "image/png")})
        assert resp.status_code == 200, resp.text
        search_mocks["process"].assert_not_called()
        assert _resolve_args(search_mocks)[2] is None

    def test_filter_browse(self, client, search_mocks):
        filters = {"product_type": "Valve", "price": {"lte": 500}}
        resp = _post(client, {
            **BASE, "filters": json.dumps(filters), "sort_by": "price", "sort_order": "asc",
            "page": "2", "page_size": "25",
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["request_id"]
        kw = search_mocks["filter_browse"].call_args.kwargs
        assert kw["filters"] == [
            {"bool": {
                "should": [
                    {"term": {"product_type_id": "Valve"}},
                    {"term": {"product_type.keyword": "Valve"}},
                ],
                "minimum_should_match": 1,
            }},
            {"range": {"price": {"lte": 500}}},
        ]
        assert kw["index"] == "asmo-search-api-index"
        assert kw["sort_by"] == "price" and kw["sort_order"] == "asc"
        assert (kw["page"], kw["page_size"]) == (2, 25)
        assert kw["request_id"] == resp.json()["request_id"]
        search_mocks["search_v1"].assert_not_awaited()

    def test_text_search_with_filters(self, client, search_mocks):
        resp = _post(client, {**BASE, "search_query": "valve", "filters": '{"currency": ["SAR", "USD"]}'})
        assert resp.status_code == 200, resp.text
        assert search_mocks["search_v1"].call_args.kwargs["filters"] == [{"terms": {"currency": ["SAR", "USD"]}}]

    def test_nested_seller_filter(self, client, search_mocks):
        resp = _post(client, {**BASE, "filters": '{"seller": "s1"}'})
        assert resp.status_code == 200, resp.text
        assert search_mocks["filter_browse"].call_args.kwargs["filters"] == [
            {"nested": {"path": "seller_details", "query": {"term": {"seller_details.seller_id": "s1"}}}},
        ]

    def test_relevance_sort_means_default_order(self, client, search_mocks):
        _post(client, {**BASE, "search_query": "valve", "sort_by": "relevance"})
        assert search_mocks["search_v1"].call_args.kwargs["sort_by"] is None

    def test_price_sort_is_passed_to_search(self, client, search_mocks):
        _post(client, {**BASE, "search_query": "valve", "sort_by": "price", "sort_order": "asc"})
        kw = search_mocks["search_v1"].call_args.kwargs
        assert (kw["sort_by"], kw["sort_order"]) == ("price", "asc")

    def test_mobile_channel_accepted(self, client, search_mocks):
        resp = _post(client, {"channel": "mobile", "language": "en", "search_query": "valve"})
        assert resp.status_code == 200, resp.text

    def test_empty_text_fields_are_treated_as_absent(self, client, search_mocks):
        resp = _post(client, {**BASE, "search_query": "valve", "sort_by": "", "filters": ""})
        assert resp.status_code == 200, resp.text
        assert search_mocks["search_v1"].call_args.kwargs["sort_by"] is None


class TestRealImageProcessing:
    """The image processor is not mocked here: real PIL images go through it."""

    @pytest.fixture
    def mocks(self, monkeypatch):
        resolve = AsyncMock(side_effect=_resolve)
        search_v1 = AsyncMock(side_effect=lambda **kw: make_search_response(request_id=kw["request_id"]))
        monkeypatch.setattr(router.embedding_store, "resolve", resolve)
        monkeypatch.setattr(search_service, "search_v1", search_v1)
        return resolve, search_v1

    @pytest.mark.parametrize("fmt, mime, expected_mime", [
        ("PNG", "image/png", "image/png"),
        ("JPEG", "image/jpeg", "image/jpeg"),
        ("WEBP", "image/webp", "image/jpeg"),  # non-PNG is re-encoded as JPEG
    ])
    def test_real_image_is_processed(self, client, mocks, fmt, mime, expected_mime):
        resolve, search_v1 = mocks
        resp = _post(client, BASE, files={"image": ("photo", _real_image(fmt), mime)})
        assert resp.status_code == 200, resp.text
        image = resolve.call_args.args[2]
        assert image[1] == expected_mime
        assert Image.open(BytesIO(image[0])).size == (8, 6)
        search_v1.assert_awaited_once()

    @pytest.mark.parametrize("data, mime", [(PNG_BYTES, "image/png"), (JPEG_BYTES, "image/jpeg")])
    def test_magic_bytes_only_are_not_decodable(self, client, mocks, data, mime):
        resolve, _ = mocks
        body = _error(_post(client, BASE, files={"image": ("x", data, mime)}), 400, "UNSUPPORTED_IMAGE_TYPE")
        assert body["message"] == UNSUPPORTED_IMAGE
        resolve.assert_not_awaited()


class TestRequestId:
    def test_request_id_alone_refines_previous_search(self, client, search_mocks):
        resp = _post(client, {**BASE, "request_id": PREVIOUS_ID})
        assert resp.status_code == 200, resp.text
        assert resp.json()["request_id"] == PREVIOUS_ID
        _, query, image, previous = _resolve_args(search_mocks)
        assert (query, image, previous) == (None, None, PREVIOUS_ID)
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["query"] == "stored query" and kw["query_vector"] == [0.5]
        assert kw["request_id"] == PREVIOUS_ID

    def test_request_id_with_filters_and_page(self, client, search_mocks):
        resp = _post(client, {**BASE, "request_id": PREVIOUS_ID, "filters": '{"currency": "SAR"}', "page": "3"})
        assert resp.status_code == 200, resp.text
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["filters"] == [{"term": {"currency": "SAR"}}]
        assert kw["page"] == 3

    def test_request_id_with_new_query_is_passed_through(self, client, search_mocks):
        resp = _post(client, {**BASE, "request_id": PREVIOUS_ID, "search_query": "pump"})
        assert resp.status_code == 200, resp.text
        request_id, query, _, previous = _resolve_args(search_mocks)
        assert (query, previous) == ("pump", PREVIOUS_ID)
        assert resp.json()["request_id"] == request_id != PREVIOUS_ID

    def test_reused_image_only_context_is_image_search(self, client, search_mocks):
        search_mocks["resolve"].side_effect = None
        search_mocks["resolve"].return_value = SearchContext(PREVIOUS_ID, None, None, [0.6], reused=True)
        resp = _post(client, {**BASE, "request_id": PREVIOUS_ID})
        assert resp.status_code == 200, resp.text
        kw = search_mocks["search_v1"].call_args.kwargs
        assert kw["image_vector"] == [0.6] and kw["query"] is None

    def test_reused_empty_context_is_filter_browse(self, client, search_mocks):
        search_mocks["resolve"].side_effect = None
        search_mocks["resolve"].return_value = SearchContext(PREVIOUS_ID, None, None, None, reused=True)
        resp = _post(client, {**BASE, "request_id": PREVIOUS_ID})
        assert resp.status_code == 200, resp.text
        assert search_mocks["filter_browse"].call_args.kwargs["request_id"] == PREVIOUS_ID
        search_mocks["search_v1"].assert_not_awaited()

    def test_unknown_request_id(self, client, search_mocks):
        search_mocks["resolve"].side_effect = RequestIdNotFound("request_id 'x' was not found or has expired")
        body = _error(_post(client, {**BASE, "request_id": PREVIOUS_ID}), 400, "REQUEST_ID_NOT_FOUND")
        assert body["message"] == "request_id 'x' was not found or has expired"
        search_mocks["search_v1"].assert_not_awaited()

    def test_embedding_store_unavailable(self, client, search_mocks):
        search_mocks["resolve"].side_effect = EmbeddingStoreUnavailable("Redis is disabled")
        body = _error(_post(client, {**BASE, "request_id": PREVIOUS_ID}), 503, "EMBEDDING_STORE_UNAVAILABLE")
        assert body["message"] == (
            "request_id cannot be resolved (Redis is disabled); send search_query / image instead."
        )


# --------------------------------------------------------------------------
# POST /v1/search — validation errors
# --------------------------------------------------------------------------


class TestSearchValidation:
    @pytest.mark.parametrize("kwargs", [
        {"data": {**BASE, "search_query": "valve"}},  # urlencoded
        {"json": {**BASE, "search_query": "valve"}},
        {"content": b"channel=web", "headers": {"content-type": "text/plain"}},
    ])
    def test_non_multipart_body(self, client, search_mocks, kwargs):
        body = _error(client.post("/v1/search", **kwargs), 400, "INVALID_CONTENT_TYPE")
        assert body["message"] == "The request must be sent as multipart/form-data."
        search_mocks["resolve"].assert_not_awaited()

    @pytest.mark.parametrize("field", ["channel", "language"])
    def test_missing_required_field(self, client, search_mocks, field):
        data = {**BASE, "search_query": "valve"}
        del data[field]
        body = _error(_post(client, data), 400, "VALIDATION_ERROR")
        assert body["message"].startswith(f"{field}:")

    @pytest.mark.parametrize("field, value, code", [
        ("channel", "tv", "UNSUPPORTED_CHANNEL"),
        ("language", "fr", "UNSUPPORTED_LANGUAGE"),
        ("language", "ar", "UNSUPPORTED_LANGUAGE"),  # only English is supported
        ("sort_by", "name", "INVALID_SORT_BY"),
        ("sort_order", "up", "INVALID_SORT_ORDER"),
    ])
    def test_unsupported_enum_values(self, client, search_mocks, field, value, code):
        body = _error(_post(client, {**BASE, "search_query": "valve", field: value}), 400, code)
        assert f"'{value}'" in body["message"]
        search_mocks["resolve"].assert_not_awaited()
        search_mocks["search_v1"].assert_not_awaited()

    @pytest.mark.parametrize("field, value", [
        ("page", "0"), ("page", "-1"), ("page", "abc"), ("page_size", "0"), ("page_size", "201"),
        ("page_size", "1.5"),
    ])
    def test_invalid_pagination(self, client, search_mocks, field, value):
        body = _error(_post(client, {**BASE, "search_query": "v", field: value}), 400, "VALIDATION_ERROR")
        assert body["message"].startswith(f"{field}:")

    @pytest.mark.parametrize("page_size", ["1", "200"])
    def test_page_size_bounds_allowed(self, client, search_mocks, page_size):
        resp = _post(client, {**BASE, "search_query": "v", "page_size": page_size})
        assert resp.status_code == 200, resp.text
        assert search_mocks["search_v1"].call_args.kwargs["page_size"] == int(page_size)

    @pytest.mark.parametrize("data", [
        BASE,
        {**BASE, "search_query": "   "},
        {**BASE, "filters": "{}"},
        {**BASE, "filters": "   "},
    ])
    def test_empty_request(self, client, search_mocks, data):
        body = _error(_post(client, data), 400, "EMPTY_REQUEST")
        assert body["message"] == "At least one of search_query, image, filters or request_id must be provided"
        search_mocks["resolve"].assert_not_awaited()
        search_mocks["filter_browse"].assert_not_awaited()

    @pytest.mark.parametrize("filters, code", [
        ("{not json", "INVALID_FILTER"),
        ('["price"]', "INVALID_FILTER"),
        ('{"colour": "red"}', "INVALID_FILTER_KEY"),
        ('{"price": "cheap"}', "INVALID_FILTER_VALUE"),
        ('{"contract_indicator": "yes"}', "INVALID_FILTER_VALUE"),
        ('{"currency": {"gte": 1}}', "INVALID_FILTER_VALUE"),
    ])
    def test_invalid_filters(self, client, search_mocks, filters, code):
        _error(_post(client, {**BASE, "filters": filters}), 400, code)
        search_mocks["resolve"].assert_not_awaited()

    def test_invalid_filters_rejected_with_query_too(self, client, search_mocks):
        _error(_post(client, {**BASE, "search_query": "valve", "filters": '{"colour": "red"}'}), 400, "INVALID_FILTER_KEY")
        search_mocks["search_v1"].assert_not_awaited()

    @pytest.mark.parametrize("declared", ["text/plain", "application/octet-stream"])
    def test_non_image_declared_type(self, client, search_mocks, declared):
        resp = _post(client, BASE, files={"image": ("a.bin", PNG_BYTES, declared)})
        body = _error(resp, 400, "UNSUPPORTED_IMAGE_TYPE")
        assert body["message"] == NOT_AN_IMAGE_PART
        search_mocks["process"].assert_not_called()

    def test_gif_declared_type_is_unsupported(self, client, search_mocks):
        resp = _post(client, BASE, files={"image": ("a.gif", GIF_BYTES, "image/gif")})
        body = _error(resp, 400, "UNSUPPORTED_IMAGE_TYPE")
        assert body["message"] == UNSUPPORTED_IMAGE
        search_mocks["process"].assert_not_called()

    def test_gif_bytes_with_supported_declared_type(self, client, search_mocks):
        resp = _post(client, BASE, files={"image": ("a.png", GIF_BYTES, "image/png")})
        body = _error(resp, 400, "UNSUPPORTED_IMAGE_TYPE")
        assert body["message"] == UNSUPPORTED_IMAGE
        search_mocks["process"].assert_not_called()

    def test_bytes_that_are_not_an_image(self, client, search_mocks):
        resp = _post(client, BASE, files={"image": ("a.png", b"definitely not a png", "image/png")})
        _error(resp, 400, "UNSUPPORTED_IMAGE_TYPE")
        search_mocks["process"].assert_not_called()
        search_mocks["resolve"].assert_not_awaited()

    def test_undecodable_image(self, client, search_mocks):
        search_mocks["process"].side_effect = InvalidImageError("cannot identify image file")
        resp = _post(client, BASE, files={"image": ("a.png", PNG_BYTES, "image/png")})
        body = _error(resp, 400, "UNSUPPORTED_IMAGE_TYPE")
        assert body["message"] == UNSUPPORTED_IMAGE
        search_mocks["resolve"].assert_not_awaited()

    def test_unexpected_image_processing_error_is_500(self, client, search_mocks):
        search_mocks["process"].side_effect = RuntimeError("pillow exploded")
        resp = _post(client, BASE, files={"image": ("a.png", PNG_BYTES, "image/png")})
        body = _error(resp, 500, "INTERNAL_ERROR")
        assert body["message"] == "pillow exploded"
        search_mocks["resolve"].assert_not_awaited()

    @pytest.mark.parametrize("files", [
        {"image": (None, PNG_BYTES)},  # no filename -> a text field
        {"image": (None, "plain text ✓".encode("utf-8"))},
        {"image": (None, b"iVBORw0KGgo=")},  # Base64 text
    ])
    def test_image_sent_as_text_field(self, client, search_mocks, files):
        body = _error(_post(client, BASE, files=files), 400, "UNSUPPORTED_IMAGE_TYPE")
        assert body["message"] == NOT_AN_IMAGE_PART
        search_mocks["process"].assert_not_called()

    def test_oversized_image(self, client, search_mocks):
        big = PNG_BYTES + b"\x00" * MAX_IMAGE_BYTES
        resp = _post(client, BASE, files={"image": ("big.png", big, "image/png")})
        body = _error(resp, 413, "IMAGE_TOO_LARGE")
        assert body["message"] == f"Image exceeds the maximum size of {MAX_IMAGE_BYTES} bytes."
        search_mocks["process"].assert_not_called()

    def test_image_exactly_at_limit_is_accepted(self, client, search_mocks):
        exact = PNG_BYTES + b"\x00" * (MAX_IMAGE_BYTES - len(PNG_BYTES))
        resp = _post(client, BASE, files={"image": ("ok.png", exact, "image/png")})
        assert resp.status_code == 200, resp.text
        search_mocks["process"].assert_called_once_with(exact)

    def test_huge_part_without_filename_is_413(self, client, search_mocks):
        huge = PNG_BYTES + b"\x00" * (2 * MAX_IMAGE_BYTES + 1)
        resp = _post(client, BASE, files={"image": (None, huge)})
        _error(resp, 413, "IMAGE_TOO_LARGE")

    def test_malformed_multipart_body(self, client, search_mocks):
        resp = client.post("/v1/search", content=b"garbage", headers={"content-type": "multipart/form-data"})
        body = _error(resp, 400, "VALIDATION_ERROR")
        assert body["message"].startswith("Malformed multipart body: ")


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
        body = _error(_post(client, {**BASE, "search_query": "v"}), status, code)
        assert body["message"] == str(error)

    @pytest.mark.parametrize("error, status, code", [
        (IndexNotFoundError("missing"), 404, "INDEX_NOT_FOUND"),
        (OpenSearchQueryError("bad"), 400, "QUERY_ERROR"),
        (ValueError("boom"), 500, "INTERNAL_ERROR"),
    ])
    def test_filter_browse_errors(self, client, search_mocks, error, status, code):
        search_mocks["filter_browse"].side_effect = error
        body = _error(_post(client, {**BASE, "filters": '{"currency": "SAR"}'}), status, code)
        assert body["message"] == str(error)

    def test_embedding_failure_is_500(self, client, search_mocks):
        search_mocks["resolve"].side_effect = RuntimeError("OCI down")
        body = _error(_post(client, {**BASE, "search_query": "v"}), 500, "INTERNAL_ERROR")
        assert body["message"] == "OCI down"
        search_mocks["search_v1"].assert_not_awaited()


# --------------------------------------------------------------------------
# GET /v1/search/suggestions
# --------------------------------------------------------------------------


@pytest.fixture
def suggestions_mock(monkeypatch):
    from models.response import SuggestionsResponse

    mock = AsyncMock(return_value=SuggestionsResponse(
        query="val",
        query_suggestions=[{"text": "Valve"}],
        product_suggestions=[{"text": "Ball Valve", "sku_id": "SKU-1"}],
    ))
    monkeypatch.setattr(search_service, "get_suggestions", mock)
    return mock


SUGGEST = {"q": "val", "channel": "web", "language": "en"}


class TestSuggestions:
    def test_success(self, client, suggestions_mock):
        resp = client.get("/v1/search/suggestions", params={**SUGGEST, "query_limit": 3, "product_limit": 7})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "query": "val",
            "query_suggestions": [{"text": "Valve"}],
            "product_suggestions": [{"text": "Ball Valve", "sku_id": "SKU-1"}],
        }
        suggestions_mock.assert_awaited_once_with(
            index="asmo-search-api-index", query="val", language="en", query_limit=3, product_limit=7,
        )

    def test_default_limits(self, client, suggestions_mock):
        client.get("/v1/search/suggestions", params=SUGGEST)
        kw = suggestions_mock.call_args.kwargs
        assert kw["query_limit"] == 5 and kw["product_limit"] == 5

    @pytest.mark.parametrize("limit", [1, 50])
    def test_limit_bounds_allowed(self, client, suggestions_mock, limit):
        resp = client.get("/v1/search/suggestions", params={**SUGGEST, "query_limit": limit, "product_limit": limit})
        assert resp.status_code == 200, resp.text

    @pytest.mark.parametrize("field, value, code", [
        ("channel", "tv", "UNSUPPORTED_CHANNEL"),
        ("language", "fr", "UNSUPPORTED_LANGUAGE"),
        ("language", "ar", "UNSUPPORTED_LANGUAGE"),
    ])
    def test_unsupported_enums(self, client, suggestions_mock, field, value, code):
        _error(client.get("/v1/search/suggestions", params={**SUGGEST, field: value}), 400, code)
        suggestions_mock.assert_not_awaited()

    @pytest.mark.parametrize("params, loc", [
        ({"channel": "web", "language": "en"}, "query.q"),
        ({**SUGGEST, "q": ""}, "query.q"),
        ({"q": "val", "language": "en"}, "query.channel"),
        ({"q": "val", "channel": "web"}, "query.language"),
        ({**SUGGEST, "query_limit": 0}, "query.query_limit"),
        ({**SUGGEST, "product_limit": 51}, "query.product_limit"),
        ({**SUGGEST, "query_limit": "many"}, "query.query_limit"),
    ])
    def test_validation_errors(self, client, suggestions_mock, params, loc):
        body = _error(client.get("/v1/search/suggestions", params=params), 400, "VALIDATION_ERROR")
        assert body["message"].startswith(f"{loc}:")
        suggestions_mock.assert_not_awaited()

    @pytest.mark.parametrize("error, status, code", [
        (IndexNotFoundError("missing"), 404, "INDEX_NOT_FOUND"),
        (OpenSearchQueryError("bad"), 400, "QUERY_ERROR"),
        (RuntimeError("boom"), 500, "INTERNAL_ERROR"),
    ])
    def test_backend_errors(self, client, suggestions_mock, error, status, code):
        suggestions_mock.side_effect = error
        body = _error(client.get("/v1/search/suggestions", params=SUGGEST), status, code)
        assert body["message"] == str(error)


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

    def test_list_indices_empty(self, client, monkeypatch):
        monkeypatch.setattr(search_service._os, "list_indices", AsyncMock(return_value=[]))
        assert client.get("/indices").json() == {"indices": [], "total": 0}

    def test_list_indices_failure(self, client, monkeypatch):
        monkeypatch.setattr(search_service._os, "list_indices", AsyncMock(side_effect=ConnectionError("down")))
        body = _error(client.get("/indices"), 500, "INTERNAL_ERROR")
        assert body["message"] == "down"

    def test_bad_index_entry_is_500(self, client, monkeypatch):
        monkeypatch.setattr(search_service._os, "list_indices", AsyncMock(return_value=[{"name": "x"}]))
        _error(client.get("/indices"), 500, "INTERNAL_ERROR")
