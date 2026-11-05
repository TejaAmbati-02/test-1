"""Endpoint tests for /v1/search, /v1/search/suggestions and /indices."""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from routers.search import MAX_IMAGE_BYTES
from models.response import ApiError
from services.embedding_store import embedding_store
from services.opensearch import os_client
from services.search import search_service
from tests.conftest import GIF_BYTES, PNG_BYTES, WEBP_BYTES, make_search_response, multipart

BASE = {"channel": "web", "language": "en"}
PREVIOUS_ID = "22222222-2222-4222-8222-222222222222"


def _real_image(fmt: str) -> bytes:
    out = BytesIO()
    img = Image.new("RGB", (8, 8), "blue")
    img.paste("yellow", (2, 2, 6, 6))  # some content: a plain colour counts as blank
    img.save(out, format=fmt)
    return out.getvalue()


REAL_PNG, REAL_JPEG = _real_image("PNG"), _real_image("JPEG")


@pytest.fixture
def mocks(monkeypatch):
    """Mock the embedding store and search service used by the router."""
    async def resolve(request_id, query, image, previous_request_id):
        if previous_request_id and not query and image is None:
            return {"request_id": previous_request_id, "query": "stored query", "query_vector": [0.5],
                    "image_vector": None, "reused": True}
        return {"request_id": request_id, "query": query, "query_vector": [0.1] if query else None,
                "image_vector": [0.3] if image else None, "reused": False}

    m = {
        "resolve": AsyncMock(side_effect=resolve),
        "search": AsyncMock(side_effect=lambda **kw: make_search_response(request_id=kw["request_id"])),
    }
    monkeypatch.setattr(embedding_store, "resolve", m["resolve"])
    monkeypatch.setattr(search_service, "search", m["search"])
    return m


def _post(client, data=None, files=None):
    return client.post("/v1/search", files=multipart(data, files))


def _error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body == {"code": code, "message": body["message"]}
    return body


class TestSearchModes:
    @pytest.mark.parametrize("text", ["p", "pu", " pu "])
    def test_text_under_3_characters_is_not_searched(self, client, mocks, text):
        resp = _post(client, {**BASE, "search_query": text})
        assert resp.status_code == 200
        assert (resp.json()["total_results"], resp.json()["products"]) == (0, [])
        mocks["resolve"].assert_not_awaited()  # no embedding call
        mocks["search"].assert_not_awaited()   # no OpenSearch call

    def test_blank_image_alone_is_not_searched(self, client, mocks):
        out = BytesIO()
        Image.new("RGB", (499, 600), (138, 255, 46)).save(out, format="PNG")  # plain green
        resp = _post(client, BASE, {"image": ("green.png", out.getvalue(), "image/png")})
        assert resp.status_code == 200
        assert (resp.json()["total_results"], resp.json()["products"]) == (0, [])
        mocks["resolve"].assert_not_awaited()  # no embedding call
        mocks["search"].assert_not_awaited()

    def test_blank_image_with_text_searches_the_text_only(self, client, mocks):
        out = BytesIO()
        Image.new("RGB", (100, 100), "white").save(out, format="PNG")
        assert _post(client, {**BASE, "search_query": "valve"}, {"image": ("w.png", out.getvalue(), "image/png")}).status_code == 200
        _, query, image, _ = mocks["resolve"].call_args.args
        assert (query, image) == ("valve", None)

    def test_short_text_with_image_searches_the_image_only(self, client, mocks):
        resp = _post(client, {**BASE, "search_query": "pu"}, {"image": ("a.png", REAL_PNG, "image/png")})
        assert resp.status_code == 200
        _, query, image, _ = mocks["resolve"].call_args.args
        assert query is None and image is not None

    def test_short_text_with_filters_browses_the_filters(self, client, mocks):
        assert _post(client, {**BASE, "search_query": "pu", "filters": '{"currency": "SAR"}'}).status_code == 200
        assert mocks["search"].call_args.kwargs["query"] is None
        assert mocks["search"].call_args.kwargs["filters"] == [{"term": {"currency": "SAR"}}]

    def test_text_search(self, client, mocks):
        resp = _post(client, {**BASE, "search_query": "  ball valve  "})
        assert resp.status_code == 200
        body = resp.json()
        assert body["products"][0]["sku_id"] == "SKU-1"
        assert body["request_id"]  # set by the request-logging middleware

        request_id, query, image, previous = mocks["resolve"].call_args.args
        assert (request_id, query, image, previous) == (body["request_id"], "ball valve", None, None)
        assert mocks["search"].call_args.kwargs == {
            "query": "ball valve", "query_vector": [0.1], "image_vector": None, "filters": [],
            "sort_by": None, "sort_order": "desc", "page": 1, "page_size": 60, "request_id": body["request_id"],
            "language": "en",
        }

    @pytest.mark.parametrize("data, declared, mime", [
        (REAL_PNG, "image/png", "image/png"),
        (REAL_JPEG, "image/jpeg", "image/jpeg"),
        (REAL_PNG, "image/jpeg", "image/png"),  # the bytes decide the format, not the header
    ])
    def test_image_search(self, client, mocks, data, declared, mime):
        resp = _post(client, BASE, {"image": ("photo", data, declared)})
        assert resp.status_code == 200
        image = mocks["resolve"].call_args.args[2]
        assert image[1] == mime
        assert mocks["search"].call_args.kwargs["image_vector"] == [0.3]

    def test_text_plus_image(self, client, mocks):
        resp = _post(client, {**BASE, "search_query": "valve"}, {"image": ("a.png", REAL_PNG, "image/png")})
        assert resp.status_code == 200
        kw = mocks["search"].call_args.kwargs
        assert kw["query"] == "valve" and kw["image_vector"] == [0.3]

    def test_empty_image_part_is_ignored(self, client, mocks):
        resp = _post(client, {**BASE, "search_query": "valve"}, {"image": ("a.png", b"", "image/png")})
        assert resp.status_code == 200
        assert mocks["resolve"].call_args.args[2] is None

    def test_filter_browse(self, client, mocks):
        resp = _post(client, {
            **BASE, "filters": json.dumps({"currency": "SAR", "price": {"lte": 500}}),
            "sort_by": "price", "sort_order": "asc", "page": "2", "page_size": "25",
        })
        assert resp.status_code == 200
        kw = mocks["search"].call_args.kwargs
        assert kw["filters"] == [{"term": {"currency": "SAR"}}, {"range": {"price": {"lte": 500}}}]
        assert (kw["query"], kw["image_vector"]) == (None, None)
        assert (kw["sort_by"], kw["sort_order"], kw["page"], kw["page_size"]) == ("price", "asc", 2, 25)

    def test_refine_with_request_id(self, client, mocks):
        resp = _post(client, {**BASE, "request_id": PREVIOUS_ID, "filters": '{"currency": "SAR"}'})
        assert resp.status_code == 200
        assert resp.json()["request_id"] == PREVIOUS_ID  # refining keeps the same id
        assert mocks["resolve"].call_args.args[3] == PREVIOUS_ID
        assert mocks["search"].call_args.kwargs["query"] == "stored query"

    def test_empty_text_fields_are_treated_as_absent(self, client, mocks):
        resp = _post(client, {**BASE, "search_query": "valve", "sort_by": "", "filters": ""})
        assert resp.status_code == 200
        assert mocks["search"].call_args.kwargs["sort_by"] is None

    def test_mobile_channel(self, client, mocks):
        assert _post(client, {"channel": "mobile", "language": "en", "search_query": "valve"}).status_code == 200

    def test_german_is_passed_through(self, client, mocks):
        assert _post(client, {"channel": "web", "language": "de", "search_query": "Ventil"}).status_code == 200
        assert mocks["search"].call_args.kwargs["language"] == "de"

    def test_max_page_size_allowed(self, client, mocks):
        assert _post(client, {**BASE, "search_query": "valve", "page_size": "200"}).status_code == 200


class TestSearchValidation:
    @pytest.mark.parametrize("field", ["channel", "language"])
    def test_missing_required_field(self, client, mocks, field):
        data = {**BASE, "search_query": "valve"}
        del data[field]
        assert _error(_post(client, data), 400, "VALIDATION_ERROR")["message"].startswith(f"{field}:")

    @pytest.mark.parametrize("field, value, code", [
        ("channel", "tv", "UNSUPPORTED_CHANNEL"),
        ("language", "ar", "UNSUPPORTED_LANGUAGE"),
        ("sort_by", "name", "INVALID_SORT_BY"),
        ("sort_order", "up", "INVALID_SORT_ORDER"),
    ])
    def test_unsupported_enum_values(self, client, mocks, field, value, code):
        _error(_post(client, {**BASE, "search_query": "valve", field: value}), 400, code)
        mocks["search"].assert_not_awaited()

    @pytest.mark.parametrize("field, value", [("page", "0"), ("page", "abc"), ("page_size", "0"), ("page_size", "201")])
    def test_invalid_pagination(self, client, mocks, field, value):
        body = _error(_post(client, {**BASE, "search_query": "valve", field: value}), 400, "VALIDATION_ERROR")
        assert body["message"].startswith(f"{field}:")

    @pytest.mark.parametrize("data", [BASE, {**BASE, "search_query": "   "}, {**BASE, "filters": "{}"}])
    def test_empty_request(self, client, mocks, data):
        _error(_post(client, data), 400, "EMPTY_REQUEST")
        mocks["resolve"].assert_not_awaited()

    @pytest.mark.parametrize("filters, code", [
        ("{not json", "INVALID_FILTER"),
        ('["price"]', "INVALID_FILTER"),
        ('{"colour": "red"}', "INVALID_FILTER_KEY"),
        ('{"price": "cheap"}', "INVALID_FILTER_VALUE"),
    ])
    def test_invalid_filters(self, client, mocks, filters, code):
        _error(_post(client, {**BASE, "filters": filters}), 400, code)

    def test_must_be_multipart(self, client, mocks):
        _error(client.post("/v1/search", data=BASE), 400, "INVALID_CONTENT_TYPE")
        _error(client.post("/v1/search", json=BASE), 400, "INVALID_CONTENT_TYPE")

    def test_malformed_multipart_body(self, client, mocks):
        resp = client.post("/v1/search", content=b"garbage", headers={"content-type": "multipart/form-data"})
        _error(resp, 400, "VALIDATION_ERROR")

    @pytest.mark.parametrize("part, expected", [
        (("a.txt", PNG_BYTES, "text/plain"), "multipart file part"),            # not an image part
        (("a.bin", PNG_BYTES, "application/octet-stream"), "multipart file part"),
        ((None, "aGVsbG8="), "multipart file part"),                            # Base64 text field
        (("a.gif", GIF_BYTES, "image/gif"), "Only JPEG, PNG and WebP"),         # unsupported image type
        (("a.png", GIF_BYTES, "image/png"), "Only JPEG, PNG and WebP"),         # GIF bytes, PNG header
        (("a.png", b"definitely not a png", "image/png"), "Only JPEG, PNG and WebP"),
        (("a.webp", WEBP_BYTES, "image/webp"), "Only JPEG, PNG and WebP"),      # right magic, undecodable
    ])
    def test_rejected_images(self, client, mocks, part, expected):
        body = _error(_post(client, BASE, {"image": part}), 400, "UNSUPPORTED_IMAGE_TYPE")
        assert expected in body["message"]
        mocks["resolve"].assert_not_awaited()

    def test_oversized_image(self, client, mocks):
        big = PNG_BYTES + b"\x00" * MAX_IMAGE_BYTES
        _error(_post(client, BASE, {"image": ("big.png", big, "image/png")}), 413, "IMAGE_TOO_LARGE")

    def test_huge_file_upload_is_413_and_never_touches_disk(self, client, mocks, monkeypatch):
        # Like a read-only container: any attempt to create a temp file fails.
        import tempfile

        def no_disk(*args, **kwargs):
            raise FileNotFoundError("No usable temporary directory found in ['/tmp', '/var/tmp', '/usr/tmp', '/app']")
        monkeypatch.setattr(tempfile, "TemporaryFile", no_disk)
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", no_disk)
        huge = PNG_BYTES + b"\x00" * (2 * MAX_IMAGE_BYTES + 1)  # bigger than the 20 MiB in-memory limit
        _error(_post(client, BASE, {"image": ("huge.png", huge, "image/png")}), 413, "IMAGE_TOO_LARGE")
        mocks["resolve"].assert_not_awaited()

    def test_huge_part_without_filename_is_413(self, client, mocks):
        huge = PNG_BYTES + b"\x00" * (2 * MAX_IMAGE_BYTES + 1)
        _error(_post(client, BASE, {"image": (None, huge)}), 413, "IMAGE_TOO_LARGE")


class TestSearchErrors:
    @pytest.mark.parametrize("error, status, code", [
        (ApiError(400, "REQUEST_ID_NOT_FOUND", "request_id 'x' was not found or its search has expired"), 400, "REQUEST_ID_NOT_FOUND"),
        (ApiError(503, "EMBEDDING_STORE_UNAVAILABLE", "Redis is disabled"), 503, "EMBEDDING_STORE_UNAVAILABLE"),
        (RuntimeError("OCI down"), 500, "INTERNAL_ERROR"),
    ])
    def test_resolve_errors(self, client, mocks, error, status, code):
        mocks["resolve"].side_effect = error
        body = _error(_post(client, {**BASE, "request_id": PREVIOUS_ID}), status, code)
        assert body["message"] == (error.detail["message"] if isinstance(error, ApiError) else str(error))

    @pytest.mark.parametrize("error, status, code", [
        (ApiError(404, "INDEX_NOT_FOUND", "Index 'x' not found"), 404, "INDEX_NOT_FOUND"),
        (ApiError(400, "QUERY_ERROR", "bad query"), 400, "QUERY_ERROR"),
        (RuntimeError("kaboom"), 500, "INTERNAL_ERROR"),
    ])
    def test_search_errors(self, client, mocks, error, status, code):
        mocks["search"].side_effect = error
        body = _error(_post(client, {**BASE, "search_query": "valve"}), status, code)
        assert body["message"] == (error.detail["message"] if isinstance(error, ApiError) else str(error))


@pytest.fixture
def suggestions_mock(monkeypatch):
    from models.response import SuggestionsResponse

    mock = AsyncMock(return_value=SuggestionsResponse(
        query="val", query_suggestions=[{"text": "Valve"}], product_suggestions=[{"text": "Valve", "sku_id": "1"}],
    ))
    monkeypatch.setattr(search_service, "get_suggestions", mock)
    return mock


SUGGEST = {"q": "val", "channel": "web", "language": "en"}


class TestSuggestions:
    def test_success(self, client, suggestions_mock):
        resp = client.get("/v1/search/suggestions", params={**SUGGEST, "query_limit": 3, "product_limit": 7})
        assert resp.status_code == 200
        assert resp.json()["query_suggestions"][0]["text"] == "Valve"
        suggestions_mock.assert_awaited_once_with(query="val", query_limit=3, product_limit=7, language="en")

    @pytest.mark.parametrize("q", ["p", "pu", " pu "])
    def test_under_3_characters_gives_no_suggestions(self, client, suggestions_mock, q):
        resp = client.get("/v1/search/suggestions", params={**SUGGEST, "q": q})
        assert resp.status_code == 200
        assert (resp.json()["query_suggestions"], resp.json()["product_suggestions"]) == ([], [])
        suggestions_mock.assert_not_awaited()

    def test_german_is_passed_through(self, client, suggestions_mock):
        assert client.get("/v1/search/suggestions", params={**SUGGEST, "language": "de"}).status_code == 200
        assert suggestions_mock.call_args.kwargs["language"] == "de"

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
        (ApiError(404, "INDEX_NOT_FOUND", "missing"), 404, "INDEX_NOT_FOUND"),
        (ApiError(400, "QUERY_ERROR", "bad"), 400, "QUERY_ERROR"),
        (RuntimeError("boom"), 500, "INTERNAL_ERROR"),
    ])
    def test_backend_errors(self, client, suggestions_mock, error, status, code):
        suggestions_mock.side_effect = error
        _error(client.get("/v1/search/suggestions", params=SUGGEST), status, code)


class TestIndices:
    def test_list_indices(self, client, monkeypatch):
        monkeypatch.setattr(os_client, "list_indices", AsyncMock(return_value=[
            {"name": "products", "doc_count": 10, "size_bytes": 2048, "status": "open"},
        ]))
        resp = client.get("/indices")
        assert resp.status_code == 200
        assert resp.json() == {
            "indices": [{"name": "products", "doc_count": 10, "size_bytes": 2048, "status": "open"}],
            "total": 1,
        }

    def test_list_indices_failure(self, client, monkeypatch):
        monkeypatch.setattr(os_client, "list_indices", AsyncMock(side_effect=ConnectionError("down")))
        assert _error(client.get("/indices"), 500, "INTERNAL_ERROR")["message"] == "down"
