"""Request validation and filter building inside POST /v1/search and GET /v1/search/suggestions.

The checks run inline in the endpoints, so they are tested through the API: the search service
and embedding store are mocked, and the filter clauses the endpoint builds are read from the
search_service.search call.
"""
from unittest.mock import AsyncMock

import pytest

from services.embedding_store import embedding_store
from services.search import search_service
from tests.conftest import make_search_response, multipart


@pytest.fixture
def search_mock(monkeypatch):
    async def resolve(request_id, query, image, previous_request_id):
        return {"request_id": request_id, "query": query, "query_vector": [0.1] if query else None,
                "image_vector": None, "reused": False}

    monkeypatch.setattr(embedding_store, "resolve", AsyncMock(side_effect=resolve))
    mock = AsyncMock(side_effect=lambda **kw: make_search_response(request_id=kw["request_id"]))
    monkeypatch.setattr(search_service, "search", mock)
    return mock


def _post(client, filters: str | None = None, **fields):
    data = {"channel": "web", "language": "en", "search_query": "valve", **fields}
    if filters is not None:
        data["filters"] = filters
    return client.post("/v1/search", files=multipart(data))


def _filters(client, search_mock, raw: str | None) -> list[dict]:
    """The OpenSearch filter clauses the endpoint built from the `filters` field."""
    resp = _post(client, raw)
    assert resp.status_code == 200, resp.text
    return search_mock.call_args.kwargs["filters"]


def _rejected(client, raw: str) -> tuple[int, str, str]:
    body = _post(client, raw)
    return body.status_code, body.json()["code"], body.json()["message"]


class TestAllowedValues:
    def test_unsupported_channel_lists_the_allowed_values(self, client, search_mock):
        resp = _post(client, channel="tv")
        assert (resp.status_code, resp.json()) == (400, {"code": "UNSUPPORTED_CHANNEL",
                                                         "message": "Unsupported channel 'tv'. Allowed: mobile, web"})
        search_mock.assert_not_awaited()


class TestBuildFilters:
    @pytest.mark.parametrize("raw", [None, "   ", "{}"])
    def test_empty(self, client, search_mock, raw):
        assert _filters(client, search_mock, raw) == []

    def test_term_terms_and_range_clauses(self, client, search_mock):
        assert _filters(client, search_mock,
            '{"currency": "SAR", "category_ids": ["c1", "c2"], "price": {"gte": 10, "lte": 100}, "contract_indicator": true}'
        ) == [
            {"term": {"currency": "SAR"}},
            {"terms": {"category_ids": ["c1", "c2"]}},
            {"range": {"price": {"gte": 10, "lte": 100}}},
            {"term": {"contract_indicator": True}},
        ]

    def test_facet_keys_map_to_id_fields(self, client, search_mock):
        assert _filters(client, search_mock, '{"category": ["c1"], "sub_category": "s1"}') == [
            {"terms": {"category_ids": ["c1"]}},
            {"term": {"sub_category_id": "s1"}},
        ]

    def test_product_type_matches_id_or_name(self, client, search_mock):
        assert _filters(client, search_mock, '{"product_type": "Valve"}') == [{"bool": {"should": [
            {"term": {"product_type_id": "Valve"}},
            {"term": {"product_type.keyword": "Valve"}},
        ], "minimum_should_match": 1}}]

    def test_nested_keys_are_wrapped(self, client, search_mock):
        assert _filters(client, search_mock, '{"seller": ["s1"], "lead_time": {"lt": 9}}') == [
            {"nested": {"path": "seller_details", "query": {"terms": {"seller_details.seller_id": ["s1"]}}}},
            {"nested": {"path": "seller_details", "query": {"range": {"seller_details.lead_time": {"lt": 9}}}}},
        ]

    def test_seller_id_and_seller_name(self, client, search_mock):
        assert _filters(client, search_mock, '{"seller_id": ["s1"], "seller_name": "Acme"}') == [
            {"nested": {"path": "seller_details", "query": {"terms": {"seller_details.seller_id": ["s1"]}}}},
            {"nested": {"path": "seller_details", "query": {"term": {"seller_details.seller_name.keyword": "Acme"}}}},
        ]

    def test_sellername_matches_id_or_name(self, client, search_mock):
        assert _filters(client, search_mock, '{"sellername": ["Acme"]}') == [{"nested": {"path": "seller_details", "query": {
            "bool": {"should": [
                {"terms": {"seller_details.seller_id": ["Acme"]}},
                {"terms": {"seller_details.seller_name.keyword": ["Acme"]}},
            ], "minimum_should_match": 1},
        }}}]

    @pytest.mark.parametrize("raw", ["{bad json", '["a"]', '"text"', "42"])
    def test_not_a_json_object(self, client, search_mock, raw):
        status, code, _ = _rejected(client, raw)
        assert (status, code) == (400, "INVALID_FILTER")

    def test_unknown_key(self, client, search_mock):
        status, code, message = _rejected(client, '{"colour": "red"}')
        assert (status, code) == (400, "INVALID_FILTER_KEY")
        assert "colour" in message

    @pytest.mark.parametrize("value", ['100', '9.5', '[10, 20]', '{"gte": 50, "lte": 150}', '{"gt": 1}'])
    def test_numeric_values_accepted(self, client, search_mock, value):
        assert _filters(client, search_mock, f'{{"price": {value}}}')

    @pytest.mark.parametrize("key, value", [
        ("price", "{}"),
        ("price", '{"between": 5}'),
        ("price", '{"gte": "5"}'),
        ("price", "true"),          # bools are not numbers here
        ("price", '"10"'),
        ("price", "[]"),
        ("price", '[1, "2"]'),
        ("currency", '{"gte": "A"}'),  # ranges only on numeric fields
        ("currency", '[""]'),
        ("currency", '""'),
        ("currency", "5"),
        ("currency", "null"),
        ("contract_indicator", '"true"'),
        ("contract_indicator", "1"),
    ])
    def test_invalid_values(self, client, search_mock, key, value):
        status, code, _ = _rejected(client, f'{{"{key}": {value}}}')
        assert (status, code) == (400, "INVALID_FILTER_VALUE")
