from __future__ import annotations

import pytest
from fastapi import HTTPException

from routers.search import _build_filters, _check_allowed


def _code(exc_info) -> str:
    return exc_info.value.detail["code"]


async def test_check_allowed():
    await _check_allowed("web", frozenset({"web", "mobile"}), "channel", "UNSUPPORTED_CHANNEL")
    with pytest.raises(HTTPException) as exc:
        await _check_allowed("tv", frozenset({"web", "mobile"}), "channel", "UNSUPPORTED_CHANNEL")
    assert exc.value.status_code == 400
    assert _code(exc) == "UNSUPPORTED_CHANNEL"
    assert exc.value.detail["message"] == "Unsupported channel 'tv'. Allowed: mobile, web"


class TestBuildFilters:
    @pytest.mark.parametrize("raw", [None, "", "   ", "{}"])
    async def test_empty(self, raw):
        assert await _build_filters(raw) == []

    async def test_term_terms_and_range_clauses(self):
        assert await _build_filters(
            '{"currency": "SAR", "category_ids": ["c1", "c2"], "price": {"gte": 10, "lte": 100}, "contract_indicator": true}'
        ) == [
            {"term": {"currency": "SAR"}},
            {"terms": {"category_ids": ["c1", "c2"]}},
            {"range": {"price": {"gte": 10, "lte": 100}}},
            {"term": {"contract_indicator": True}},
        ]

    async def test_facet_keys_map_to_id_fields(self):
        assert await _build_filters('{"category": ["c1"], "sub_category": "s1"}') == [
            {"terms": {"category_ids": ["c1"]}},
            {"term": {"sub_category_id": "s1"}},
        ]

    async def test_product_type_matches_id_or_name(self):
        assert await _build_filters('{"product_type": "Valve"}') == [{"bool": {"should": [
            {"term": {"product_type_id": "Valve"}},
            {"term": {"product_type.keyword": "Valve"}},
        ], "minimum_should_match": 1}}]

    async def test_nested_keys_are_wrapped(self):
        assert await _build_filters('{"seller": ["s1"], "lead_time": {"lt": 9}}') == [
            {"nested": {"path": "seller_details", "query": {"terms": {"seller_details.seller_id": ["s1"]}}}},
            {"nested": {"path": "seller_details", "query": {"range": {"seller_details.lead_time": {"lt": 9}}}}},
        ]

    async def test_seller_id_and_seller_name(self):
        assert await _build_filters('{"seller_id": ["s1"], "seller_name": "Acme"}') == [
            {"nested": {"path": "seller_details", "query": {"terms": {"seller_details.seller_id": ["s1"]}}}},
            {"nested": {"path": "seller_details", "query": {"term": {"seller_details.seller_name.keyword": "Acme"}}}},
        ]

    async def test_sellername_matches_id_or_name(self):
        assert await _build_filters('{"sellername": ["Acme"]}') == [{"nested": {"path": "seller_details", "query": {
            "bool": {"should": [
                {"terms": {"seller_details.seller_id": ["Acme"]}},
                {"terms": {"seller_details.seller_name.keyword": ["Acme"]}},
            ], "minimum_should_match": 1},
        }}}]

    @pytest.mark.parametrize("raw", ["{bad json", '["a"]', '"text"', "42"])
    async def test_not_a_json_object(self, raw):
        with pytest.raises(HTTPException) as exc:
            await _build_filters(raw)
        assert exc.value.status_code == 400
        assert _code(exc) == "INVALID_FILTER"

    async def test_unknown_key(self):
        with pytest.raises(HTTPException) as exc:
            await _build_filters('{"colour": "red"}')
        assert _code(exc) == "INVALID_FILTER_KEY"
        assert "colour" in exc.value.detail["message"]

    @pytest.mark.parametrize("value", [
        '100', '9.5', '[10, 20]', '{"gte": 50, "lte": 150}', '{"gt": 1}',
    ])
    async def test_numeric_values_accepted(self, value):
        assert await _build_filters(f'{{"price": {value}}}')

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
    async def test_invalid_values(self, key, value):
        with pytest.raises(HTTPException) as exc:
            await _build_filters(f'{{"{key}": {value}}}')
        assert exc.value.status_code == 400
        assert _code(exc) == "INVALID_FILTER_VALUE"
