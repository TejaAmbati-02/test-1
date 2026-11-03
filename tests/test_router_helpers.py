from __future__ import annotations

import pytest
from fastapi import HTTPException

from routers.search import (
    _api_error,
    _build_os_filters,
    _image_mime,
    _parse_filters,
    _valid_scalar,
    _validate_enum,
    _validate_filter_value,
)
from tests.conftest import GIF_BYTES, JPEG_BYTES, PNG_BYTES, WEBP_BYTES


def _code(exc_info) -> str:
    return exc_info.value.detail["code"]


@pytest.mark.parametrize("data, mime", [
    (PNG_BYTES, "image/png"),
    (JPEG_BYTES, "image/jpeg"),
    (WEBP_BYTES, "image/webp"),
    # GIF is not a supported upload format.
    (GIF_BYTES, None),
    (b"GIF87a" + b"\x00" * 8, None),
    (b"RIFF\x00\x00\x00\x00WAVE", None),
    (b"\xff\xd8", None),  # truncated JPEG marker
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


def test_validate_enum_is_case_sensitive():
    with pytest.raises(HTTPException) as exc:
        _validate_enum("EN", frozenset({"en"}), "language", "UNSUPPORTED_LANGUAGE")
    assert _code(exc) == "UNSUPPORTED_LANGUAGE"
    assert exc.value.detail["message"] == "Unsupported language 'EN'. Allowed: en"


class TestParseFilters:
    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_empty(self, raw):
        assert _parse_filters(raw) == {}

    def test_object(self):
        assert _parse_filters('{"currency": "SAR"}') == {"currency": "SAR"}

    @pytest.mark.parametrize("raw", ["{bad json", '["a"]', '"text"', "42", "null"])
    def test_invalid(self, raw):
        with pytest.raises(HTTPException) as exc:
            _parse_filters(raw)
        assert exc.value.status_code == 400
        assert _code(exc) == "INVALID_FILTER"

    def test_invalid_json_message(self):
        with pytest.raises(HTTPException) as exc:
            _parse_filters("{bad json")
        assert exc.value.detail["message"].startswith("filters must be a valid JSON object: ")

    def test_non_object_message(self):
        with pytest.raises(HTTPException) as exc:
            _parse_filters('["a"]')
        assert exc.value.detail["message"] == "filters must be a JSON object"


class TestFilterValues:
    @pytest.mark.parametrize("key, value, ok", [
        ("price", 10, True),
        ("price", 9.5, True),
        ("price", True, False),  # bools are not numbers here
        ("price", "10", False),
        ("lead_time", 3, True),
        ("moq", None, False),
        ("contract_indicator", True, True),
        ("contract_indicator", 1, False),
        ("under_agreement", False, True),
        ("temporary_unavailable", "false", False),
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
        ("lead_time", {"lte": 7}),
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
        ("price", {"gte": 5, "eq": 6}),
        ("currency", {"gte": "A"}),  # ranges only on numeric fields
        ("contract_indicator", {"gte": True}),
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

    def test_reject_message_contains_json_value(self):
        with pytest.raises(HTTPException) as exc:
            _validate_filter_value("price", {"gte": "5"})
        assert exc.value.detail["message"] == "Invalid value for filter 'price': {\"gte\": \"5\"}"


class TestBuildOsFilters:
    def test_builds_term_terms_and_range_clauses(self):
        clauses = _build_os_filters({
            "sold_by": "ACME",
            "category_ids": ["c1", "c2"],
            "price": {"gte": 10, "lte": 100},
            "contract_indicator": True,
        })
        assert clauses == [
            {"term": {"sold_by.keyword": "ACME"}},
            {"terms": {"category_ids": ["c1", "c2"]}},
            {"range": {"price": {"gte": 10, "lte": 100}}},
            {"term": {"contract_indicator": True}},
        ]

    def test_facet_keys_map_to_id_fields(self):
        assert _build_os_filters({"category": ["c1"], "sub_category": "s1"}) == [
            {"terms": {"category_ids": ["c1"]}},
            {"term": {"sub_category_id": "s1"}},
        ]

    def test_legacy_facet_names_still_accepted(self):
        assert _build_os_filters({"category_name": "c1", "sub_category_name": "s1"}) == [
            {"term": {"category_ids": "c1"}},
            {"term": {"sub_category_id": "s1"}},
        ]

    def test_product_type_matches_id_or_name(self):
        assert _build_os_filters({"product_type": "Valve"}) == [
            {"bool": {
                "should": [
                    {"term": {"product_type_id": "Valve"}},
                    {"term": {"product_type.keyword": "Valve"}},
                ],
                "minimum_should_match": 1,
            }},
        ]

    def test_product_type_list(self):
        assert _build_os_filters({"product_type": ["a", "b"]}) == [
            {"bool": {
                "should": [
                    {"terms": {"product_type_id": ["a", "b"]}},
                    {"terms": {"product_type.keyword": ["a", "b"]}},
                ],
                "minimum_should_match": 1,
            }},
        ]

    @pytest.mark.parametrize("key", ["seller", "sellername"])
    def test_seller_is_nested(self, key):
        assert _build_os_filters({key: ["s1", "s2"]}) == [
            {"nested": {
                "path": "seller_details",
                "query": {"terms": {"seller_details.seller_id": ["s1", "s2"]}},
            }},
        ]

    def test_lead_time_range_is_nested(self):
        assert _build_os_filters({"lead_time": {"lte": 7}}) == [
            {"nested": {
                "path": "seller_details",
                "query": {"range": {"seller_details.lead_time": {"lte": 7}}},
            }},
        ]

    def test_empty(self):
        assert _build_os_filters({}) == []

    def test_unknown_key(self):
        with pytest.raises(HTTPException) as exc:
            _build_os_filters({"colour": "red"})
        assert exc.value.status_code == 400
        assert _code(exc) == "INVALID_FILTER_KEY"
        assert exc.value.detail["message"] == "The filter key 'colour' is not supported."

    def test_invalid_value(self):
        with pytest.raises(HTTPException) as exc:
            _build_os_filters({"price": "cheap"})
        assert _code(exc) == "INVALID_FILTER_VALUE"

    def test_unknown_key_reported_even_after_valid_keys(self):
        with pytest.raises(HTTPException) as exc:
            _build_os_filters({"currency": "SAR", "colour": "red"})
        assert _code(exc) == "INVALID_FILTER_KEY"
