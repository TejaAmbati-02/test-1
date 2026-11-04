from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.response import (
    ApiError,
    ErrorResponse,
    FacetTermValue,
    HealthResponse,
    LatencyResponse,
    ProductItem,
    SearchApiResponse,
    SearchFacets,
    SellerDetail,
    SuggestionsResponse,
)


def test_product_item_defaults():
    p = ProductItem()
    assert p.sku_id is None and p.sku_name is None
    assert p.seller_details == []


def test_product_item_nests_seller_details():
    p = ProductItem(seller_details=[{"name": "Acme", "price": 9.5, "moq": 2}])
    assert isinstance(p.seller_details[0], SellerDetail)
    assert p.seller_details[0].price == 9.5


def test_seller_moq_defaults_to_empty_string():
    assert SellerDetail().moq == ""


def test_search_response_defaults_empty_facets():
    r = SearchApiResponse(total_results=0, page=1, page_size=10, products=[])
    assert r.facets == SearchFacets()
    assert r.facets.category == [] and r.facets.seller == []
    assert r.request_id is None


def test_search_response_requires_core_fields():
    with pytest.raises(ValidationError):
        SearchApiResponse(page=1, page_size=10, products=[])


def test_facet_term_value_requires_all_fields():
    with pytest.raises(ValidationError):
        FacetTermValue(value="x", count=1)


def test_suggestions_response_shape():
    r = SuggestionsResponse(
        query="val",
        query_suggestions=[{"text": "Valve"}],
        product_suggestions=[{"text": "Valve", "sku_id": "SKU-1"}],
    )
    assert r.product_suggestions[0].sku_id == "SKU-1"


def test_health_response_redis_optional():
    h = HealthResponse(status="ok", opensearch_connected=True, embedding_model_loaded=True, embedding_model_name="m")
    assert h.redis_connected is None
    assert h.opensearch_cluster is None


def test_latency_and_error_models():
    lat = LatencyResponse(target="oci", endpoint="e", samples=1, samples_ms=[1.0], min_ms=1, avg_ms=1, max_ms=1, ok=True)
    assert lat.error is None
    assert ErrorResponse(code="X", message="y").model_dump() == {"code": "X", "message": "y"}


def test_api_error_carries_the_error_contract():
    exc = ApiError(418, "TEAPOT", "short and stout", {"X-Extra": "1"})
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 418
    assert exc.detail == {"code": "TEAPOT", "message": "short and stout"}
    assert exc.headers == {"X-Extra": "1"}
