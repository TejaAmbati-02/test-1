from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.response import (
    ErrorResponse,
    FacetTermValue,
    HealthResponse,
    IndexInfo,
    IndicesResponse,
    LatencyResponse,
    ProductItem,
    ProductSuggestion,
    SearchApiResponse,
    SearchFacets,
    SellerDetail,
    SuggestionsResponse,
)


def test_product_item_defaults():
    p = ProductItem()
    assert p.model_dump() == {
        "sku_id": None,
        "sku_name": None,
        "primary_image_url": None,
        "asmo_material_number": None,
        "aramco_material_number": None,
        "seller_details": [],
    }


def test_product_item_ignores_unknown_fields():
    p = ProductItem(sku_id="S1", product_name="Valve", contract_indicator=True)
    assert p.sku_id == "S1"
    assert "product_name" not in p.model_dump()
    assert "contract_indicator" not in p.model_dump()


def test_product_item_nests_seller_details():
    p = ProductItem(seller_details=[{"seller_id": "S-1", "name": "Acme", "price": 9.5, "moq": 2}])
    seller = p.seller_details[0]
    assert isinstance(seller, SellerDetail)
    assert (seller.seller_id, seller.name, seller.price, seller.moq) == ("S-1", "Acme", 9.5, 2)


def test_seller_detail_defaults():
    s = SellerDetail()
    assert s.moq == ""  # "" when the seller has no MOQ
    assert s.price is None and s.lead_time is None and s.currency is None
    assert s.temporary_unavailable is None
    assert s.under_agreement is None


def test_seller_detail_availability_flags():
    s = SellerDetail(temporary_unavailable=True, under_agreement=True)
    assert s.temporary_unavailable is True
    assert s.under_agreement is True
    assert SellerDetail(temporary_unavailable=True, under_agreement=False).under_agreement is False


@pytest.mark.parametrize("moq", [5, ""])
def test_seller_detail_moq_accepts_int_or_empty_string(moq):
    assert SellerDetail(moq=moq).moq == moq


def test_seller_detail_rejects_bad_price():
    with pytest.raises(ValidationError):
        SellerDetail(price="cheap")


def test_search_response_defaults_empty_facets():
    r = SearchApiResponse(total_results=0, page=1, page_size=10, products=[])
    assert r.facets == SearchFacets()
    assert r.facets.category == [] and r.facets.seller == []
    assert r.facets.sub_category == [] and r.facets.product_type == []
    assert r.request_id is None


def test_search_response_requires_core_fields():
    with pytest.raises(ValidationError):
        SearchApiResponse(page=1, page_size=10, products=[])


def test_search_response_nests_products_and_facets():
    r = SearchApiResponse(
        request_id="rid",
        total_results=1, page=1, page_size=60,
        products=[{"sku_id": "S1", "sku_name": "Valve", "seller_details": [{"price": 1.0}]}],
        facets={"seller": [{"id": "S-1", "value": "Acme", "count": 3}]},
    )
    assert isinstance(r.products[0], ProductItem)
    assert r.products[0].seller_details[0].price == 1.0
    assert r.facets.seller[0] == FacetTermValue(id="S-1", value="Acme", count=3)


def test_facet_term_value_requires_id_value_and_count():
    with pytest.raises(ValidationError):
        FacetTermValue(value="x")
    with pytest.raises(ValidationError):
        FacetTermValue(value="x", count=1)  # id is required too
    assert FacetTermValue(id="c1", value="Valves", count=2).id == "c1"


def test_suggestions_response_shape():
    r = SuggestionsResponse(
        query="val",
        query_suggestions=[{"text": "Valve"}],
        product_suggestions=[{"text": "Ball Valve", "sku_id": "SKU-1"}, {"text": "Gate Valve"}],
    )
    assert r.query_suggestions[0].text == "Valve"
    assert r.product_suggestions[0].sku_id == "SKU-1"
    assert r.product_suggestions[1].sku_id is None


def test_product_suggestion_has_no_destination_url():
    assert "destination_url" not in ProductSuggestion(text="x").model_dump()


def test_health_response_redis_optional():
    h = HealthResponse(status="ok", opensearch_connected=True, embedding_model_loaded=True, embedding_model_name="m")
    assert h.redis_connected is None
    assert h.opensearch_cluster is None


def test_indices_response():
    r = IndicesResponse(indices=[{"name": "i", "doc_count": 3, "size_bytes": 10, "status": "green"}], total=1)
    assert isinstance(r.indices[0], IndexInfo)
    assert r.indices[0].doc_count == 3


def test_latency_and_error_models():
    lat = LatencyResponse(target="oci", endpoint="e", samples=1, samples_ms=[1.0], min_ms=1, avg_ms=1, max_ms=1, ok=True)
    assert lat.error is None
    assert ErrorResponse(code="X", message="y").model_dump() == {"code": "X", "message": "y"}
