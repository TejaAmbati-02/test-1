from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.search import (
    ALLOWED_FILTER_KEYS,
    FILTER_KEY_MAP,
    IMAGE_VECTOR_FIELD,
    KEYWORD_FIELDS,
    PIPELINE_ID,
    TEXT_IMAGE_PIPELINE_ID,
    SearchService,
    _build_aggs,
    _parse_facets,
    _to_product,
)
from tests.conftest import os_search_result

AGGS = {
    "category": {"buckets": [{"key": "c1", "doc_count": 4, "label": {"buckets": [{"key": "Valves"}]}}]},
    "product_type": {"buckets": [{"key": "pt1", "doc_count": 3, "label": {"buckets": [{"key": "Valve"}]}}]},
    "seller": {"values": {"buckets": [
        {"key": "s1", "doc_count": 5, "label": {"buckets": [{"key": "Wholesale Warehouse"}]}, "products": {"doc_count": 2}},
    ]}},
}


@pytest.fixture
def os_client():
    client = MagicMock()
    client.search = AsyncMock(return_value=os_search_result(
        [{"_source": {"sku_id": "SKU-1", "product_name": "Ball Valve"}}], total=42, aggs=AGGS,
    ))
    client.pipeline_exists = AsyncMock(return_value=True)
    client.create_search_pipeline = AsyncMock()
    return client


@pytest.fixture
def service(os_client):
    return SearchService(os_client, MagicMock())


def _body(os_client) -> dict:
    return os_client.search.call_args.args[1]


class TestHelpers:
    def test_filter_keys_cover_map(self):
        assert ALLOWED_FILTER_KEYS == frozenset(FILTER_KEY_MAP)
        assert FILTER_KEY_MAP["product_type"] == "product_type.keyword"
        assert FILTER_KEY_MAP["sold_by"] == "sold_by.keyword"

    def test_build_aggs(self):
        aggs = _build_aggs()
        assert set(aggs) == {"category", "sub_category", "product_type", "seller"}
        assert aggs["product_type"] == {
            "terms": {"field": "product_type_id", "size": 50},
            "aggs": {"label": {"terms": {"field": "product_type.keyword", "size": 1}}},
        }
        assert aggs["seller"]["nested"] == {"path": "seller_details"}
        assert aggs["seller"]["aggs"]["values"]["aggs"]["products"] == {"reverse_nested": {}}

    def test_parse_facets(self):
        facets = _parse_facets(AGGS)
        assert facets.category[0].model_dump() == {"id": "c1", "value": "Valves", "count": 4}
        assert facets.product_type[0].model_dump() == {"id": "pt1", "value": "Valve", "count": 3}
        # Seller counts products (reverse_nested), not seller offers.
        assert facets.seller[0].model_dump() == {"id": "s1", "value": "Wholesale Warehouse", "count": 2}
        assert facets.sub_category == []
        assert set(facets.model_dump()) == {"category", "sub_category", "product_type", "seller"}

    def test_parse_facets_empty(self):
        facets = _parse_facets({})
        assert facets.category == facets.sub_category == facets.product_type == facets.seller == []

    def test_new_facet_filter_keys(self):
        assert FILTER_KEY_MAP["category"] == FILTER_KEY_MAP["category_name"] == "category_ids"
        assert FILTER_KEY_MAP["sub_category"] == FILTER_KEY_MAP["sub_category_name"] == "sub_category_id"
        assert FILTER_KEY_MAP["seller"] == FILTER_KEY_MAP["sellername"] == "seller_details.seller_id"

    def test_to_product_maps_all_fields(self):
        product = _to_product({
            "product_key": "pk", "sku_id": "s", "product_name": "n", "product_url": "/p",
            "primary_image_url": "img", "asmo_material_number": "a", "aramco_material_number": "b",
            "product_type": "t", "material_type": "m", "moq": 2, "price": 9.9, "currency": "SAR",
            "sold_by": "Acme", "supplier_count": 3, "contract_indicator": True, "category_ids": ["c1"],
            "seller_details": [{"seller_name": "Acme", "lead_time": 5, "moq": 2, "price": 9.9, "currency": "SAR"}],
            "content_vector": [0.1],  # unknown fields are ignored
        })
        assert product.sku_id == "s"
        assert product.contract_indicator is True
        assert product.category_ids == ["c1"]
        assert product.seller_details[0].lead_time == 5

    @pytest.mark.parametrize("raw", ["", None, False, 0])
    def test_to_product_falsy_contract_indicator(self, raw):
        product = _to_product({"contract_indicator": raw, "seller_details": None, "category_ids": None})
        assert product.contract_indicator is False
        assert product.seller_details == []
        assert product.category_ids == []

    def test_sort_clause(self):
        assert SearchService._sort_clause("price", "asc") == [{"price": {"order": "asc"}}]
        assert SearchService._sort_clause(None, "desc") is None
        assert SearchService._sort_clause("relevance", "desc") is None


def _created_pipelines(os_client) -> dict[str, dict]:
    """Map pipeline id -> kwargs of each create_search_pipeline call."""
    return {c.args[0]: c.kwargs for c in os_client.create_search_pipeline.await_args_list}


class TestSetupPipeline:
    def test_text_image_fusion_off_until_setup(self, service):
        assert service._text_image_fusion is False

    async def test_weighted_text_image_pipeline_is_always_refreshed(self, service, os_client):
        await service.setup_pipeline()
        created = _created_pipelines(os_client)
        assert set(created) == {TEXT_IMAGE_PIPELINE_ID}
        assert created[TEXT_IMAGE_PIPELINE_ID]["rank_constant"] == 60
        # TEXT_IMAGE_IMAGE_WEIGHT=0.6 -> BM25 0.2, text kNN 0.2, image kNN 0.6
        assert created[TEXT_IMAGE_PIPELINE_ID]["weights"] == pytest.approx([0.2, 0.2, 0.6])
        assert service._text_image_fusion is True

    async def test_existing_rrf_pipeline_is_kept(self, service, os_client):
        await service.setup_pipeline()
        os_client.pipeline_exists.assert_awaited_once_with(PIPELINE_ID)
        assert PIPELINE_ID not in _created_pipelines(os_client)

    async def test_missing_rrf_pipeline_is_created(self, service, os_client):
        os_client.pipeline_exists.return_value = False
        await service.setup_pipeline()
        os_client.create_search_pipeline.assert_any_await(PIPELINE_ID, rank_constant=60)
        assert set(_created_pipelines(os_client)) == {TEXT_IMAGE_PIPELINE_ID, PIPELINE_ID}

    async def test_weighted_pipeline_failure_falls_back(self, service, os_client):
        async def create(pipeline_id, **kwargs):
            if pipeline_id == TEXT_IMAGE_PIPELINE_ID:
                raise RuntimeError("weights not supported")
        os_client.pipeline_exists.return_value = False
        os_client.create_search_pipeline.side_effect = create
        service._text_image_fusion = True
        await service.setup_pipeline()  # must not raise
        assert service._text_image_fusion is False
        os_client.create_search_pipeline.assert_any_await(PIPELINE_ID, rank_constant=60)

    async def test_rrf_pipeline_failure_propagates(self, service, os_client):
        os_client.pipeline_exists.return_value = False
        os_client.create_search_pipeline.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.setup_pipeline()


class TestFilterBrowse:
    async def test_with_filters_and_sort(self, service, os_client):
        filters = [{"term": {"currency": "SAR"}}]
        result = await service.filter_browse("idx", filters, "price", "asc", page=3, page_size=20, request_id="r1")
        body = _body(os_client)
        assert os_client.search.call_args.args[0] == "idx"
        assert body["from"] == 40 and body["size"] == 20
        assert body["query"] == {"bool": {"filter": filters}}
        assert body["sort"] == [{"price": {"order": "asc"}}]
        assert body["_source"] == {"excludes": ["content_vector", "images"]}
        assert body["aggs"] == _build_aggs()
        assert result.request_id == "r1"
        assert result.total_results == 42
        assert result.page == 3 and result.page_size == 20
        assert result.products[0].sku_id == "SKU-1"
        assert result.facets.seller[0].value == "Wholesale Warehouse"

    async def test_without_filters_matches_all(self, service, os_client):
        await service.filter_browse("idx", [], None, "desc", page=1, page_size=10)
        body = _body(os_client)
        assert body["query"] == {"match_all": {}}
        assert "sort" not in body

    async def test_failure_propagates(self, service, os_client):
        os_client.search.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.filter_browse("idx", [], None, "desc", 1, 10)


def _search_kwargs(**overrides):
    kwargs = dict(
        index="idx", query=None, query_vector=None, image_vector=None, vector_field="content_vector",
        language="en", filters=[], sort_by=None, sort_order="desc", page=1, page_size=10,
        retrieval_k=100, ef_search=500, request_id="r1",
    )
    kwargs.update(overrides)
    return kwargs


class TestSearchV1:
    async def test_text_search_is_bm25_plus_knn(self, service, os_client):
        result = await service.search_v1(**_search_kwargs(query="valve", query_vector=[0.1, 0.2]))
        hybrid = _body(os_client)["query"]["hybrid"]
        bm25, knn = hybrid["queries"]
        assert bm25["multi_match"]["query"] == "valve"
        assert bm25["multi_match"]["fields"] == KEYWORD_FIELDS
        assert knn == {"knn": {"content_vector": {
            "vector": [0.1, 0.2], "k": 100, "method_parameters": {"ef_search": 500},
        }}}
        assert "filter" not in hybrid
        assert os_client.search.call_args.kwargs["pipeline_id"] == PIPELINE_ID
        assert result.total_results == 42
        assert result.products[0].product_name == "Ball Valve"

    async def test_image_search_is_plain_nested_knn(self, service, os_client):
        await service.search_v1(**_search_kwargs(image_vector=[0.3]))
        query = _body(os_client)["query"]
        assert "hybrid" not in query
        nested = query["nested"]
        assert nested["path"] == "images"
        assert nested["score_mode"] == "max"
        knn = nested["query"]["knn"][IMAGE_VECTOR_FIELD]
        assert knn == {"vector": [0.3], "k": 100, "method_parameters": {"ef_search": 500}}
        # A single kNN list needs no RRF pipeline.
        assert os_client.search.call_args.kwargs["pipeline_id"] is None

    async def test_image_search_prefilters_inside_knn(self, service, os_client):
        filters = [{"term": {"currency": "SAR"}}]
        await service.search_v1(**_search_kwargs(image_vector=[0.3], filters=filters, page=3, page_size=50))
        knn = _body(os_client)["query"]["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]
        assert knn["filter"] == {"bool": {"filter": filters}}
        assert knn["k"] == 150  # max(retrieval_k=100, 3*50)

    async def test_image_min_score_wraps_in_function_score(self, service, os_client):
        await service.search_v1(**_search_kwargs(image_vector=[0.3], image_min_score=0.6))
        fs = _body(os_client)["query"]["function_score"]
        assert fs["min_score"] == 0.6
        assert fs["query"]["nested"]["path"] == "images"

    async def test_text_plus_image_falls_back_to_text_filter(self, service, os_client):
        # Weighted RRF unavailable (the default before setup_pipeline succeeds).
        service._text_image_fusion = False
        user_filters = [{"term": {"currency": "SAR"}}]
        await service.search_v1(**_search_kwargs(
            query="valve", query_vector=[0.1], image_vector=[0.3], filters=user_filters,
        ))
        hybrid = _body(os_client)["query"]["hybrid"]
        assert len(hybrid["queries"]) == 1 and "nested" in hybrid["queries"][0]
        applied = hybrid["filter"]["bool"]["filter"]
        assert applied[0] == {"term": {"currency": "SAR"}}
        assert applied[1]["multi_match"]["operator"] == "and"
        assert applied[1]["multi_match"]["query"] == "valve"
        assert os_client.search.call_args.kwargs["pipeline_id"] == PIPELINE_ID
        assert user_filters == [{"term": {"currency": "SAR"}}]  # caller's list not mutated

    async def test_text_plus_image_weighted_fusion(self, service, os_client):
        service._text_image_fusion = True
        await service.search_v1(**_search_kwargs(
            query="valve", query_vector=[0.1], image_vector=[0.3], image_min_score=0.6,
        ))
        hybrid = _body(os_client)["query"]["hybrid"]
        bm25, text_knn, image = hybrid["queries"]  # order matches the pipeline weights
        assert bm25["multi_match"]["query"] == "valve"
        assert "operator" not in bm25["multi_match"]
        assert text_knn["knn"]["content_vector"]["vector"] == [0.1]
        assert image["function_score"]["min_score"] == 0.6
        assert image["function_score"]["query"]["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]["vector"] == [0.3]
        assert "filter" not in hybrid  # the text ranks, it does not filter
        assert os_client.search.call_args.kwargs["pipeline_id"] == TEXT_IMAGE_PIPELINE_ID

    async def test_filters_sort_and_pagination(self, service, os_client):
        await service.search_v1(**_search_kwargs(
            query="valve", query_vector=[0.1], filters=[{"term": {"moq": 1}}],
            sort_by="price", sort_order="asc", page=5, page_size=30,
        ))
        body = _body(os_client)
        assert body["from"] == 120 and body["size"] == 30
        assert body["sort"] == [{"price": {"order": "asc"}}]
        assert body["query"]["hybrid"]["pagination_depth"] == 150  # max(100, 5*30)
        assert body["query"]["hybrid"]["filter"] == {"bool": {"filter": [{"term": {"moq": 1}}]}}

    async def test_no_ef_search_omits_method_parameters(self, service, os_client):
        await service.search_v1(**_search_kwargs(query="v", query_vector=[0.1], ef_search=None))
        knn = _body(os_client)["query"]["hybrid"]["queries"][1]["knn"]["content_vector"]
        assert "method_parameters" not in knn

    async def test_failure_propagates(self, service, os_client):
        os_client.search.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.search_v1(**_search_kwargs(query="v", query_vector=[0.1]))


class TestSuggestions:
    async def test_builds_query_and_product_suggestions(self, service, os_client):
        os_client.search.return_value = os_search_result([
            {"_source": {"product_name": "Ball Valve", "product_url": "/p/1"}},
            {"_source": {"product_name": "Ball Valve", "product_url": "/p/2"}},
            {"_source": {"product_name": "", "product_url": "/p/3"}},
            {"_source": {"product_name": "Gate Valve & Co"}},
        ])
        result = await service.get_suggestions("idx", "val", "en", query_limit=5, product_limit=2)

        body = _body(os_client)
        assert body["size"] == 5
        assert body["query"]["match_phrase_prefix"]["product_name"]["query"] == "val"
        assert result.query == "val"
        assert [q.text for q in result.query_suggestions] == ["Ball Valve", "Gate Valve & Co"]
        assert result.query_suggestions[1].destination_url == "/search?q=Gate%20Valve%20%26%20Co"
        assert [(p.text, p.destination_url) for p in result.product_suggestions] == [
            ("Ball Valve", "/p/1"), ("Ball Valve", "/p/2"),
        ]

    async def test_query_limit_and_missing_url(self, service, os_client):
        os_client.search.return_value = os_search_result([
            {"_source": {"product_name": "A"}}, {"_source": {"product_name": "B"}},
        ])
        result = await service.get_suggestions("idx", "x", "en", query_limit=1, product_limit=5)
        assert [q.text for q in result.query_suggestions] == ["A"]
        assert result.product_suggestions[0].destination_url == ""

    async def test_failure_propagates(self, service, os_client):
        os_client.search.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.get_suggestions("idx", "x", "en")
