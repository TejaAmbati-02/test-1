from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.search import (
    ALLOWED_FILTER_KEYS,
    FACET_AGGS,
    FILTER_KEY_MAP,
    IMAGE_VECTOR_FIELD,
    KEYWORD_FIELDS,
    KEYWORD_FIELDS_DE,
    PIPELINE_ID,
    TEXT_IMAGE_PIPELINE_ID,
    SearchService,
    _to_search_response,
)
from tests.conftest import os_search_result

AGGS = {
    "category": {"buckets": [{"key": "c1", "doc_count": 4, "label": {"buckets": [{"key": "Valves"}]}}]},
    "product_type": {"buckets": [{"key": "pt1", "doc_count": 3, "label": {"buckets": []}}]},
    "seller": {"values": {"buckets": [
        {"key": "s1", "doc_count": 5, "label": {"buckets": [{"key": "Wholesale Warehouse"}]}, "products": {"doc_count": 2}},
    ]}},
}
HIT = {"_source": {
    "sku_id": "SKU-1", "product_name": "Ball Valve", "primary_image_url": "img", "temporary_unavailable": True,
    "seller_details": [
        {"seller_id": "s1", "seller_name": "Acme", "lead_time": 5, "price": 9.9, "moq": None},
        {"seller_id": "s2", "moq": 3, "under_agreement": True},
    ],
}}


@pytest.fixture
def os_client():
    client = MagicMock()
    client.search = AsyncMock(return_value=os_search_result([HIT], total=42, aggs=AGGS))
    client.pipeline_exists = AsyncMock(return_value=True)
    client.create_search_pipeline = AsyncMock()
    return client


@pytest.fixture
def service(os_client):
    return SearchService(os_client)


def _body(os_client) -> dict:
    return os_client.search.call_args.args[1]


def _pipeline(os_client) -> str | None:
    return os_client.search.call_args.kwargs.get("pipeline_id")


async def _search(service, **overrides):
    kwargs = dict(query=None, query_vector=None, image_vector=None, filters=[], sort_by=None,
                  sort_order="desc", page=1, page_size=10, request_id="r1")
    kwargs.update(overrides)
    return await service.search(**kwargs)


class TestTables:
    def test_filter_keys_cover_map(self):
        assert ALLOWED_FILTER_KEYS == frozenset(FILTER_KEY_MAP)
        assert FILTER_KEY_MAP["category"] == FILTER_KEY_MAP["category_name"] == "category_ids"
        assert FILTER_KEY_MAP["seller"] == FILTER_KEY_MAP["sellername"] == "seller_details.seller_id"

    def test_facet_aggs(self):
        assert set(FACET_AGGS) == {"category", "sub_category", "product_type", "seller"}
        assert FACET_AGGS["product_type"] == {
            "terms": {"field": "product_type_id", "size": 50},
            "aggs": {"label": {"terms": {"field": "product_type.keyword", "size": 1}}},
        }
        assert FACET_AGGS["seller"]["nested"] == {"path": "seller_details"}
        assert FACET_AGGS["seller"]["aggs"]["values"]["aggs"]["products"] == {"reverse_nested": {}}


class TestToSearchResponse:
    async def test_products_and_facets(self):
        r = await _to_search_response(os_search_result([HIT], total=42, aggs=AGGS), 2, 10, "r1")
        assert (r.request_id, r.total_results, r.page, r.page_size) == ("r1", 42, 2, 10)
        p = r.products[0]
        assert (p.sku_id, p.sku_name) == ("SKU-1", "Ball Valve")  # sku_name falls back to product_name
        first, second = p.seller_details
        assert (first.name, first.moq, first.temporary_unavailable) == ("Acme", "", True)  # root fallback for seller 0
        assert (second.moq, second.temporary_unavailable, second.under_agreement) == (3, None, True)
        assert r.facets.category[0].model_dump() == {"id": "c1", "value": "Valves", "count": 4}
        assert r.facets.product_type[0].value == "pt1"  # no label -> the id
        # Seller counts products (reverse_nested), not seller offers.
        assert r.facets.seller[0].model_dump() == {"id": "s1", "value": "Wholesale Warehouse", "count": 2}
        assert r.facets.sub_category == []

    async def test_empty(self):
        r = await _to_search_response(os_search_result(), 1, 10, None)
        assert r.products == [] and r.facets.category == []


class TestSetupPipeline:
    def _created(self, os_client) -> dict[str, dict]:
        return {c.args[0]: c.kwargs for c in os_client.create_search_pipeline.await_args_list}

    async def test_weighted_pipeline_is_always_refreshed(self, service, os_client):
        assert service._text_image_fusion is False
        await service.setup_pipeline()
        created = self._created(os_client)
        assert set(created) == {TEXT_IMAGE_PIPELINE_ID}  # the RRF pipeline already exists
        assert created[TEXT_IMAGE_PIPELINE_ID]["weights"] == pytest.approx([0.2, 0.2, 0.6])
        assert service._text_image_fusion is True

    async def test_missing_rrf_pipeline_is_created(self, service, os_client):
        os_client.pipeline_exists.return_value = False
        await service.setup_pipeline()
        os_client.create_search_pipeline.assert_any_await(PIPELINE_ID, rank_constant=60)

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


class TestSearch:
    async def test_filter_browse(self, service, os_client):
        filters = [{"term": {"currency": "SAR"}}]
        result = await _search(service, filters=filters, sort_by="price", sort_order="asc", page=3, page_size=20)
        body = _body(os_client)
        assert os_client.search.call_args.args[0] == "asmo-search-api-index"
        assert (body["from"], body["size"]) == (40, 20)
        assert body["query"] == {"bool": {"filter": filters}}
        assert body["sort"] == [{"price": {"order": "asc"}}]
        assert body["_source"] == {"excludes": ["content_vector", "images"]}
        assert body["aggs"] == FACET_AGGS
        assert _pipeline(os_client) is None
        assert (result.request_id, result.total_results) == ("r1", 42)

    async def test_browse_without_filters_matches_all(self, service, os_client):
        await _search(service, sort_by="relevance")
        assert _body(os_client)["query"] == {"match_all": {}}
        assert "sort" not in _body(os_client)

    async def test_text_is_bm25_plus_knn(self, service, os_client):
        await _search(service, query="valve", query_vector=[0.1, 0.2])
        hybrid = _body(os_client)["query"]["hybrid"]
        bm25, knn = hybrid["queries"]
        main, seller = bm25["bool"]["should"]
        assert main["multi_match"]["query"] == "valve"
        assert main["multi_match"]["fields"] == KEYWORD_FIELDS == [
            "aramco_material_number^8", "asmo_material_number^8", "sku_name^7", "product_name^7",
            "product_description^3", "category_name^1", "sub_category_name^1", "product_type^1", "content^1",
        ]
        assert seller["nested"]["path"] == "seller_details"
        assert seller["nested"]["query"]["multi_match"]["fields"] == [
            "seller_details.seller_name^5", "seller_details.seller_id^5",
        ]
        assert knn == {"knn": {"content_vector": {"vector": [0.1, 0.2], "k": 100, "method_parameters": {"ef_search": 500}}}}
        assert _pipeline(os_client) == PIPELINE_ID

    async def test_text_keeps_only_keyword_matches_when_some_exist(self, service, os_client):
        user_filters = [{"term": {"currency": "SAR"}}]
        await _search(service, query="pump", query_vector=[0.1], filters=user_filters)
        check_body = os_client.search.await_args_list[0].args[1]  # 1st call: the keyword-match check
        bm25 = _body(os_client)["query"]["hybrid"]["queries"][0]
        assert check_body["size"] == 0
        assert check_body["query"] == {"bool": {"filter": [bm25, *user_filters]}}
        # The real search keeps only products matching the words (plus the user's filters).
        assert _body(os_client)["query"]["hybrid"]["filter"] == {"bool": {"filter": [*user_filters, bm25]}}
        assert user_filters == [{"term": {"currency": "SAR"}}]  # caller's list not modified

    async def test_text_falls_back_to_vector_when_no_keyword_match(self, service, os_client):
        os_client.search.side_effect = [os_search_result([], total=0), os_search_result([HIT], total=42, aggs=AGGS)]
        await _search(service, query="pmup", query_vector=[0.1])
        assert "filter" not in _body(os_client)["query"]["hybrid"]  # vector results are not restricted
        assert os_client.search.await_count == 2

    async def test_text_keyword_check_can_be_disabled(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_require_keyword_match", False)
        await _search(service, query="pump", query_vector=[0.1])
        assert os_client.search.await_count == 1  # no check
        assert "filter" not in _body(os_client)["query"]["hybrid"]

    async def test_german_searches_de_fields_and_content(self, service, os_client):
        await _search(service, query="Ventil", query_vector=[0.1], language="de")
        bm25 = _body(os_client)["query"]["hybrid"]["queries"][0]
        assert bm25["bool"]["should"][0]["multi_match"]["fields"] == KEYWORD_FIELDS_DE == [
            "aramco_material_number^8", "asmo_material_number^8", "sku_name_de^7", "product_name_de^7",
            "product_description_de^3", "category_name_de^1", "sub_category_name_de^1",
            "product_type_de^1", "content^1",
        ]

    async def test_image_is_plain_nested_knn_with_min_score(self, service, os_client):
        await _search(service, image_vector=[0.3])
        fs = _body(os_client)["query"]["function_score"]
        assert fs["min_score"] == 0.6
        nested = fs["query"]["nested"]
        assert (nested["path"], nested["score_mode"]) == ("images", "max")
        assert nested["query"]["knn"][IMAGE_VECTOR_FIELD] == {"vector": [0.3], "k": 100, "method_parameters": {"ef_search": 500}}
        assert _pipeline(os_client) is None  # one kNN list needs no RRF

    async def test_image_without_min_score(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "image_min_score", 0.0)
        await _search(service, image_vector=[0.3])
        assert "nested" in _body(os_client)["query"]

    async def test_image_prefilters_inside_knn(self, service, os_client):
        filters = [{"term": {"currency": "SAR"}}]
        await _search(service, image_vector=[0.3], filters=filters, page=3, page_size=50)
        knn = _body(os_client)["query"]["function_score"]["query"]["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]
        assert knn["filter"] == {"bool": {"filter": filters}}
        assert knn["k"] == 150  # max(retrieval_k=100, 3*50)

    async def test_text_plus_image_weighted_fusion(self, service, os_client):
        service._text_image_fusion = True
        await _search(service, query="valve", query_vector=[0.1], image_vector=[0.3])
        hybrid = _body(os_client)["query"]["hybrid"]
        bm25, text_knn, image = hybrid["queries"]  # order matches the pipeline weights
        main, seller = bm25["bool"]["should"]
        assert "operator" not in main["multi_match"]
        assert "operator" not in seller["nested"]["query"]["multi_match"]
        assert text_knn["knn"]["content_vector"]["vector"] == [0.1]
        assert image["function_score"]["query"]["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]["vector"] == [0.3]
        assert "filter" not in hybrid  # the text ranks, it does not filter
        assert _pipeline(os_client) == TEXT_IMAGE_PIPELINE_ID

    async def test_text_plus_image_falls_back_to_text_filter(self, service, os_client):
        user_filters = [{"term": {"currency": "SAR"}}]
        await _search(service, query="valve", query_vector=[0.1], image_vector=[0.3], filters=user_filters)
        hybrid = _body(os_client)["query"]["hybrid"]
        assert len(hybrid["queries"]) == 1 and "function_score" in hybrid["queries"][0]
        applied = hybrid["filter"]["bool"]["filter"]
        assert applied[0] == {"term": {"currency": "SAR"}}
        main, seller = applied[1]["bool"]["should"]
        assert main["multi_match"]["operator"] == "and"
        assert seller["nested"]["query"]["multi_match"]["operator"] == "and"
        assert _pipeline(os_client) == PIPELINE_ID
        assert user_filters == [{"term": {"currency": "SAR"}}]  # caller's list not modified

    async def test_filters_sort_and_pagination(self, service, os_client):
        await _search(service, query="valve", query_vector=[0.1], filters=[{"term": {"moq": 1}}],
                      sort_by="price", sort_order="asc", page=5, page_size=30)
        body = _body(os_client)
        assert (body["from"], body["size"]) == (120, 30)
        assert body["sort"] == [{"price": {"order": "asc"}}]
        assert body["query"]["hybrid"]["pagination_depth"] == 150  # max(100, 5*30)
        applied = body["query"]["hybrid"]["filter"]["bool"]["filter"]
        assert applied[0] == {"term": {"moq": 1}}  # the user's filter (+ the keyword-match filter after it)

    async def test_no_ef_search_omits_method_parameters(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "knn_ef_search", 0)
        await _search(service, query="v", query_vector=[0.1])
        assert "method_parameters" not in _body(os_client)["query"]["hybrid"]["queries"][1]["knn"]["content_vector"]

    async def test_failure_propagates(self, service, os_client):
        os_client.search.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await _search(service, query="v", query_vector=[0.1])


class TestSuggestions:
    async def test_builds_query_and_product_suggestions(self, service, os_client):
        os_client.search.return_value = os_search_result([
            {"_source": {"product_name": "Ball Valve", "sku_id": "1"}},
            {"_source": {"product_name": "Ball Valve", "sku_id": "2"}},
            {"_source": {"product_name": "", "sku_id": "3"}},
            {"_source": {"product_name": "Gate Valve"}},
        ])
        result = await service.get_suggestions("val", query_limit=5, product_limit=2)
        body = _body(os_client)
        assert body["size"] == 5
        assert body["query"]["match_phrase_prefix"]["product_name"]["query"] == "val"
        assert result.query == "val"
        assert [q.text for q in result.query_suggestions] == ["Ball Valve", "Gate Valve"]
        assert [(p.text, p.sku_id) for p in result.product_suggestions] == [("Ball Valve", "1"), ("Ball Valve", "2")]

    async def test_german_suggests_from_product_name_de(self, service, os_client):
        os_client.search.return_value = os_search_result([{"_source": {"product_name_de": "Kugelhahn", "sku_id": "1"}}])
        result = await service.get_suggestions("kug", language="de")
        body = _body(os_client)
        assert body["query"]["match_phrase_prefix"]["product_name_de"]["query"] == "kug"
        assert [q.text for q in result.query_suggestions] == ["Kugelhahn"]

    async def test_query_limit(self, service, os_client):
        os_client.search.return_value = os_search_result([{"_source": {"product_name": "A"}}, {"_source": {"product_name": "B"}}])
        result = await service.get_suggestions("x", query_limit=1, product_limit=5)
        assert [q.text for q in result.query_suggestions] == ["A"]
        assert len(result.product_suggestions) == 2

    async def test_failure_propagates(self, service, os_client):
        os_client.search.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.get_suggestions("x")
