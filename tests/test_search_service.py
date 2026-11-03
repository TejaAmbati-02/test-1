from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.search import (
    ALLOWED_FILTER_KEYS,
    FILTER_ALT_FIELDS,
    FILTER_KEY_MAP,
    IMAGE_VECTOR_FIELD,
    KEYWORD_FIELDS,
    NESTED_FILTER_PATHS,
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

    def test_filter_key_extras(self):
        assert FILTER_KEY_MAP["under_agreement"] == "under_agreement"
        assert FILTER_KEY_MAP["lead_time"] == "seller_details.lead_time"
        assert NESTED_FILTER_PATHS == {
            "seller": "seller_details", "sellername": "seller_details", "lead_time": "seller_details",
        }
        assert set(NESTED_FILTER_PATHS) <= ALLOWED_FILTER_KEYS
        assert FILTER_ALT_FIELDS["product_type"] == ("product_type_id", "product_type.keyword")

    def test_build_aggs_category_and_seller_shape(self):
        aggs = _build_aggs()
        assert aggs["category"] == {
            "terms": {"field": "category_ids", "size": 50},
            "aggs": {"label": {"terms": {"field": "category_name.keyword", "size": 1}}},
        }
        values = aggs["seller"]["aggs"]["values"]
        assert values["terms"] == {"field": "seller_details.seller_id", "size": 50}
        assert values["aggs"]["label"] == {"terms": {"field": "seller_details.seller_name.keyword", "size": 1}}

    def test_parse_facets_label_falls_back_to_key(self):
        facets = _parse_facets({"sub_category": {"buckets": [{"key": "sc1", "doc_count": 7}]}})
        assert facets.sub_category[0].model_dump() == {"id": "sc1", "value": "sc1", "count": 7}

    def test_parse_facets_malformed_bucket_raises(self):
        with pytest.raises(KeyError):
            _parse_facets({"category": {"buckets": [{"doc_count": 1}]}})  # no "key"

    def test_to_product_maps_all_fields(self):
        product = _to_product({
            "product_key": "pk", "sku_id": "s", "sku_name": "Sku Name", "product_name": "n",
            "primary_image_url": "img", "asmo_material_number": "a", "aramco_material_number": "b",
            "seller_details": [{
                "seller_id": "sel1", "seller_name": "Acme", "lead_time": 5, "unit_type": "Ea",
                "moq": 2, "price": 9.9, "currency": "SAR",
                "temporary_unavailable": False, "under_agreement": False,
            }],
            "content_vector": [0.1],  # unknown fields are ignored
        })
        assert product.sku_id == "s"
        assert product.sku_name == "Sku Name"  # sku_name wins over product_name
        assert product.primary_image_url == "img"
        assert product.asmo_material_number == "a"
        assert product.aramco_material_number == "b"
        assert product.seller_details[0].model_dump() == {
            "seller_id": "sel1", "name": "Acme", "lead_time": 5, "unit_type": "Ea", "price": 9.9,
            "moq": 2, "currency": "SAR", "temporary_unavailable": False, "under_agreement": False,
        }

    def test_to_product_sku_name_falls_back_to_product_name(self):
        assert _to_product({"product_name": "Ball Valve"}).sku_name == "Ball Valve"
        assert _to_product({"sku_name": "", "product_name": "Ball Valve"}).sku_name == "Ball Valve"

    @pytest.mark.parametrize("raw", [None, []])
    def test_to_product_without_sellers(self, raw):
        product = _to_product({"seller_details": raw})
        assert product.seller_details == []
        assert product.sku_id is None and product.sku_name is None

    def test_seller_without_moq_returns_empty_string(self):
        product = _to_product({"seller_details": [{"seller_id": "a", "moq": None}, {"seller_id": "b"}]})
        assert [s.moq for s in product.seller_details] == ["", ""]

    def test_root_flags_fall_back_to_first_seller_only(self):
        product = _to_product({
            "temporary_unavailable": True, "under_agreement": True,
            "seller_details": [{"seller_id": "a"}, {"seller_id": "b"}],
        })
        first, second = product.seller_details
        assert first.temporary_unavailable is True and first.under_agreement is True
        assert second.temporary_unavailable is None and second.under_agreement is None

    def test_seller_flags_win_over_root_flags(self):
        product = _to_product({
            "temporary_unavailable": True, "under_agreement": True,
            "seller_details": [{"seller_id": "a", "temporary_unavailable": False, "under_agreement": False}],
        })
        assert product.seller_details[0].temporary_unavailable is False
        assert product.seller_details[0].under_agreement is False

    def test_to_product_bad_seller_raises(self):
        with pytest.raises(AttributeError):
            _to_product({"sku_id": "s", "seller_details": ["not-a-dict"]})

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

    async def test_text_image_weights_follow_setting(self, service, os_client, monkeypatch, test_settings):
        import services.search as search_module

        monkeypatch.setattr(search_module, "_cfg", test_settings.model_copy(update={"text_image_image_weight": 0.8}))
        await service.setup_pipeline()
        assert _created_pipelines(os_client)[TEXT_IMAGE_PIPELINE_ID]["weights"] == pytest.approx([0.1, 0.1, 0.8])

    async def test_text_image_pipeline_is_set_up_first(self, service, os_client):
        calls = []
        os_client.create_search_pipeline.side_effect = lambda pid, **kw: calls.append(("create", pid))
        os_client.pipeline_exists.side_effect = lambda pid: calls.append(("exists", pid)) or False
        await service.setup_pipeline()
        assert calls == [("create", TEXT_IMAGE_PIPELINE_ID), ("exists", PIPELINE_ID), ("create", PIPELINE_ID)]

    async def test_pipeline_exists_failure_propagates(self, service, os_client):
        os_client.pipeline_exists.side_effect = RuntimeError("down")
        with pytest.raises(RuntimeError, match="down"):
            await service.setup_pipeline()
        assert service._text_image_fusion is True  # weighted pipeline was still created

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
        assert body["from"] == 0 and body["size"] == 10

    async def test_uses_no_search_pipeline(self, service, os_client):
        await service.filter_browse("idx", [], None, "desc", page=1, page_size=10)
        assert os_client.search.call_args.kwargs == {}
        assert len(os_client.search.call_args.args) == 2

    async def test_missing_aggregations_and_source(self, service, os_client):
        os_client.search.return_value = os_search_result([{"_id": "1"}], total=1)
        result = await service.filter_browse("idx", [], None, "desc", page=1, page_size=10)
        assert result.facets.model_dump() == {"category": [], "sub_category": [], "product_type": [], "seller": []}
        assert result.products[0].sku_id is None
        assert result.request_id is None

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
        assert bm25["multi_match"]["type"] == "best_fields"
        assert bm25["multi_match"]["lenient"] is True
        assert result.total_results == 42
        assert result.request_id == "r1"
        assert result.page == 1 and result.page_size == 10
        # product_name is exposed as sku_name in the frontend contract.
        assert result.products[0].sku_name == "Ball Valve"
        assert result.facets.category[0].id == "c1"

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

    async def test_body_excludes_vectors_and_has_aggs(self, service, os_client):
        await service.search_v1(**_search_kwargs(query="v", query_vector=[0.1], vector_field="alt_vector"))
        body = _body(os_client)
        assert body["_source"] == {"excludes": ["alt_vector", "images"]}
        assert body["aggs"] == _build_aggs()
        assert "sort" not in body
        assert set(body["query"]["hybrid"]["queries"][1]["knn"]) == {"alt_vector"}

    async def test_text_search_ignores_image_min_score(self, service, os_client):
        await service.search_v1(**_search_kwargs(query="v", query_vector=[0.1], image_min_score=0.6))
        queries = _body(os_client)["query"]["hybrid"]["queries"]
        assert len(queries) == 2
        assert "multi_match" in queries[0] and "knn" in queries[1]

    async def test_text_knn_k_is_retrieval_k_not_page_depth(self, service, os_client):
        await service.search_v1(**_search_kwargs(query="v", query_vector=[0.1], page=4, page_size=50))
        hybrid = _body(os_client)["query"]["hybrid"]
        assert hybrid["pagination_depth"] == 200
        assert hybrid["queries"][1]["knn"]["content_vector"]["k"] == 100

    async def test_image_min_score_with_filters_and_no_ef_search(self, service, os_client):
        filters = [{"term": {"currency": "SAR"}}]
        await service.search_v1(**_search_kwargs(
            image_vector=[0.3], filters=filters, image_min_score=0.6, ef_search=None,
        ))
        fs = _body(os_client)["query"]["function_score"]
        knn = fs["query"]["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]
        assert knn == {"vector": [0.3], "k": 100, "filter": {"bool": {"filter": filters}}}
        assert os_client.search.call_args.kwargs["pipeline_id"] is None

    async def test_image_search_without_filters_has_no_knn_filter(self, service, os_client):
        await service.search_v1(**_search_kwargs(image_vector=[0.3]))
        knn = _body(os_client)["query"]["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]
        assert "filter" not in knn

    async def test_text_plus_image_fallback_without_user_filters(self, service, os_client):
        service._text_image_fusion = False
        await service.search_v1(**_search_kwargs(query="valve", query_vector=[0.1], image_vector=[0.3]))
        hybrid = _body(os_client)["query"]["hybrid"]
        assert hybrid["filter"] == {"bool": {"filter": [{"multi_match": {
            "query": "valve", "fields": KEYWORD_FIELDS, "type": "best_fields", "lenient": True, "operator": "and",
        }}]}}
        # The image kNN is not pre-filtered in the hybrid query; the hybrid filter applies.
        assert "filter" not in hybrid["queries"][0]["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]

    async def test_text_plus_image_weighted_fusion_with_filters_and_depth(self, service, os_client):
        service._text_image_fusion = True
        filters = [{"term": {"currency": "SAR"}}]
        await service.search_v1(**_search_kwargs(
            query="valve", query_vector=[0.1], image_vector=[0.3], filters=filters, page=3, page_size=50,
        ))
        hybrid = _body(os_client)["query"]["hybrid"]
        assert hybrid["filter"] == {"bool": {"filter": filters}}
        assert hybrid["pagination_depth"] == 150
        _bm25, text_knn, image = hybrid["queries"]
        assert text_knn["knn"]["content_vector"]["k"] == 100
        assert image["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]["k"] == 150  # no min_score -> plain nested

    async def test_image_only_ignores_text_image_fusion(self, service, os_client):
        service._text_image_fusion = True
        await service.search_v1(**_search_kwargs(image_vector=[0.3]))
        assert "nested" in _body(os_client)["query"]
        assert os_client.search.call_args.kwargs["pipeline_id"] is None

    async def test_hits_without_source(self, service, os_client):
        os_client.search.return_value = os_search_result([{"_id": "1"}], total=1)
        result = await service.search_v1(**_search_kwargs(query="v", query_vector=[0.1]))
        assert result.total_results == 1
        assert result.products[0].sku_id is None
        assert result.facets.seller == []

    @pytest.mark.parametrize("error", [RuntimeError("boom"), KeyError("hits")])
    async def test_image_search_failure_propagates(self, service, os_client, error):
        os_client.search.side_effect = error
        with pytest.raises(type(error)):
            await service.search_v1(**_search_kwargs(image_vector=[0.3]))

    async def test_malformed_response_raises(self, service, os_client):
        os_client.search.return_value = {"took": 1}
        with pytest.raises(KeyError):
            await service.search_v1(**_search_kwargs(query="v", query_vector=[0.1]))

    async def test_failure_propagates(self, service, os_client):
        os_client.search.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.search_v1(**_search_kwargs(query="v", query_vector=[0.1]))


class TestSuggestions:
    async def test_builds_query_and_product_suggestions(self, service, os_client):
        os_client.search.return_value = os_search_result([
            {"_source": {"product_name": "Ball Valve", "sku_id": "S1"}},
            {"_source": {"product_name": "Ball Valve", "sku_id": "S2"}},
            {"_source": {"product_name": "", "sku_id": "S3"}},
            {"_source": {"product_name": "Gate Valve & Co"}},
        ])
        result = await service.get_suggestions("idx", "val", "en", query_limit=5, product_limit=2)

        body = _body(os_client)
        assert body["size"] == 5
        assert body["_source"] == ["product_name", "sku_id"]
        assert body["query"] == {"match_phrase_prefix": {"product_name": {"query": "val", "max_expansions": 20}}}
        assert os_client.search.call_args.kwargs == {}  # no search pipeline
        assert result.query == "val"
        # Query suggestions are deduplicated and drawn from all hits, not just product_limit.
        assert [q.text for q in result.query_suggestions] == ["Ball Valve", "Gate Valve & Co"]
        assert [(p.text, p.sku_id) for p in result.product_suggestions] == [
            ("Ball Valve", "S1"), ("Ball Valve", "S2"),
        ]

    async def test_query_limit_and_missing_sku(self, service, os_client):
        os_client.search.return_value = os_search_result([
            {"_source": {"product_name": "A"}}, {"_source": {"product_name": "B"}},
        ])
        result = await service.get_suggestions("idx", "x", "en", query_limit=1, product_limit=5)
        assert [q.text for q in result.query_suggestions] == ["A"]
        assert [p.text for p in result.product_suggestions] == ["A", "B"]
        assert result.product_suggestions[0].sku_id is None

    async def test_size_covers_larger_limit(self, service, os_client):
        await service.get_suggestions("idx", "x", "en", query_limit=8, product_limit=3)
        assert _body(os_client)["size"] == 8

    async def test_no_hits(self, service, os_client):
        os_client.search.return_value = os_search_result([])
        result = await service.get_suggestions("idx", "zzz", "en")
        assert result.query == "zzz"
        assert result.query_suggestions == [] and result.product_suggestions == []

    async def test_hits_without_name_are_skipped(self, service, os_client):
        os_client.search.return_value = os_search_result([{"_source": {"sku_id": "S1"}}, {"_id": "2"}])
        result = await service.get_suggestions("idx", "x", "en")
        assert result.query_suggestions == [] and result.product_suggestions == []

    async def test_failure_propagates(self, service, os_client):
        os_client.search.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.get_suggestions("idx", "x", "en")
