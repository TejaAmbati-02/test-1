from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.search import (
    FACET_AGGS,
    FILTERS,
    IMAGE_VECTOR_FIELD,
    KEYWORD_FIELDS,
    KEYWORD_FIELDS_DE,
    PIPELINE_ID,
    TEXT_IMAGE_PIPELINE_ID,
    SearchService,
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
    def test_filter_table(self):
        assert FILTERS["category"][0] == FILTERS["category_name"][0] == ("category_ids",)
        assert FILTERS["seller"][0] == ("seller_details.seller_id",)
        assert FILTERS["sellername"][0] == ("seller_details.seller_id", "seller_details.seller_name.keyword")
        assert all(value_type in ("text", "single", "number", "bool") for _, value_type, _, _ in FILTERS.values())

    def test_facet_aggs(self):
        assert set(FACET_AGGS) == {"category", "sub_category", "product_type", "seller"}
        assert FACET_AGGS["product_type"] == {
            "terms": {"field": "product_type_id", "size": 50},
            "aggs": {"label": {"terms": {"field": "product_type.keyword", "size": 1}}},
        }
        assert FACET_AGGS["seller"]["nested"] == {"path": "seller_details"}
        assert FACET_AGGS["seller"]["aggs"]["values"]["aggs"]["products"] == {"reverse_nested": {}}


class TestResponseMapping:
    async def test_products_and_facets(self, service):
        r = await _search(service, page=2, page_size=10, request_id="r1")  # the os_client fixture returns HIT + AGGS
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

    async def test_multi_select_facet_counted_without_its_own_filter(self, service, os_client):
        category = {"term": {"category_ids": "c1"}}
        sub = {"terms": {"sub_category_id": ["s1"]}}
        unfiltered = {**AGGS, "sub_category": {"buckets": [
            {"key": "s1", "doc_count": 2, "label": {"buckets": []}},
            {"key": "s2", "doc_count": 5, "label": {"buckets": []}},
        ]}}

        async def fake_search(index, body, pipeline_id=None):
            has_sub = sub in body["query"]["bool"]["filter"]
            return os_search_result([HIT], total=2 if has_sub else 7, aggs=AGGS if has_sub else unfiltered)

        os_client.search.side_effect = fake_search
        r = await _search(service, filters=[category, sub], multi_select_filters={"sub_category": [sub]})
        assert r.total_results == 2  # the results still use every filter
        assert [(f.id, f.count) for f in r.facets.sub_category] == [("s1", 2), ("s2", 5)]
        assert r.facets.category[0].id == "c1"  # the other facets use every filter
        recount_body = next(c.args[1] for c in os_client.search.call_args_list
                            if sub not in c.args[1]["query"]["bool"]["filter"])
        assert recount_body["query"]["bool"]["filter"] == [category]  # its other filters still apply

    async def test_no_recount_without_multi_select_filters(self, service, os_client):
        await _search(service, filters=[{"term": {"category_ids": "c1"}}])
        assert os_client.search.await_count == 1

    async def test_empty(self, service, os_client):
        os_client.search.return_value = os_search_result()
        r = await _search(service, request_id=None)
        assert r.products == [] and r.facets.category == []


class TestSetupPipeline:
    def _created(self, os_client) -> dict[str, dict]:
        return {c.args[0]: c.kwargs for c in os_client.create_search_pipeline.await_args_list}

    async def test_weighted_pipeline_is_always_refreshed(self, service, os_client):
        assert service._text_image_fusion is False
        await service.setup_pipeline()
        created = self._created(os_client)
        assert set(created) == {TEXT_IMAGE_PIPELINE_ID}  # the RRF pipeline already exists
        assert created[TEXT_IMAGE_PIPELINE_ID]["weights"] == pytest.approx([0.3, 0.3, 0.4])  # text 60%, image 40%
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
        main, seller, fuzzy = bm25["bool"]["should"]
        # Must stay required when OpenSearch merges filters into this bool (else should becomes optional).
        assert bm25["bool"]["minimum_should_match"] == 1
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

    async def test_text_returns_nothing_when_no_product_has_the_words(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_prefix_min_chars", 0)
        monkeypatch.setattr(search_module._cfg, "semantic_fallback_min_score", 0.0)
        os_client.search.return_value = os_search_result([], total=0)
        r = await _search(service, query="pump", query_vector=[0.1])
        assert os_client.search.await_count == 1  # only the keyword check; no vector search
        assert (r.total_results, r.products) == (0, [])
        assert r.facets.category == [] and r.facets.seller == []

    async def test_semantic_fallback_returns_close_products_when_no_keyword_matches(self, service, os_client):
        # No product has the words, even after relaxing -> the closest in meaning, high scores only.
        os_client.search.side_effect = [
            os_search_result([], total=0),                # keyword check: nothing
            os_search_result([], total=0),                # partial-word check: nothing
            os_search_result([HIT], total=1, aggs=AGGS),  # the semantic fallback search
        ]
        user_filters = [{"term": {"currency": "SAR"}}]
        r = await _search(service, query="pump", query_vector=[0.1], filters=user_filters)
        assert os_client.search.await_count == 3
        fs = _body(os_client)["query"]["function_score"]
        assert fs["min_score"] == 0.8
        assert fs["query"]["knn"]["content_vector"] == {
            "vector": [0.1], "k": 100, "method_parameters": {"ef_search": 500},
            "filter": {"bool": {"filter": user_filters}},
        }
        assert _pipeline(os_client) is None
        assert r.total_results == 1 and r.products[0].sku_id == "SKU-1"

    async def test_semantic_fallback_runs_after_the_relaxed_retry_fails(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_prefix_min_chars", 0)
        neighbours = os_search_result([], total=0, aggs={"word_0": {"doc_count": 0}, "word_1": {"doc_count": 4}})
        os_client.search.side_effect = [
            os_search_result([], total=0),                # strict keyword check
            neighbours,                                   # closest products contain "products"
            os_search_result([], total=0),                # relaxed keyword check: still nothing
            os_search_result([HIT], total=1, aggs=AGGS),  # the semantic fallback search
        ]
        r = await _search(service, query="want products", query_vector=[0.1])
        assert os_client.search.await_count == 4
        assert "function_score" in _body(os_client)["query"] and r.total_results == 1

    async def test_semantic_fallback_needs_a_text_vector(self, service, os_client):
        os_client.search.return_value = os_search_result([], total=0)
        r = await _search(service, query="pump", query_vector=None)
        assert os_client.search.await_count == 2 and r.total_results == 0  # keyword + partial-word checks

    async def test_no_semantic_fallback_when_keywords_match(self, service, os_client):
        await _search(service, query="pump", query_vector=[0.1])
        assert "hybrid" in _body(os_client)["query"]

    async def test_partial_word_finds_longer_words_when_nothing_matches_as_typed(self, service, os_client):
        # "elec": no product has the word "elec" -> retried as the start of a longer word (electrical).
        os_client.search.side_effect = [
            os_search_result([], total=0),                # keyword check as typed: nothing
            os_search_result([], total=5),                # partial-word check: found
            os_search_result([HIT], total=5, aggs=AGGS),  # the search
        ]
        r = await _search(service, query="elec", query_vector=[0.1])
        calls = [c.args[1] for c in os_client.search.await_args_list]
        assert "phrase_prefix" not in str(calls[0])  # as typed first: no partial words
        bm25 = calls[1]["query"]["bool"]["filter"][0]
        prefix = bm25["bool"]["should"][3]["bool"]
        assert prefix["minimum_should_match"] == "2<75%"
        exact, typo, partial = prefix["should"][0]["bool"]["should"]
        assert exact["multi_match"]["query"] == typo["multi_match"]["query"] == "elec"
        assert partial == {"multi_match": {"query": "elec", "type": "phrase_prefix", "lenient": True,
                                           "max_expansions": 100,
                                           "fields": [f for f in KEYWORD_FIELDS if "material_number" not in f]}}
        assert _body(os_client)["query"]["hybrid"]["queries"][0] == bm25  # the search uses it too
        assert r.total_results == 5

    async def test_no_partial_words_when_the_words_match_as_typed(self, service, os_client):
        await _search(service, query="elec", query_vector=[0.1])
        assert os_client.search.await_count == 2 and "phrase_prefix" not in str(_body(os_client))

    async def test_identifiers_and_short_words_never_match_as_partial_words(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "semantic_fallback_min_score", 0.0)
        os_client.search.return_value = os_search_result([], total=0)
        await _search(service, query="JL086A", query_vector=[0.1])
        assert os_client.search.await_count == 1  # no word can be partial -> no partial-word check
        os_client.search.reset_mock()
        os_client.search.side_effect = [os_search_result([], total=0), os_search_result([], total=1),
                                        os_search_result([HIT], total=1, aggs=AGGS)]
        await _search(service, query="elec 10mm ab", query_vector=[0.1])
        prefix = os_client.search.await_args_list[1].args[1]["query"]["bool"]["filter"][0]["bool"]["should"][-1]["bool"]
        per_word = {w["bool"]["should"][0]["multi_match"]["query"]: str(w) for w in prefix["should"]}
        assert "phrase_prefix" in per_word["elec"]
        assert "phrase_prefix" not in per_word["10mm"] and "phrase_prefix" not in per_word["ab"]

    async def test_partial_words_can_be_switched_off(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "semantic_fallback_min_score", 0.0)
        monkeypatch.setattr(search_module._cfg, "text_prefix_min_chars", 0)
        os_client.search.return_value = os_search_result([], total=0)
        await _search(service, query="elec", query_vector=[0.1])
        assert os_client.search.await_count == 1

    async def test_relaxed_retry_keeps_partial_words(self, service, os_client):
        # "i want elec": nothing as typed, nothing with partial words (3 of 3 words needed) -> the closest
        # products contain "elec..." (counted as a partial word too) -> retried as "elec", still partial.
        neighbours = os_search_result([], total=0, aggs={
            "word_0": {"doc_count": 0}, "word_1": {"doc_count": 0}, "word_2": {"doc_count": 8},
        })
        os_client.search.side_effect = [
            os_search_result([], total=0), os_search_result([], total=0), neighbours,
            os_search_result([], total=3), os_search_result([HIT], total=3, aggs=AGGS),
        ]
        r = await _search(service, query="i want elec", query_vector=[0.1])
        calls = [c.args[1] for c in os_client.search.await_args_list]
        word_aggs = calls[2]["aggs"]
        assert "phrase_prefix" in str(word_aggs["word_2"]) and "phrase_prefix" in str(word_aggs["word_1"])
        assert "phrase_prefix" not in str(word_aggs["word_0"])  # "i" is too short
        bm25 = _body(os_client)["query"]["hybrid"]["queries"][0]
        assert bm25["bool"]["should"][0]["multi_match"]["query"] == "elec"
        assert "phrase_prefix" in str(bm25["bool"]["should"][3])
        assert r.total_results == 3

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
        assert nested["query"]["knn"][IMAGE_VECTOR_FIELD] == {"vector": [0.3], "k": 1000, "method_parameters": {"ef_search": 500}}
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
        assert knn["k"] == 1000  # SEARCH_DEPTH: the same on every page

    async def test_text_plus_image_weighted_fusion(self, service, os_client):
        service._text_image_fusion = True
        await _search(service, query="valve", query_vector=[0.1], image_vector=[0.3])
        hybrid = _body(os_client)["query"]["hybrid"]
        bm25, text_knn, image = hybrid["queries"]  # order matches the pipeline weights
        main, seller, fuzzy = bm25["bool"]["should"]
        assert "operator" not in main["multi_match"]
        assert "operator" not in seller["nested"]["query"]["multi_match"]
        assert text_knn["knn"]["content_vector"]["vector"] == [0.1]
        # The text kNN only ranks keyword matches, so image-only look-alikes get just the image vote.
        assert text_knn["knn"]["content_vector"]["filter"] == bm25
        assert image["function_score"]["query"]["nested"]["query"]["knn"][IMAGE_VECTOR_FIELD]["vector"] == [0.3]
        # Products containing the words OR matching the image (not only keyword matches).
        assert hybrid["filter"] == {"bool": {"filter": [
            {"bool": {"should": [bm25, image], "minimum_should_match": 1}},
        ]}}
        assert _pipeline(os_client) == TEXT_IMAGE_PIPELINE_ID

    async def test_text_only_knn_is_not_filtered(self, service, os_client):
        await _search(service, query="valve", query_vector=[0.1])
        knn = _body(os_client)["query"]["hybrid"]["queries"][1]
        assert "filter" not in knn["knn"]["content_vector"]

    async def test_text_plus_image_still_searches_when_no_product_has_the_words(self, service, os_client):
        # e.g. a scissors photo + "battery": the image matches must still come back.
        service._text_image_fusion = True
        await _search(service, query="zzqxvw", query_vector=[0.1], image_vector=[0.3])
        assert os_client.search.await_count == 1  # no keyword pre-check: straight to the search
        assert "hybrid" in _body(os_client)["query"]

    async def test_minimum_words_match(self, service, os_client):
        os_client.search.side_effect = [  # keyword check, exact material-number check ("3.3kv"): none, search
            os_search_result([], total=3), os_search_result([], total=0), os_search_result([HIT], total=3, aggs=AGGS),
        ]
        await _search(service, query="busbar system 3.3kv", query_vector=[0.1])
        bm25 = _body(os_client)["query"]["hybrid"]["queries"][0]
        main, seller, fuzzy, _material = bm25["bool"]["should"]
        assert main["multi_match"]["minimum_should_match"] == "2<75%"
        assert seller["nested"]["query"]["multi_match"]["minimum_should_match"] == "2<75%"
        # "3.3kv" is an identifier -> the typo match is per word, with the same minimum.
        assert fuzzy["bool"]["minimum_should_match"] == "2<75%"

    async def test_words_no_close_product_has_are_dropped_when_nothing_matches(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_prefix_min_chars", 0)
        # "i want steel valvs": no product has "i" / "want" -> keep the words the closest products contain.
        neighbours = os_search_result([], total=0, aggs={
            "word_0": {"doc_count": 0}, "word_1": {"doc_count": 0},
            "word_2": {"doc_count": 12}, "word_3": {"doc_count": 9},
        })
        os_client.search.side_effect = [
            os_search_result([], total=0),           # strict keyword check: nothing
            neighbours,                              # which words do the closest products contain?
            os_search_result([], total=1),           # relaxed keyword check: found
            os_search_result([HIT], total=1, aggs=AGGS),  # the search
        ]
        user_filters = [{"term": {"currency": "SAR"}}]
        r = await _search(service, query="I want steel valvs", query_vector=[0.1], filters=user_filters)
        calls = [c.args[1] for c in os_client.search.await_args_list]
        assert calls[0]["query"]["bool"]["filter"][0]["bool"]["should"][0]["multi_match"]["query"] == "I want steel valvs"
        knn = calls[1]["query"]["knn"]["content_vector"]
        assert (calls[1]["size"], knn["vector"], knn["k"]) == (0, [0.1], 20)
        assert knn["filter"] == {"bool": {"filter": user_filters}}
        assert sorted(calls[1]["aggs"]) == ["catalogue", "word_0", "word_1", "word_2", "word_3"]
        assert calls[1]["aggs"]["catalogue"]["global"] == {}
        assert sorted(calls[1]["aggs"]["catalogue"]["aggs"]) == ["word_0", "word_1", "word_2", "word_3"]
        bm25 = calls[3]["query"]["hybrid"]["queries"][0]
        main, seller, fuzzy = bm25["bool"]["should"]
        assert main["multi_match"]["query"] == "steel valvs"
        assert seller["nested"]["query"]["multi_match"]["query"] == "steel valvs"
        assert fuzzy["multi_match"]["query"] == "steel valvs"
        assert calls[2]["query"]["bool"]["filter"][0] == bm25
        assert r.total_results == 1 and len(r.products) == 1

    async def test_words_common_to_the_whole_catalogue_are_not_searched_alone(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_prefix_min_chars", 0)
        monkeypatch.setattr(search_module._cfg, "semantic_fallback_min_score", 0.0)
        # "glass tube for cooking": the closest products (office supplies) only share "for", which 40 of
        # the 100 catalogue products contain -> no retry with "for" (it returned paper clips), 0 results.
        neighbours = os_search_result([], total=0, aggs={
            "word_0": {"doc_count": 0}, "word_1": {"doc_count": 0}, "word_2": {"doc_count": 20}, "word_3": {"doc_count": 0},
            "catalogue": {"doc_count": 100, "word_0": {"doc_count": 2}, "word_1": {"doc_count": 3},
                          "word_2": {"doc_count": 40}, "word_3": {"doc_count": 1}},
        })
        os_client.search.side_effect = [os_search_result([], total=0), neighbours]
        r = await _search(service, query="glass tube for cooking", query_vector=[0.1])
        assert os_client.search.await_count == 2 and r.total_results == 0 and r.products == []

    async def test_rare_words_are_still_kept_when_a_common_one_is_dropped(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_prefix_min_chars", 0)
        neighbours = os_search_result([], total=0, aggs={
            "word_0": {"doc_count": 5}, "word_1": {"doc_count": 20}, "word_2": {"doc_count": 0},
            "catalogue": {"doc_count": 100, "word_0": {"doc_count": 4}, "word_1": {"doc_count": 60},
                          "word_2": {"doc_count": 0}},
        })
        os_client.search.side_effect = [os_search_result([], total=0), neighbours,
                                        os_search_result([], total=1), os_search_result([HIT], total=1, aggs=AGGS)]
        await _search(service, query="valve for cooking", query_vector=[0.1])
        main = _body(os_client)["query"]["hybrid"]["queries"][0]["bool"]["should"][0]
        assert main["multi_match"]["query"] == "valve"

    async def test_no_relaxing_when_the_closest_products_have_none_or_all_of_the_words(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_prefix_min_chars", 0)
        monkeypatch.setattr(search_module._cfg, "semantic_fallback_min_score", 0.0)
        for counts in ({"word_0": {"doc_count": 0}, "word_1": {"doc_count": 0}},
                       {"word_0": {"doc_count": 3}, "word_1": {"doc_count": 3}}):
            os_client.search.reset_mock()
            os_client.search.side_effect = [os_search_result([], total=0), os_search_result([], total=0, aggs=counts)]
            r = await _search(service, query="zzq xvw", query_vector=[0.1])
            assert os_client.search.await_count == 2 and r.total_results == 0

    async def test_no_relaxing_for_one_word_or_when_off(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_prefix_min_chars", 0)
        monkeypatch.setattr(search_module._cfg, "semantic_fallback_min_score", 0.0)
        os_client.search.return_value = os_search_result([], total=0)
        await _search(service, query="zzqxvw", query_vector=[0.1])
        assert os_client.search.await_count == 1
        monkeypatch.setattr(search_module._cfg, "text_relax_neighbors", 0)
        os_client.search.reset_mock()
        await _search(service, query="zzq xvw", query_vector=[0.1])
        assert os_client.search.await_count == 1

    async def test_typo_tolerant_match_on_text_fields_only(self, service, os_client):
        await _search(service, query="steel valvs", query_vector=[0.1])
        fuzzy = _body(os_client)["query"]["hybrid"]["queries"][0]["bool"]["should"][2]["multi_match"]
        assert (fuzzy["fuzziness"], fuzzy["prefix_length"]) == ("AUTO:4,7", 1)
        assert fuzzy["fields"] == [f for f in KEYWORD_FIELDS if "material_number" not in f]

    async def test_material_number_query_ranks_the_exact_match_first(self, service, os_client):
        # 880000629771 used to also find 880000629778 / ...774 / ...779 (typo match on `content`), and
        # RRF fusion ranked them above the exact owner.
        os_client.search.side_effect = [
            os_search_result([], total=39),               # keyword check
            os_search_result([], total=1),                # exact material-number check: found
            os_search_result([HIT], total=1, aggs=AGGS),  # the search
        ]
        await _search(service, query="880000629771", query_vector=[0.1])
        calls = [c.args[1] for c in os_client.search.await_args_list]
        material = {"bool": {"should": [
            {"term": {f: {"value": v, "case_insensitive": True}}}
            for f in ("aramco_material_number", "asmo_material_number") for v in ("880000629771", "a-880000629771")
        ], "minimum_should_match": 1}}
        bm25 = calls[0]["query"]["bool"]["filter"][0]
        main, seller, exact = bm25["bool"]["should"]  # no typo-tolerant clause for an identifier
        assert "fuzziness" not in str(bm25) and exact == material
        assert calls[1]["query"] == {"bool": {"filter": [material]}}
        query = calls[2]["query"]
        assert "hybrid" not in query and _pipeline(os_client) is None
        assert query["bool"]["must"] == [bm25]
        assert query["bool"]["should"] == [{"constant_score": {"filter": material, "boost": 100000}}]

    async def test_number_without_an_exact_owner_stays_hybrid(self, service, os_client):
        os_client.search.side_effect = [
            os_search_result([], total=3), os_search_result([], total=0), os_search_result([HIT], total=3, aggs=AGGS),
        ]
        await _search(service, query="a-880000629771", query_vector=[0.1])
        terms = os_client.search.await_args_list[1].args[1]["query"]["bool"]["filter"][0]["bool"]["should"]
        assert [t["term"]["asmo_material_number"]["value"] for t in terms if "asmo_material_number" in t["term"]] == ["a-880000629771"]
        assert "hybrid" in _body(os_client)["query"] and _pipeline(os_client) == PIPELINE_ID

    async def test_mixed_query_keeps_typos_for_words_but_not_identifiers(self, service, os_client):
        os_client.search.side_effect = [
            os_search_result([], total=3), os_search_result([], total=0), os_search_result([HIT], total=3, aggs=AGGS),
        ]
        await _search(service, query="JL086A powr", query_vector=[0.1])
        fuzzy = _body(os_client)["query"]["hybrid"]["queries"][0]["bool"]["should"][2]["bool"]
        ident, word = fuzzy["should"]
        assert ident["multi_match"]["query"] == "JL086A" and "fuzziness" not in ident["multi_match"]
        assert word["multi_match"]["query"] == "powr" and word["multi_match"]["fuzziness"] == "AUTO:4,7"
        assert fuzzy["minimum_should_match"] == "2<75%"

    async def test_german_typo_match_uses_de_fields(self, service, os_client):
        await _search(service, query="Ventil", query_vector=[0.1], language="de")
        fuzzy = _body(os_client)["query"]["hybrid"]["queries"][0]["bool"]["should"][2]["multi_match"]
        assert fuzzy["fields"] == [f for f in KEYWORD_FIELDS_DE if "material_number" not in f]

    async def test_typo_tolerance_can_be_off(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_fuzziness", "")
        await _search(service, query="steel valvs", query_vector=[0.1])
        should = _body(os_client)["query"]["hybrid"]["queries"][0]["bool"]["should"]
        assert len(should) == 2 and all("fuzziness" not in str(c) for c in should)

    async def test_minimum_words_match_can_be_off(self, service, os_client, monkeypatch):
        from services import search as search_module
        monkeypatch.setattr(search_module._cfg, "text_min_words_match", "")
        await _search(service, query="busbar system", query_vector=[0.1])
        main = _body(os_client)["query"]["hybrid"]["queries"][0]["bool"]["should"][0]
        assert "minimum_should_match" not in main["multi_match"]

    async def test_text_plus_image_without_weighted_rrf_uses_plain_rrf(self, service, os_client):
        user_filters = [{"term": {"currency": "SAR"}}]
        await _search(service, query="valve", query_vector=[0.1], image_vector=[0.3], filters=user_filters)
        hybrid = _body(os_client)["query"]["hybrid"]
        bm25, text_knn, image = hybrid["queries"]  # same 3 lists, plain RRF
        assert "multi_match" in bm25["bool"]["should"][0] and "function_score" in image
        assert text_knn["knn"]["content_vector"]["filter"] == bm25  # text kNN ranks keyword matches only
        assert "operator" not in bm25["bool"]["should"][0]["multi_match"]
        assert hybrid["filter"]["bool"]["filter"] == [
            {"term": {"currency": "SAR"}},
            {"bool": {"should": [bm25, image], "minimum_should_match": 1}},
        ]
        assert _pipeline(os_client) == PIPELINE_ID
        assert user_filters == [{"term": {"currency": "SAR"}}]  # caller's list not modified

    async def test_filters_sort_and_pagination(self, service, os_client):
        await _search(service, query="valve", query_vector=[0.1], filters=[{"term": {"moq": 1}}],
                      sort_by="price", sort_order="asc", page=5, page_size=30)
        body = _body(os_client)
        assert (body["from"], body["size"]) == (120, 30)
        assert body["sort"] == [{"price": {"order": "asc"}}]
        assert body["query"]["hybrid"]["pagination_depth"] == 1000  # SEARCH_DEPTH: the same on every page
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
        assert body["size"] == 15
        name, aramco, asmo, seller = body["query"]["bool"]["should"]
        assert name["match_phrase_prefix"]["product_name"]["query"] == "val"
        assert aramco["prefix"]["aramco_material_number"] == {
            "value": "val", "case_insensitive": True, "boost": 8, "_name": "aramco_material_number"}
        assert asmo["prefix"]["asmo_material_number"]["value"] == "val"
        assert seller["nested"]["path"] == "seller_details"
        seller_name, seller_id = seller["nested"]["query"]["bool"]["should"]
        assert seller_name["match_phrase_prefix"]["seller_details.seller_name"]["query"] == "val"
        assert seller_id["prefix"]["seller_details.seller_id"] == {"value": "val", "case_insensitive": True}
        assert result.query == "val"
        assert [q.text for q in result.query_suggestions] == ["Ball Valve", "Gate Valve"]
        assert [(p.text, p.sku_id) for p in result.product_suggestions] == [("Ball Valve", "1"), ("Ball Valve", "2")]

    async def test_german_suggests_from_product_name_de(self, service, os_client):
        os_client.search.return_value = os_search_result([{"_source": {"product_name_de": "Kugelhahn", "sku_id": "1"}}])
        result = await service.get_suggestions("kug", language="de")
        body = _body(os_client)
        assert body["query"]["bool"]["should"][0]["match_phrase_prefix"]["product_name_de"]["query"] == "kug"
        assert [q.text for q in result.query_suggestions] == ["Kugelhahn"]

    @pytest.mark.parametrize("matched, expected", [
        (["aramco_material_number"], "JH231A"),
        (["asmo_material_number"], "a-880000629768"),
    ])
    async def test_suggests_the_matched_material_number(self, service, os_client, matched, expected):
        os_client.search.return_value = os_search_result([{
            "matched_queries": matched,
            "_source": {"product_name": "HPE X142 40G QSFP", "sku_id": "S1",
                        "aramco_material_number": "JH231A", "asmo_material_number": "a-880000629768"},
        }])
        result = await service.get_suggestions(expected[:4])
        assert [q.text for q in result.query_suggestions] == [expected]
        assert [(p.text, p.sku_id) for p in result.product_suggestions] == [("HPE X142 40G QSFP", "S1")]

    @pytest.mark.parametrize("typed, expected", [("wholes", "Wholesale Warehouse"), ("C7AF", "c7afe285-2a70")])
    async def test_suggests_the_matched_seller(self, service, os_client, typed, expected):
        os_client.search.return_value = os_search_result([{
            "matched_queries": ["seller"],
            "_source": {"product_name": "Aruba 40G QSFP", "sku_id": "S2"},
            "inner_hits": {"seller_details": {"hits": {"hits": [
                {"_source": {"seller_name": "Wholesale Warehouse", "seller_id": "c7afe285-2a70"}}]}}},
        }])
        result = await service.get_suggestions(typed)
        assert [q.text for q in result.query_suggestions] == [expected]
        assert [p.text for p in result.product_suggestions] == ["Aruba 40G QSFP"]

    async def test_product_name_match_wins(self, service, os_client):
        os_client.search.return_value = os_search_result([{
            "matched_queries": ["product_name", "asmo_material_number"],
            "_source": {"product_name": "Aruba 40G QSFP", "asmo_material_number": "a-1", "sku_id": "S3"},
        }])
        result = await service.get_suggestions("a")
        assert [q.text for q in result.query_suggestions] == ["Aruba 40G QSFP"]

    async def test_query_limit(self, service, os_client):
        os_client.search.return_value = os_search_result([{"_source": {"product_name": "A"}}, {"_source": {"product_name": "B"}}])
        result = await service.get_suggestions("x", query_limit=1, product_limit=5)
        assert [q.text for q in result.query_suggestions] == ["A"]
        assert len(result.product_suggestions) == 2

    async def test_failure_propagates(self, service, os_client):
        os_client.search.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.get_suggestions("x")


class TestPaginationConsistency:
    @pytest.mark.parametrize("query, image", [("valve", None), ("valve", [0.3]), (None, [0.3])])
    async def test_every_page_sends_the_same_search(self, service, os_client, query, image):
        """Only from/size change between pages, so the total and the order are the same on every page."""
        bodies = []
        for page in (1, 2, 3, 10):
            await _search(service, query=query, query_vector=[0.1] if query else None, image_vector=image,
                          page=page, page_size=60)
            body = dict(_body(os_client))
            assert (body.pop("from"), body.pop("size")) == ((page - 1) * 60, 60)
            bodies.append(body)
        assert all(b == bodies[0] for b in bodies)

    async def test_page_after_the_last_result_is_empty_not_an_error(self, service, os_client):
        """OpenSearch answers 400 "Reached end of search result" for a hybrid page past the end:
        the API returns an empty page with the same total and facets as page 1."""
        from models.response import ApiError
        end = ApiError(400, "QUERY_ERROR", "OpenSearch query error: {'reason': 'Reached end of search result, "
                                           "increase pagination_depth value to see more results'}")
        check = os_search_result([HIT], total=1)
        page1 = os_search_result([HIT], total=42, aggs=AGGS)
        os_client.search.side_effect = [check, end, page1]
        response = await _search(service, query="valve", query_vector=[0.1], page=9, page_size=60)
        assert (response.products, response.total_results, response.page) == ([], 42, 9)
        assert response.facets.category[0].value == "Valves"
        assert os_client.search.call_args.args[1]["from"] == 0  # the re-read is the same search from page 1

    async def test_other_query_errors_still_fail(self, service, os_client):
        from models.response import ApiError
        os_client.search.side_effect = [os_search_result([HIT], total=1), ApiError(400, "QUERY_ERROR", "bad query")]
        with pytest.raises(ApiError):
            await _search(service, query="valve", query_vector=[0.1])
