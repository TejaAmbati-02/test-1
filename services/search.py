from __future__ import annotations

import asyncio
from typing import Any

from models.response import (
    CategorySuggestion,
    FacetTermValue,
    ProductResult,
    ProductSuggestion,
    RangeFacet,
    SearchApiResponse,
    SearchHit,
    SuggestionsResponse,
    TermsFacet
)
from config import settings as _cfg
from utils.logger import logger
from services.embedder import EmbeddingService, embedding_service
from services.opensearch import OpenSearchClientService, os_client
from services.reranker import RerankerService, reranker_service

PIPELINE_ID = "hybrid-rrf-pipeline"
CURRENCY_FIELD = _cfg.currency_field

FILTER_KEY_MAP: dict[str, str] = {
    "category_ids":"rootCategoryKey",
    "sub_category":"subCategoryKey",
    "material_type":"materialType",
    "customer_material_number":"customerMaterialNumber",
    "unit_of_measure":"unitOfMeasure",
    "availability":"inStockAnyChannel",
    "price":"price",
    "currency":"currencyType",
    "sku":"sku",
    "product_group":"productGroup",
    "manufacturer_number":"manufacturerNumber",
    "product_manufacturer_number":"productManufacturerNumber",
    "erp_material_number":"erpMaterialNumber",
    "chemical_hazard_bulletin":"chemicalHazardBulletin",
    "product_key":"productKey",
    "variant_key":"variantKey",
    "customer_id":"customerId",
    "hazard_health":"hazardHealth.code",
    "hazard_flame":"hazardFlame.code",
    "hazard_reactive":"hazardReactive.code",
    "root_category":"rootCategoryKey",
}
ALLOWED_FILTER_KEYS:frozenset[str] = frozenset(FILTER_KEY_MAP)

FACET_CONFIG:list[tuple[str, str, str]] = [
    ("category_ids","rootCategoryKey","terms"),
    ("price","price","range"),
    ("material_type","materialType","terms"),
    ("customer_material_number","customerMaterialNumber","terms"),
    ("unit_of_measure","unitOfMeasure","terms"),
    ("availability","inStockAnyChannel","terms"),
    ("product_group","productGroup","terms"),
    ("manufacturer_number","manufacturerNumber","terms")
]


def _build_aggs() -> dict[str, Any]:
    aggs:dict[str, Any] = {}
    for api_key, os_field, agg_type in FACET_CONFIG:
        if agg_type == "terms":
            aggs[api_key] = {"terms":{"field": os_field, "size": 50}}
        else:
            # Range facets (price) are bucketed per currency so each currency
            # reports its own min/max instead of mixing them into one range.
            aggs[api_key] = {
                "terms": {"field": CURRENCY_FIELD, "size": 20},
                "aggs": {"stats": {"field": os_field}},
            }
    return aggs


def _parse_facets(aggs_raw: dict[str, Any]) -> list[TermsFacet | RangeFacet]:
    facets: list[TermsFacet | RangeFacet] = []
    for api_key, _, agg_type in FACET_CONFIG:
        bucket = aggs_raw.get(api_key, {})
        if agg_type == "terms":
            values = [
                FacetTermValue(value=str(b["key"]), count=b["doc_count"])
                for b in bucket.get("buckets", [])
            ]
            if values:
                facets.append(TermsFacet(key=api_key, values=values))
        else:
            # One RangeFacet per currency bucket, each carrying its own code.
            for b in bucket.get("buckets", []):
                stats = b.get("stats", {})
                mn, mx = stats.get("min"), stats.get("max")
                if mn is not None and mx is not None:
                    facets.append(RangeFacet(key=api_key, min=mn, max=mx, currency=str(b["key"])))
    return facets


class SearchService:
    def __init__(
        self,
        os_client: OpenSearchClientService,
        embedding_svc: EmbeddingService,
        reranker_svc: RerankerService,
    ) -> None:
        self._os = os_client
        self._emb = embedding_svc
        self._reranker = reranker_svc

    async def setup_pipeline(self) -> None:
        logger.info(f"SearchService.setup_pipeline — checking for existing pipeline '{PIPELINE_ID}'")
        try:
            exists = await self._os.pipeline_exists(PIPELINE_ID)
            if exists:
                logger.info(
                    "SearchService.setup_pipeline — pipeline already exists, skipping creation",
                    extra={"status": "success", "input_data": {"pipeline_id": PIPELINE_ID}},
                )
                return
            logger.info("SearchService.setup_pipeline — pipeline not found, creating now")
            await self._os.create_search_pipeline(PIPELINE_ID, rank_constant=60)
            logger.info(
                "SearchService.setup_pipeline — pipeline created successfully",
                extra={"status": "success", "input_data": {"pipeline_id": PIPELINE_ID}},
            )
        except Exception as exc:
            logger.error(
                "SearchService.setup_pipeline — failed",
                extra={"status": "error", "error_details": str(exc), "input_data": {"pipeline_id": PIPELINE_ID}},
            )
            raise

    @staticmethod
    def _parse_hits(raw: dict[str, Any], index: str) -> tuple[int, int, list[SearchHit]]:
        took_ms = raw.get("took", 0)
        total = raw["hits"]["total"]["value"]
        hits = [
            SearchHit(
                id=h["_id"],
                index=h.get("_index", index),
                score=float(h.get("_score") or 0.0),
                source=h.get("_source", {}),
                highlights=h.get("highlight"),
            )
            for h in raw["hits"]["hits"]
        ]
        return took_ms, total, hits

    @staticmethod
    def _sort_clause(sort_by: str | None, sort_order: str) -> list[dict] | None:
        if sort_by == "price":
            return [{"price": {"order": sort_order}}]
        return None

    async def filter_browse(
        self,
        index: str,
        filters: list[dict],
        sort_by: str | None,
        sort_order: str,
        page: int,
        page_size: int,
    ) -> SearchApiResponse:
        logger.info(
            "SearchService.filter_browse — starting",
            extra={"input_data": {"index": index, "filter_count": len(filters), "page": page, "page_size": page_size, "sort_by": sort_by}},
        )
        try:
            logger.debug("SearchService.filter_browse — building query body")
            body: dict[str, Any] = {
                "from": (page - 1) * page_size,
                "size": page_size,
                "query": {"bool": {"filter": filters}} if filters else {"match_all": {}},
                "aggs": _build_aggs(),
            }
            sort = self._sort_clause(sort_by, sort_order)
            if sort:
                body["sort"] = sort
                logger.debug(f"SearchService.filter_browse — sort applied: {sort}")

            logger.debug("SearchService.filter_browse — executing OpenSearch query")
            raw = await self._os.search(index, body)

            took_ms = raw.get("took", 0)
            total = raw["hits"]["total"]["value"]
            logger.debug(f"SearchService.filter_browse — parsing {len(raw['hits']['hits'])} hits from total={total}")

            results = [
                ProductResult(id=h["_id"], score=float(h.get("_score") or 0.0), source=h.get("_source", {}))
                for h in raw["hits"]["hits"]
            ]
            total_pages = max(1, -(-total // page_size))

            logger.debug("SearchService.filter_browse — parsing facets")
            facets = _parse_facets(raw.get("aggregations", {}))

            logger.info(
                "SearchService.filter_browse — done",
                extra={
                    "status": "success",
                    "response_payload": {"total": total, "returned": len(results), "total_pages": total_pages, "facets": len(facets), "took_ms": took_ms},
                },
            )
            return SearchApiResponse(
                search_type="filter",
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                took_ms=took_ms,
                results=results,
                facets=facets,
            )
        except Exception as exc:
            logger.error(
                "SearchService.filter_browse — failed",
                extra={"status": "error", "error_details": str(exc), "input_data": {"index": index, "page": page}},
            )
            raise

    async def text_search_v1(
        self,
        index: str,
        query: str,
        vector_field: str,
        language: str,
        filters: list[dict],
        sort_by: str | None,
        sort_order: str,
        page: int,
        page_size: int,
        retrieval_k: int,
        fetch_k: int,
        rerank_top_k: int,
        request_id: str | None = None,
        image_ref: str | None = None,
    ) -> SearchApiResponse:
        logger.info(
            "SearchService.text_search_v1 — starting",
            extra={
                "search_query": query,
                "language": language,
                "input_data": {
                    "index": index, "page": page, "page_size": page_size,
                    "retrieval_k": retrieval_k, "fetch_k": fetch_k, "rerank_top_k": rerank_top_k,
                    "filter_count": len(filters), "sort_by": sort_by,
                },
            },
        )
        try:
            logger.debug(f"SearchService.text_search_v1 — encoding query vector for '{query}'")
            query_vector = await self._emb.encode_query(query)
            logger.debug(
                "SearchService.text_search_v1 — query vector encoded",
                extra={"input_data": {"dim": len(query_vector), "preview": [round(v, 5) for v in query_vector[:5]]}},
            )

            lang = language if language in ("en", "de") else "en"
            name_field = f"productName_{lang}"
            desc_field = f"productDescription_{lang}"
            kw_fields = [name_field, desc_field, "content", "rootCategoryName", "subCategoryName"]
            logger.debug(f"SearchService.text_search_v1 — resolved language='{lang}', keyword fields={kw_fields}")

            logger.debug("SearchService.text_search_v1 — building hybrid query (BM25 + kNN)")
            hybrid_query: dict[str, Any] = {
                "hybrid": {
                    "queries": [
                        {"multi_match": {"query": query, "fields": kw_fields, "type": "best_fields", "lenient": True}},
                        {"knn": {vector_field: {"vector": query_vector, "k": retrieval_k}}},
                    ]
                }
            }
            body: dict[str, Any] = {
                "size": fetch_k,
                "query": {"bool": {"must": [hybrid_query], "filter": filters}} if filters else hybrid_query,
                "aggs": _build_aggs(),
            }
            sort = self._sort_clause(sort_by, sort_order)
            if sort:
                body["sort"] = sort
                logger.debug(f"SearchService.text_search_v1 — sort applied: {sort}")

            logger.info(f"SearchService.text_search_v1 — executing hybrid RRF search via pipeline '{PIPELINE_ID}'")
            raw = await self._os.search(index, body, pipeline_id=PIPELINE_ID)

            took_ms, total, candidates = self._parse_hits(raw, index)
            logger.info(
                f"SearchService.text_search_v1 — RRF returned {len(candidates)} candidates from total={total} (took {took_ms}ms)",
            )
            logger.debug(
                "SearchService.text_search_v1 — candidate ids/scores (pre-rerank)",
                extra={"input_data": {"candidate_ids": [c.id for c in candidates[:20]], "top_scores": [round(c.score, 4) for c in candidates[:5]]}},
            )

            logger.debug("SearchService.text_search_v1 — parsing facets from aggregations")
            facets = _parse_facets(raw.get("aggregations", {}))

            logger.info(f"SearchService.text_search_v1 — reranking {len(candidates)} candidates (top_k={rerank_top_k})")
            reranked = await self._reranker.rerank(query, candidates, top_k=rerank_top_k)
            logger.info(f"SearchService.text_search_v1 — reranking produced {len(reranked)} results")

            logger.debug(f"SearchService.text_search_v1 — paginating: page={page} page_size={page_size}")
            start = (page - 1) * page_size
            page_hits = reranked[start: start + page_size]
            results = [ProductResult(id=h.id, score=h.score, source=h.source) for h in page_hits]
            total_pages = max(1, -(-len(reranked) // page_size))

            logger.info(
                "SearchService.text_search_v1 — done",
                extra={
                    "status": "success",
                    "search_query": query,
                    "response_payload": {
                        "total": total, "reranked": len(reranked), "returned": len(results),
                        "page": page, "total_pages": total_pages, "facets": len(facets), "took_ms": took_ms,
                    },
                },
            )
            return SearchApiResponse(
                request_id=request_id,
                image_ref=image_ref,
                search_type="hybrid",
                query=query,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                took_ms=took_ms,
                results=results,
                facets=facets,
            )
        except Exception as exc:
            logger.error(
                "SearchService.text_search_v1 — failed",
                extra={
                    "status": "error",
                    "search_query": query,
                    "error_details": str(exc),
                    "input_data": {"index": index, "page": page, "language": language},
                },
            )
            raise

    async def get_suggestions(
        self,
        index: str,
        query: str,
        language: str,
        product_limit: int = 5,
        category_limit: int = 3,
    ) -> SuggestionsResponse:
        logger.info(
            "SearchService.get_suggestions — starting",
            extra={"search_query": query, "language": language, "input_data": {"product_limit": product_limit, "category_limit": category_limit}},
        )
        try:
            lang = language if language in ("en", "de") else "en"
            name_field = f"productName_{lang}"
            logger.debug(f"SearchService.get_suggestions — using name field '{name_field}'")

            product_body: dict[str, Any] = {
                "size": product_limit,
                "_source": [name_field, "sku"],
                "query": {"match_phrase_prefix": {name_field: {"query": query, "max_expansions": 20}}},
            }
            category_body: dict[str, Any] = {
                "size": 0,
                "query": {
                    "bool": {
                        "should": [
                            {"match_phrase_prefix": {"rootCategoryName": {"query": query, "max_expansions": 10}}},
                            {"match_phrase_prefix": {"subCategoryName": {"query": query, "max_expansions": 10}}},
                        ]
                    }
                },
                "aggs": {
                    "root_cats": {"terms": {"field": "rootCategoryName.keyword", "size": category_limit}},
                    "sub_cats":{"terms": {"field": "subCategoryName.keyword",  "size": category_limit}},
                },
            }

            logger.debug("SearchService.get_suggestions — firing product + category queries in parallel")
            prod_raw, cat_raw = await asyncio.gather(
                self._os.search(index, product_body),
                self._os.search(index, category_body),
            )
            logger.debug("SearchService.get_suggestions — parallel queries complete")

            product_suggestions: list[ProductSuggestion] = []
            for h in prod_raw["hits"]["hits"]:
                src = h.get("_source", {})
                text = src.get(name_field) or ""
                sku = src.get("sku") or ""
                if text and sku:
                    product_suggestions.append(ProductSuggestion(text=text, sku=sku))
            logger.debug(f"SearchService.get_suggestions — parsed {len(product_suggestions)} product suggestions")

            aggs = cat_raw.get("aggregations", {})
            # Order-preserving dedup across both aggregations via dict.fromkeys.
            unique_names = dict.fromkeys(
                b.get("key", "")
                for agg_key in ("root_cats", "sub_cats")
                for b in aggs.get(agg_key, {}).get("buckets", [])
                if b.get("key")
            )
            category_suggestions = [CategorySuggestion(text=name) for name in list(unique_names)[:category_limit]]
            logger.debug(f"SearchService.get_suggestions — parsed {len(category_suggestions)} category suggestions")

            logger.info(
                "SearchService.get_suggestions — done",
                extra={
                    "status": "success",
                    "search_query": query,
                    "response_payload": {"products": len(product_suggestions), "categories": len(category_suggestions)},
                },
            )
            return SuggestionsResponse(
                query=query,
                product_suggestions=product_suggestions,
                category_suggestions=category_suggestions,
            )
        except Exception as exc:
            logger.error(
                "SearchService.get_suggestions — failed",
                extra={"status": "error", "search_query": query, "error_details": str(exc)},
            )
            raise


search_service = SearchService(os_client, embedding_service, reranker_service)
