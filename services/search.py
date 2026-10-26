from __future__ import annotations

import time
from typing import Any

from urllib.parse import quote

from models.response import (
    FacetTermValue,
    PriceFacet,
    ProductItem,
    ProductSuggestion,
    QuerySuggestion,
    SearchApiResponse,
    SearchFacets,
    SellerDetail,
    SuggestionsResponse,
)
from config import settings as _cfg
from utils.logger import logger
from services.embedder import EmbeddingService, embedding_service
from services.opensearch import OpenSearchClientService, os_client

PIPELINE_ID = "hybrid-rrf-pipeline"

# Filter keys are the OpenSearch index field names themselves (§4: "Filter keys
# should match the OpenSearch index field names"). The value is the field the
# clause runs against — the same name, or its ``.keyword`` sub-field for text
# fields so exact-match term filters work.
FILTER_KEY_MAP: dict[str, str] = {
    "category_ids": "category_ids",
    "sub_category_id": "sub_category_id",
    "product_type_id": "product_type_id",
    "product_type": "product_type.keyword",
    "material_type": "material_type",
    "availability": "availability",
    "price": "price",
    "moq": "moq",
    "currency": "currency",
    "sold_by": "sold_by.keyword",
    "supplier_count": "supplier_count",
    "contract_indicator": "contract_indicator",
    "sku_id": "sku_id",
    "product_key": "product_key",
    "asmo_material_number": "asmo_material_number",
    "aramco_material_number": "aramco_material_number",
}
ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(FILTER_KEY_MAP)

# Facets exposed to the frontend (§2.4). The contract defines exactly three:
# two term facets (product_type, material_type) and one price range (min/max).
# They are computed over the full matching population, not just the current page
# (OpenSearch aggregations run across the whole query result set, not from/size).
FACET_TERMS: dict[str, str] = {
    "product_type": "product_type.keyword",
    "material_type": "material_type",
}
PRICE_FIELD = "price"

# Nested image field written by the ingest pipeline: [{image_url, image_vector}, ...].
IMAGE_PATH = "images"
IMAGE_VECTOR_FIELD = "images.image_vector"

# Text fields searched by the BM25 half of the hybrid query.
KEYWORD_FIELDS = [
    "product_name", "product_description", "content",
    "category_name", "sub_category_name", "product_type",
]


def _build_aggs() -> dict[str, Any]:
    aggs: dict[str, Any] = {
        api_key: {"terms": {"field": os_field, "size": 50}}
        for api_key, os_field in FACET_TERMS.items()
    }
    aggs["price"] = {"stats": {"field": PRICE_FIELD}}
    return aggs


def _parse_facets(aggs_raw: dict[str, Any]) -> SearchFacets:
    def terms(api_key: str) -> list[FacetTermValue]:
        return [
            FacetTermValue(value=str(b["key"]), count=b["doc_count"])
            for b in aggs_raw.get(api_key, {}).get("buckets", [])
        ]

    price_stats = aggs_raw.get("price", {})
    pmin, pmax = price_stats.get("min"), price_stats.get("max")
    # OpenSearch stats returns null min/max when zero documents matched.
    price = PriceFacet(min=pmin, max=pmax) if pmin is not None else None

    return SearchFacets(
        product_type=terms("product_type"),
        material_type=terms("material_type"),
        price=price,
    )


def _seller_detail(seller: dict[str, Any]) -> SellerDetail:
    return SellerDetail(
        seller_name=seller.get("seller_name"),
        lead_time=seller.get("lead_time"),
        moq=seller.get("moq"),
        price=seller.get("price"),
        currency=seller.get("currency"),
    )


def _to_product(src: dict[str, Any]) -> ProductItem:
    """Map a raw OpenSearch _source document to the frontend product contract (§2.4).

    Index field names equal the contract keys, so each key is read as-is.
    """
    return ProductItem(
        product_key=src.get("product_key"),
        sku_id=src.get("sku_id"),
        product_name=src.get("product_name"),
        product_url=src.get("product_url"),
        primary_image_url=src.get("primary_image_url"),
        asmo_material_number=src.get("asmo_material_number"),
        aramco_material_number=src.get("aramco_material_number"),
        product_type=src.get("product_type"),
        material_type=src.get("material_type"),
        seller_details=[_seller_detail(s) for s in src.get("seller_details") or []],
        moq=src.get("moq"),
        price=src.get("price"),
        currency=src.get("currency"),
        sold_by=src.get("sold_by"),
        supplier_count=src.get("supplier_count"),
        # Indexed as "" until a source is agreed; anything falsy means no contract.
        contract_indicator=bool(src.get("contract_indicator")),
        category_ids=src.get("category_ids") or [],
    )


class SearchService:
    def __init__(
        self,
        os_client: OpenSearchClientService,
        embedding_svc: EmbeddingService,
    ) -> None:
        self._os = os_client
        self._emb = embedding_svc

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
        request_id: str | None = None,
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
                # Never return the raw embedding vector — it's large and useless to clients.
                "_source": {"excludes": [_cfg.opensearch_vector_field, IMAGE_PATH]},
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

            products = [_to_product(h.get("_source", {})) for h in raw["hits"]["hits"]]

            logger.debug("SearchService.filter_browse — parsing facets")
            facets = _parse_facets(raw.get("aggregations", {}))

            logger.info(
                "SearchService.filter_browse — done",
                extra={
                    "status": "success",
                    "response_payload": {"total_results": total, "returned": len(products), "took_ms": took_ms},
                },
            )
            return SearchApiResponse(
                request_id=request_id,
                total_results=total,
                page=page,
                page_size=page_size,
                products=products,
                facets=facets,
            )
        except Exception as exc:
            logger.error(
                "SearchService.filter_browse — failed",
                extra={"status": "error", "error_details": str(exc), "input_data": {"index": index, "page": page}},
            )
            raise

    async def search_v1(
        self,
        index: str,
        query: str | None,
        query_vector: list[float] | None,
        image_vector: list[float] | None,
        vector_field: str,
        language: str,
        filters: list[dict],
        sort_by: str | None,
        sort_order: str,
        page: int,
        page_size: int,
        retrieval_k: int,
        ef_search: int | None = None,
        request_id: str | None = None,
    ) -> SearchApiResponse:
        """Text, image or text + image search (§2.2), fused with RRF.

        Sub-queries of the hybrid query:
          * text  → BM25 multi_match + kNN on ``content_vector``;
          * image → nested kNN on ``images.image_vector`` (each SKU scores by its
            best-matching image).
          * text + image → the image kNN alone ranks; the text only narrows the
            results to products containing every query word (it is not fused in,
            so it cannot outvote the image).

        Embeddings are resolved by the caller (Redis cache or OCI) and passed in:
        ``query`` + ``query_vector`` for text, ``image_vector`` for an image.
        """
        image = image_vector is not None
        mode = "text+image" if (query and image) else ("image" if image else "text")
        logger.info(
            f"SearchService.search_v1 — starting (mode={mode})",
            extra={
                "search_query": query,
                "language": language,
                "input_data": {
                    "index": index, "mode": mode, "page": page, "page_size": page_size,
                    "retrieval_k": retrieval_k, "filter_count": len(filters), "sort_by": sort_by,
                },
            },
        )
        try:
            logger.debug(
                "SearchService.search_v1 — vectors received",
                extra={"input_data": {"text_dim": len(query_vector or []), "image_dim": len(image_vector or [])}},
            )

            # Per-query HNSW candidate count; Lucene ignores the index-level ef_search.
            knn_params: dict[str, Any] = {"method_parameters": {"ef_search": ef_search}} if ef_search else {}
            subqueries: list[dict[str, Any]] = []
            filters = list(filters)
            if query and image:
                # Every query word must appear in one of the text fields (the
                # catch-all `content` field holds all of them).
                filters.append({"multi_match": {
                    "query": query, "fields": KEYWORD_FIELDS, "type": "best_fields",
                    "operator": "and", "lenient": True,
                }})
            elif query:
                # The index holds English text only, so every language searches the same fields.
                subqueries.append({"multi_match": {"query": query, "fields": KEYWORD_FIELDS, "type": "best_fields", "lenient": True}})
                subqueries.append({"knn": {vector_field: {"vector": query_vector, "k": retrieval_k, **knn_params}}})
            if image:
                subqueries.append({
                    "nested": {
                        "path": IMAGE_PATH,
                        "score_mode": "max",
                        "query": {"knn": {IMAGE_VECTOR_FIELD: {"vector": image_vector, "k": retrieval_k, **knn_params}}},
                    }
                })
            logger.debug(f"SearchService.search_v1 — hybrid query with {len(subqueries)} sub-queries")

            # A hybrid query must be the top-level query, so filters go in its own
            # `filter` (applied to every sub-query) rather than an outer bool.
            hybrid: dict[str, Any] = {
                "queries": subqueries,
                # Each sub-query must retrieve deep enough to cover the requested page.
                "pagination_depth": max(retrieval_k, page * page_size),
            }
            if filters:
                hybrid["filter"] = {"bool": {"filter": filters}}

            # RRF-fused results, paginated natively by OpenSearch (from/size) — no rerank step.
            body: dict[str, Any] = {
                "from": (page - 1) * page_size,
                "size": page_size,
                # Never return embedding vectors — they are large and useless to clients.
                "_source": {"excludes": [vector_field, IMAGE_PATH]},
                "query": {"hybrid": hybrid},
                "aggs": _build_aggs(),
            }
            sort = self._sort_clause(sort_by, sort_order)
            if sort:
                body["sort"] = sort
                logger.debug(f"SearchService.search_v1 — sort applied: {sort}")

            logger.info(f"SearchService.search_v1 — executing hybrid RRF search via pipeline '{PIPELINE_ID}'")
            _t_search0 = time.perf_counter()
            raw = await self._os.search(index, body, pipeline_id=PIPELINE_ID)
            search_wall_ms = (time.perf_counter() - _t_search0) * 1000

            took_ms = raw.get("took", 0)
            total = raw["hits"]["total"]["value"]
            raw_hits = raw["hits"]["hits"]

            # Phase breakdown so slow searches can be attributed precisely:
            #   search_wall_ms = full OpenSearch call wall time (network + server)
            #   took_ms        = OpenSearch server-side time only
            #   os_network_ms  = search_wall_ms - took_ms  (round-trip + (de)serialization)
            os_network_ms = round(search_wall_ms - took_ms, 1)
            logger.info(
                f"SearchService.search_v1 — PHASE BREAKDOWN: "
                f"os_wall={search_wall_ms:.1f}ms os_server(took)={took_ms}ms os_network={os_network_ms}ms",
                extra={
                    "status": "success",
                    "search_query": query,
                    "latency": f"{search_wall_ms:.1f}ms",
                    "response_payload": {
                        "os_wall_ms": round(search_wall_ms, 1),
                        "os_server_took_ms": took_ms,
                        "os_network_ms": os_network_ms,
                    },
                },
            )
            logger.info(
                f"SearchService.search_v1 — RRF returned {len(raw_hits)} hits for page {page} of total={total} (took {took_ms}ms)",
            )

            logger.debug("SearchService.search_v1 — parsing facets from aggregations")
            facets = _parse_facets(raw.get("aggregations", {}))

            products = [_to_product(h.get("_source", {})) for h in raw_hits]

            logger.info(
                "SearchService.search_v1 — done",
                extra={
                    "status": "success",
                    "search_query": query,
                    "response_payload": {
                        "total_results": total, "returned": len(products),
                        "page": page, "took_ms": took_ms,
                    },
                },
            )
            return SearchApiResponse(
                request_id=request_id,
                total_results=total,
                page=page,
                page_size=page_size,
                products=products,
                facets=facets,
            )
        except Exception as exc:
            logger.error(
                "SearchService.search_v1 — failed",
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
        query_limit: int = 5,
        product_limit: int = 5,
    ) -> SuggestionsResponse:
        logger.info(
            "SearchService.get_suggestions — starting",
            extra={"search_query": query, "language": language, "input_data": {"query_limit": query_limit, "product_limit": product_limit}},
        )
        try:
            name_field = "product_name"
            logger.debug(f"SearchService.get_suggestions — using name field '{name_field}'")

            product_body: dict[str, Any] = {
                # Enough hits to fill both the product and the query suggestions.
                "size": max(product_limit, query_limit),
                "_source": [name_field, "product_url"],
                "query": {"match_phrase_prefix": {name_field: {"query": query, "max_expansions": 20}}},
            }
            logger.debug("SearchService.get_suggestions — executing product query")
            prod_raw = await self._os.search(index, product_body)

            product_suggestions: list[ProductSuggestion] = []
            for h in prod_raw["hits"]["hits"]:
                src = h.get("_source", {})
                text = src.get(name_field) or ""
                if text:
                    product_suggestions.append(ProductSuggestion(
                        text=text,
                        destination_url=src.get("product_url") or "",
                    ))
            logger.debug(f"SearchService.get_suggestions — parsed {len(product_suggestions)} product suggestions")

            # Query suggestions: distinct product-name phrases (order-preserving dedup).
            unique_queries = dict.fromkeys(
                p.text for p in product_suggestions if p.text
            )
            query_suggestions = [
                QuerySuggestion(text=t, destination_url=f"/search?q={quote(t)}")
                for t in list(unique_queries)[:query_limit]
            ]
            product_suggestions = product_suggestions[:product_limit]
            logger.debug(f"SearchService.get_suggestions — parsed {len(query_suggestions)} query suggestions")

            logger.info(
                "SearchService.get_suggestions — done",
                extra={
                    "status": "success",
                    "search_query": query,
                    "response_payload": {"queries": len(query_suggestions), "products": len(product_suggestions)},
                },
            )
            return SuggestionsResponse(
                query=query,
                query_suggestions=query_suggestions,
                product_suggestions=product_suggestions,
            )
        except Exception as exc:
            logger.error(
                "SearchService.get_suggestions — failed",
                extra={"status": "error", "search_query": query, "error_details": str(exc)},
            )
            raise


search_service = SearchService(os_client, embedding_service)
