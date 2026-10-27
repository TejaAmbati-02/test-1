from __future__ import annotations

import time
from typing import Any


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
# Weighted RRF for text + image: sub-query order is [BM25, text kNN, image kNN].
TEXT_IMAGE_PIPELINE_ID = "hybrid-rrf-text-image-pipeline"

# Filter keys are the OpenSearch index field names themselves (§4: "Filter keys
# should match the OpenSearch index field names"). The value is the field the
# clause runs against — the same name, or its ``.keyword`` sub-field for text
# fields so exact-match term filters work.
FILTER_KEY_MAP: dict[str, str] = {
    # Facet filters: send back the facet's ``id`` (see FACETS below).
    "category_name": "category_ids",
    "sub_category_name": "sub_category_id",
    "sellername": "seller_details.seller_id",
    "lead_time": "seller_details.lead_time",
    # Other filters.
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
    "temporary_unavailable": "temporary_unavailable",
    "sku_id": "sku_id",
    "product_key": "product_key",
    "asmo_material_number": "asmo_material_number",
    "aramco_material_number": "aramco_material_number",
}
ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(FILTER_KEY_MAP)
# Filter keys on a nested field: the clause is wrapped in a nested query on this path.
NESTED_FILTER_PATHS: dict[str, str] = {"sellername": "seller_details", "lead_time": "seller_details"}
# Filter keys that match any of several fields: product_type takes the facet id
# (product_type_id) or, as before, the product-type name.
FILTER_ALT_FIELDS: dict[str, tuple[str, ...]] = {"product_type": ("product_type_id", "product_type.keyword")}

# Facets returned to the frontend, computed over the full matching population, not
# just the current page (aggregations run across the whole result set, not from/size).
# Each is {id, value, count}: ``id`` is what the frontend sends back as the filter
# with the same key, ``value`` is the label. ``count`` counts products.
#   key -> (id field, label field or None to show the id, nested path or None)
FACETS: dict[str, tuple[str, str | None, str | None]] = {
    "category_name":     ("category_ids", "category_name.keyword", None),
    "sub_category_name": ("sub_category_id", "sub_category_name.keyword", None),
    "product_type":      ("product_type_id", "product_type.keyword", None),
    "sellername":        ("seller_details.seller_id", "seller_details.seller_name.keyword", "seller_details"),
    "moq":               ("moq", None, None),
    "lead_time":         ("seller_details.lead_time", None, "seller_details"),
    "material_type":     ("material_type", None, None),
}
FACET_SIZE = 50
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
    aggs: dict[str, Any] = {}
    for key, (id_field, label_field, nested_path) in FACETS.items():
        terms: dict[str, Any] = {"terms": {"field": id_field, "size": FACET_SIZE}}
        sub: dict[str, Any] = {}
        if label_field:
            sub["label"] = {"terms": {"field": label_field, "size": 1}}
        if nested_path:
            # Count products, not seller offers.
            sub["products"] = {"reverse_nested": {}}
        if sub:
            terms["aggs"] = sub
        aggs[key] = {"nested": {"path": nested_path}, "aggs": {"values": terms}} if nested_path else terms
    aggs["price"] = {"stats": {"field": PRICE_FIELD}}
    return aggs


def _parse_facets(aggs_raw: dict[str, Any]) -> SearchFacets:
    def terms(api_key: str) -> list[FacetTermValue]:
        _id_field, label_field, nested_path = FACETS[api_key]
        raw = aggs_raw.get(api_key, {})
        buckets = raw.get("values", {}).get("buckets", []) if nested_path else raw.get("buckets", [])
        out = []
        for b in buckets:
            labels = b.get("label", {}).get("buckets", [])
            label = labels[0]["key"] if label_field and labels else b["key"]
            count = b["products"]["doc_count"] if nested_path else b["doc_count"]
            out.append(FacetTermValue(id=b["key"], value=str(label), count=count))
        return out

    price_stats = aggs_raw.get("price", {})
    pmin, pmax = price_stats.get("min"), price_stats.get("max")
    # OpenSearch stats returns null min/max when zero documents matched.
    price = PriceFacet(min=pmin, max=pmax) if pmin is not None else None

    return SearchFacets(**{key: terms(key) for key in FACETS}, price=price)


def _seller_detail(seller: dict[str, Any]) -> SellerDetail:
    return SellerDetail(
        seller_id=seller.get("seller_id"),
        name=seller.get("seller_name"),
        lead_time=seller.get("lead_time"),
        unit_type=seller.get("unit_type"),
        price=seller.get("price"),
        moq=seller.get("moq"),
        currency=seller.get("currency"),
        temporary_unavailable=seller.get("temporary_unavailable"),
    )


def _to_product(src: dict[str, Any]) -> ProductItem:
    """Map a raw OpenSearch _source document to the frontend product contract (§2.4).

    Index field names equal the contract keys, so each key is read as-is.
    """
    sellers = src.get("seller_details") or []
    return ProductItem(
        product_key=src.get("product_key"),
        sku_id=src.get("sku_id"),
        # sku_name is written by ingest_json.py; older indexes only carry product_name.
        sku_name=src.get("sku_name") or src.get("product_name"),
        primary_image_url=src.get("primary_image_url"),
        asmo_material_number=src.get("asmo_material_number"),
        aramco_material_number=src.get("aramco_material_number"),
        product_type=src.get("product_type"),
        material_type=src.get("material_type"),
        seller_details=[_seller_detail(s) for s in sellers],
        moq=src.get("moq"),
        price=src.get("price"),
        currency=src.get("currency"),
        sold_by=src.get("sold_by"),
        supplier_count=src.get("supplier_count"),
        # Indexed as "" until a source is agreed; anything falsy means no contract.
        contract_indicator=bool(src.get("contract_indicator")),
        category_ids=src.get("category_ids") or [],
        # Primary seller (seller_details[0]), like price / moq / sold_by. Read from the
        # root field when the index has it, else from the nested offer.
        temporary_unavailable=(
            src["temporary_unavailable"] if "temporary_unavailable" in src
            else sellers[0].get("temporary_unavailable") if sellers else None
        ),
    )


class SearchService:
    def __init__(
        self,
        os_client: OpenSearchClientService,
        embedding_svc: EmbeddingService,
    ) -> None:
        self._os = os_client
        self._emb = embedding_svc
        # Set by setup_pipeline(): False when the cluster rejects weighted RRF,
        # in which case text + image falls back to text-as-filter.
        self._text_image_fusion = False

    async def setup_pipeline(self) -> None:
        await self._setup_text_image_pipeline()
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

    async def _setup_text_image_pipeline(self) -> None:
        """Create/refresh the weighted RRF pipeline used for text + image search."""
        w = _cfg.text_image_image_weight
        weights = [(1 - w) / 2, (1 - w) / 2, w]  # BM25, text kNN, image kNN
        try:
            await self._os.create_search_pipeline(TEXT_IMAGE_PIPELINE_ID, rank_constant=60, weights=weights)
            self._text_image_fusion = True
            logger.info(f"SearchService.setup_pipeline — '{TEXT_IMAGE_PIPELINE_ID}' ready (weights={weights})")
        except Exception as exc:
            self._text_image_fusion = False
            logger.warning(
                f"SearchService.setup_pipeline — weighted RRF not available ({exc}); "
                "text + image search falls back to filtering by the text",
            )

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
        image_min_score: float = 0.0,
        request_id: str | None = None,
    ) -> SearchApiResponse:
        """Text, image or text + image search (§2.2).

          * text  → hybrid BM25 multi_match + kNN on ``content_vector``, fused with RRF;
          * image → a plain nested kNN on ``images.image_vector`` (each SKU scores
            by its best-matching image). RRF over a single list would not change
            the order, so no hybrid query / pipeline is used;
          * text + image → hybrid BM25 + text kNN + image kNN fused with weighted
            RRF (image weighted highest), so the text helps rank but never
            removes an image match. If the cluster has no weighted RRF, the text
            instead filters (every word must appear) and the image kNN ranks.

        ``image_min_score`` drops image matches scoring below it. Scores are
        ``(1 + cosine) / 2``; 0 disables the cut-off.

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
            depth = max(retrieval_k, page * page_size)  # deep enough to cover the requested page
            filters = list(filters)
            bm25 = {"multi_match": {"query": query, "fields": KEYWORD_FIELDS, "type": "best_fields", "lenient": True}} if query else None
            fuse_text_image = bool(query and image and self._text_image_fusion)

            def image_query(knn_filter: list[dict] | None = None) -> dict[str, Any]:
                knn: dict[str, Any] = {"vector": image_vector, "k": depth, **knn_params}
                if knn_filter:
                    knn["filter"] = {"bool": {"filter": knn_filter}}
                q: dict[str, Any] = {"nested": {
                    "path": IMAGE_PATH, "score_mode": "max",
                    "query": {"knn": {IMAGE_VECTOR_FIELD: knn}},
                }}
                # function_score with no functions keeps the kNN score; min_score drops weak matches.
                return {"function_score": {"query": q, "min_score": image_min_score}} if image_min_score > 0 else q

            pipeline_id: str | None = PIPELINE_ID
            if image and not query:
                # Plain nested kNN; filters go inside the kNN so it pre-filters
                # (exact search when the filter is selective) instead of post-filtering.
                query_clause = image_query(filters)
                pipeline_id = None
            else:
                subqueries: list[dict[str, Any]] = []
                if query and image and not fuse_text_image:
                    # Fallback: every query word must appear in one of the text fields.
                    filters.append({"multi_match": {**bm25["multi_match"], "operator": "and"}})
                elif query:
                    # The index holds English text only, so every language searches the same fields.
                    subqueries.append(bm25)
                    subqueries.append({"knn": {vector_field: {"vector": query_vector, "k": retrieval_k, **knn_params}}})
                if image:
                    subqueries.append(image_query())
                if fuse_text_image:
                    pipeline_id = TEXT_IMAGE_PIPELINE_ID
                # A hybrid query must be the top-level query, so filters go in its own
                # `filter` (applied to every sub-query) rather than an outer bool.
                hybrid: dict[str, Any] = {"queries": subqueries, "pagination_depth": depth}
                if filters:
                    hybrid["filter"] = {"bool": {"filter": filters}}
                query_clause = {"hybrid": hybrid}
            logger.debug(f"SearchService.search_v1 — mode={mode} pipeline={pipeline_id} fused_text_image={fuse_text_image}")

            # Paginated natively by OpenSearch (from/size) — no rerank step.
            body: dict[str, Any] = {
                "from": (page - 1) * page_size,
                "size": page_size,
                # Never return embedding vectors — they are large and useless to clients.
                "_source": {"excludes": [vector_field, IMAGE_PATH]},
                "query": query_clause,
                "aggs": _build_aggs(),
            }
            sort = self._sort_clause(sort_by, sort_order)
            if sort:
                body["sort"] = sort
                logger.debug(f"SearchService.search_v1 — sort applied: {sort}")

            logger.info(f"SearchService.search_v1 — executing {mode} search (pipeline={pipeline_id})")
            _t_search0 = time.perf_counter()
            raw = await self._os.search(index, body, pipeline_id=pipeline_id)
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
                "_source": [name_field, "sku_id"],
                "query": {"match_phrase_prefix": {name_field: {"query": query, "max_expansions": 20}}},
            }
            logger.debug("SearchService.get_suggestions — executing product query")
            prod_raw = await self._os.search(index, product_body)

            product_suggestions: list[ProductSuggestion] = []
            for h in prod_raw["hits"]["hits"]:
                src = h.get("_source", {})
                text = src.get(name_field) or ""
                if text:
                    product_suggestions.append(ProductSuggestion(text=text, sku_id=src.get("sku_id")))
            logger.debug(f"SearchService.get_suggestions — parsed {len(product_suggestions)} product suggestions")

            # Query suggestions: distinct product-name phrases (order-preserving dedup).
            unique_queries = dict.fromkeys(
                p.text for p in product_suggestions if p.text
            )
            query_suggestions = [
                QuerySuggestion(text=t)
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
