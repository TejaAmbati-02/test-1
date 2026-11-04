"""Builds the OpenSearch queries for /v1/search and /v1/search/suggestions and shapes the results.

Search modes (2.2):
  * text          -> hybrid BM25 multi_match + kNN on the text vector, fused with RRF
  * image         -> plain nested kNN on images.image_vector (each SKU scores by its best image)
  * text + image  -> hybrid BM25 + text kNN + image kNN fused with weighted RRF (image weighted
                     highest). If the cluster has no weighted RRF, the text instead filters
                     (every word must appear) and the image kNN ranks.
  * filter-browse -> no query and no image: just the filters (or match_all)

Language: ``en`` runs BM25 on the English fields; ``de`` on the ``*_de`` fields + ``content``.
"""
from __future__ import annotations

import time
from typing import Any

from config import settings as _cfg
from models.response import (
    FacetTermValue,
    ProductItem,
    ProductSuggestion,
    QuerySuggestion,
    SearchApiResponse,
    SearchFacets,
    SellerDetail,
    SuggestionsResponse,
)
from services.opensearch import OpenSearchClientService, os_client
from utils.logger import logger

PIPELINE_ID = "hybrid-rrf-pipeline"
# Weighted RRF for text + image: sub-query order is [BM25, text kNN, image kNN].
TEXT_IMAGE_PIPELINE_ID = "hybrid-rrf-text-image-pipeline"

# Filter keys (4) -> the index field each one filters on. Text fields use their
# ``.keyword`` sub-field so exact-match term filters work.
FILTER_KEY_MAP: dict[str, str] = {
    # Facet filters: the frontend sends back the facet's ``id`` (see FACETS below).
    "category": "category_ids",
    "sub_category": "sub_category_id",
    "seller": "seller_details.seller_id",
    # Earlier names of the facet filters, still accepted.
    "category_name": "category_ids",
    "sub_category_name": "sub_category_id",
    "sellername": "seller_details.seller_id",
    # Seller by its own field names: matches a product if ANY of its sellers matches
    # (not just the primary one, unlike sold_by).
    "seller_id": "seller_details.seller_id",
    "seller_name": "seller_details.seller_name.keyword",
    # Not a facet, but filterable by a number or range.
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
    "under_agreement": "under_agreement",
    "sku_id": "sku_id",
    "product_key": "product_key",
    "asmo_material_number": "asmo_material_number",
    "aramco_material_number": "aramco_material_number",
}
ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(FILTER_KEY_MAP)
# Filter keys on a nested field: the clause is wrapped in a nested query on this path.
NESTED_FILTER_PATHS: dict[str, str] = {
    "seller": "seller_details", "sellername": "seller_details", "lead_time": "seller_details",
    "seller_id": "seller_details", "seller_name": "seller_details",
}
# Filter keys that match any of several fields: product_type takes the facet id
# (product_type_id) or, as before, the product-type name; sellername takes the
# seller id or the seller name (it used to compare names against the id field).
FILTER_ALT_FIELDS: dict[str, tuple[str, ...]] = {
    "product_type": ("product_type_id", "product_type.keyword"),
    "sellername": ("seller_details.seller_id", "seller_details.seller_name.keyword"),
}

# Facets returned to the frontend, counted over ALL matching products (not just the page).
# Each value is {id, value, count}: ``id`` is what the frontend sends back as the filter
# with the same key, ``value`` is the label.
#   key -> (id field, label field or None to show the id, nested path or None)
FACETS: dict[str, tuple[str, str | None, str | None]] = {
    "category":     ("category_ids", "category_name.keyword", None),
    "sub_category": ("sub_category_id", "sub_category_name.keyword", None),
    "product_type": ("product_type_id", "product_type.keyword", None),
    "seller":       ("seller_details.seller_id", "seller_details.seller_name.keyword", "seller_details"),
}
FACET_SIZE = 50

# The OpenSearch aggregations that compute FACETS; built once from the table above.
FACET_AGGS: dict[str, Any] = {}
for _key, (_id_field, _label_field, _nested_path) in FACETS.items():
    _terms: dict[str, Any] = {"terms": {"field": _id_field, "size": FACET_SIZE}}
    _sub: dict[str, Any] = {}
    if _label_field:
        _sub["label"] = {"terms": {"field": _label_field, "size": 1}}
    if _nested_path:
        _sub["products"] = {"reverse_nested": {}}  # count products, not seller offers
    if _sub:
        _terms["aggs"] = _sub
    FACET_AGGS[_key] = {"nested": {"path": _nested_path}, "aggs": {"values": _terms}} if _nested_path else _terms

# Nested image field written by the ingest pipeline: [{image_url, image_vector}, ...].
IMAGE_PATH = "images"
IMAGE_VECTOR_FIELD = "images.image_vector"

# Text fields searched by the BM25 half of the hybrid query, per request language.
# English uses the base fields; German uses their ``*_de`` copies plus the shared
# ``content`` blob. Fields missing from the index are skipped (``lenient``), so German
# search still works on ``content`` until the index carries the ``*_de`` fields.
_LOCALIZED_FIELDS = ["product_name", "product_description", "category_name", "sub_category_name", "product_type"]
KEYWORD_FIELDS = [*_LOCALIZED_FIELDS, "content"]
KEYWORD_FIELDS_DE = [*(f"{f}_de" for f in _LOCALIZED_FIELDS), "content"]
KEYWORD_FIELDS_BY_LANGUAGE: dict[str, list[str]] = {"en": KEYWORD_FIELDS, "de": KEYWORD_FIELDS_DE}
# Field the autocomplete matches on (and shows), per language.
SUGGEST_FIELD_BY_LANGUAGE: dict[str, str] = {"en": "product_name", "de": "product_name_de"}


class SearchService:
    def __init__(self, os_client: OpenSearchClientService) -> None:
        self._os = os_client
        # Set by setup_pipeline(): False when the cluster rejects weighted RRF,
        # in which case text + image falls back to text-as-filter.
        self._text_image_fusion = False

    async def setup_pipeline(self) -> None:
        """Create the RRF search pipelines (called once at startup)."""
        # 1. Weighted text + image pipeline — always (re)written so weight changes apply.
        #    Optional: if the cluster rejects it, text + image search falls back to text-as-filter.
        w = _cfg.text_image_image_weight
        weights = [(1 - w) / 2, (1 - w) / 2, w]  # BM25, text kNN, image kNN
        try:
            await self._os.create_search_pipeline(TEXT_IMAGE_PIPELINE_ID, rank_constant=60, weights=weights)
            self._text_image_fusion = True
            logger.info(f"Search — '{TEXT_IMAGE_PIPELINE_ID}' ready (weights={weights})")
        except Exception as exc:
            self._text_image_fusion = False
            logger.warning(
                f"Search — weighted RRF not available ({type(exc).__name__}: {exc}); "
                "text + image search falls back to filtering by the text",
                extra={"status": "error", "error_details": str(exc)},
            )

        # 2. Plain RRF pipeline for text search — required, created only if missing.
        if await self._os.pipeline_exists(PIPELINE_ID):
            logger.info(f"Search — pipeline '{PIPELINE_ID}' already exists")
            return
        await self._os.create_search_pipeline(PIPELINE_ID, rank_constant=60)
        logger.info(f"Search — pipeline '{PIPELINE_ID}' created")

    async def search(
        self,
        query: str | None,
        query_vector: list[float] | None,
        image_vector: list[float] | None,
        filters: list[dict],
        sort_by: str | None,
        sort_order: str,
        page: int,
        page_size: int,
        request_id: str | None = None,
        language: str = "en",
    ) -> SearchApiResponse:
        """Run a text / image / text + image / filter-browse search (mode picked from the inputs).

        ``filters`` are ready-made OpenSearch clauses. ``sort_by="price"`` sorts by price;
        anything else keeps relevance order. ``language`` picks the BM25 text fields
        (``de`` -> the ``*_de`` fields + ``content``).
        """
        has_image = image_vector is not None
        filters = list(filters)  # never modify the caller's list
        pipeline_id: str | None = None

        if not query and not has_image:
            mode = "filter-browse"
            query_clause: dict[str, Any] = {"bool": {"filter": filters}} if filters else {"match_all": {}}
        else:
            # Per-query HNSW candidate count (Lucene ignores the index-level ef_search setting).
            knn_params = {"method_parameters": {"ef_search": _cfg.knn_ef_search}} if _cfg.knn_ef_search else {}
            depth = max(_cfg.hybrid_retrieval_k, page * page_size)  # deep enough to reach the requested page

            image_clause: dict[str, Any] = {}
            image_knn: dict[str, Any] = {}
            if has_image:
                image_knn = {"vector": image_vector, "k": depth, **knn_params}
                image_clause = {"nested": {
                    "path": IMAGE_PATH, "score_mode": "max",
                    "query": {"knn": {IMAGE_VECTOR_FIELD: image_knn}},
                }}
                if _cfg.image_min_score > 0:
                    # function_score with no functions keeps the kNN score; min_score drops weak matches.
                    image_clause = {"function_score": {"query": image_clause, "min_score": _cfg.image_min_score}}

            if not query:
                mode = "image"
                # Filters go inside the kNN so it pre-filters instead of post-filtering. One kNN
                # list needs no RRF, so no hybrid query / pipeline.
                if filters:
                    image_knn["filter"] = {"bool": {"filter": filters}}
                query_clause = image_clause
            else:
                fields = KEYWORD_FIELDS_BY_LANGUAGE.get(language, KEYWORD_FIELDS)
                bm25 = {"multi_match": {"query": query, "fields": fields, "type": "best_fields", "lenient": True}}
                text_knn = {"knn": {_cfg.opensearch_vector_field: {
                    "vector": query_vector, "k": _cfg.hybrid_retrieval_k, **knn_params,
                }}}
                if not has_image:
                    mode = "text"
                    subqueries = [bm25, text_knn]
                    pipeline_id = PIPELINE_ID
                elif self._text_image_fusion:
                    mode = "text+image"
                    subqueries = [bm25, text_knn, image_clause]  # order matches the pipeline weights
                    pipeline_id = TEXT_IMAGE_PIPELINE_ID
                else:
                    mode = "text+image (text as filter)"
                    filters.append({"multi_match": {**bm25["multi_match"], "operator": "and"}})  # every word must appear
                    subqueries = [image_clause]
                    pipeline_id = PIPELINE_ID
                # A hybrid query must be the top-level query, so filters go in its own `filter`
                # (applied to every sub-query) rather than an outer bool.
                hybrid: dict[str, Any] = {"queries": subqueries, "pagination_depth": depth}
                if filters:
                    hybrid["filter"] = {"bool": {"filter": filters}}
                query_clause = {"hybrid": hybrid}

        body: dict[str, Any] = {
            "from": (page - 1) * page_size,
            "size": page_size,
            # Never return embedding vectors — they're large and useless to clients.
            "_source": {"excludes": [_cfg.opensearch_vector_field, IMAGE_PATH]},
            "query": query_clause,
            "aggs": FACET_AGGS,
        }
        if sort_by == "price":
            body["sort"] = [{"price": {"order": sort_order}}]

        start = time.perf_counter()
        raw = await self._os.search(_cfg.opensearch_index_name, body, pipeline_id=pipeline_id)
        wall_ms = (time.perf_counter() - start) * 1000
        took_ms = raw.get("took", 0)
        logger.info(
            f"Search — {mode}: {raw['hits']['total']['value']} results, page {page} "
            f"(OpenSearch wall={wall_ms:.1f}ms, server={took_ms}ms, network={wall_ms - took_ms:.1f}ms, pipeline={pipeline_id})",
            extra={"status": "success", "search_query": query, "latency": f"{wall_ms:.1f}ms"},
        )
        return await _to_search_response(raw, page, page_size, request_id)

    async def get_suggestions(
        self, query: str, query_limit: int = 5, product_limit: int = 5, language: str = "en",
    ) -> SuggestionsResponse:
        """Autocomplete: product names (in ``language``) that start with ``query``."""
        name_field = SUGGEST_FIELD_BY_LANGUAGE.get(language, "product_name")
        body = {
            "size": max(product_limit, query_limit),  # enough hits to fill both lists
            "_source": [name_field, "sku_id"],
            "query": {"match_phrase_prefix": {name_field: {"query": query, "max_expansions": 20}}},
        }
        raw = await self._os.search(_cfg.opensearch_index_name, body)

        products: list[ProductSuggestion] = []
        for hit in raw["hits"]["hits"]:
            src = hit.get("_source", {})
            if src.get(name_field):
                products.append(ProductSuggestion(text=src[name_field], sku_id=src.get("sku_id")))
        # Query suggestions: the distinct product names, in order.
        distinct_names = list(dict.fromkeys(p.text for p in products))

        logger.info(f"Search — suggestions for '{query}': {len(products)} products", extra={"status": "success"})
        return SuggestionsResponse(
            query=query,
            query_suggestions=[QuerySuggestion(text=name) for name in distinct_names[:query_limit]],
            product_suggestions=products[:product_limit],
        )


async def _to_search_response(raw: dict[str, Any], page: int, page_size: int, request_id: str | None) -> SearchApiResponse:
    """Map a raw OpenSearch response to the frontend contract (2.4): products + facets."""
    products: list[ProductItem] = []
    for hit in raw["hits"]["hits"]:
        src = hit.get("_source", {})
        sellers: list[SellerDetail] = []
        for i, seller in enumerate(src.get("seller_details") or []):
            # Older indexes keep temporary_unavailable / under_agreement on the root document
            # (the primary, cheapest seller's offer): use those for seller 0 when its offer has none.
            fallback = src if i == 0 else {}
            sellers.append(SellerDetail(
                seller_id=seller.get("seller_id"),
                name=seller.get("seller_name"),
                lead_time=seller.get("lead_time"),
                unit_type=seller.get("unit_type"),
                price=seller.get("price"),
                moq=seller["moq"] if seller.get("moq") is not None else "",  # the FE expects "" when there's no MOQ
                currency=seller.get("currency"),
                temporary_unavailable=(
                    seller["temporary_unavailable"] if seller.get("temporary_unavailable") is not None
                    else fallback.get("temporary_unavailable")
                ),
                under_agreement=(
                    seller["under_agreement"] if seller.get("under_agreement") is not None
                    else fallback.get("under_agreement")
                ),
            ))
        products.append(ProductItem(
            sku_id=src.get("sku_id"),
            sku_name=src.get("sku_name") or src.get("product_name"),  # older indexes only have product_name
            primary_image_url=src.get("primary_image_url"),
            asmo_material_number=src.get("asmo_material_number"),
            aramco_material_number=src.get("aramco_material_number"),
            seller_details=sellers,
        ))

    aggs = raw.get("aggregations", {})
    facets: dict[str, list[FacetTermValue]] = {}
    for key, (_id_field, label_field, nested_path) in FACETS.items():
        agg = aggs.get(key, {})
        buckets = agg.get("values", {}).get("buckets", []) if nested_path else agg.get("buckets", [])
        facets[key] = []
        for bucket in buckets:
            labels = bucket.get("label", {}).get("buckets", [])
            label = labels[0]["key"] if label_field and labels else bucket["key"]
            count = bucket["products"]["doc_count"] if nested_path else bucket["doc_count"]
            facets[key].append(FacetTermValue(id=bucket["key"], value=str(label), count=count))

    return SearchApiResponse(
        request_id=request_id,
        total_results=raw["hits"]["total"]["value"],
        page=page,
        page_size=page_size,
        products=products,
        facets=SearchFacets(**facets),
    )


search_service = SearchService(os_client)
