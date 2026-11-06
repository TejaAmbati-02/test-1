"""Builds the OpenSearch queries for /v1/search and /v1/search/suggestions and shapes the results.

Search modes (2.2):
  * text          -> hybrid BM25 multi_match + kNN on the text vector, fused with RRF
  * image         -> plain nested kNN on images.image_vector (each SKU scores by its best image)
  * text + image  -> hybrid BM25 + text kNN + image kNN fused with weighted RRF. A product is
                     returned if it contains the query words OR its image matches (>= image_min_score).
                     The text kNN only ranks keyword matches, so keyword matches outrank image-only
                     look-alikes. If the cluster has no weighted RRF: the same 3 lists with plain RRF.
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
# "field^N" = weight N (higher = a match in that field ranks higher; no ^ means 1).
# English uses the base fields; German uses their ``*_de`` copies. Fields missing from the
# index are skipped (``lenient``), so German search still works on ``content`` until the
# index carries the ``*_de`` fields.
KEYWORD_FIELDS = [
    "aramco_material_number^8",
    "asmo_material_number^8",
    "sku_name^7",
    "product_name^7",  # older indexes name the product here instead of sku_name
    "product_description^3",
    "category_name^1",
    "sub_category_name^1",
    "product_type^1",
    "content^1",
]
KEYWORD_FIELDS_DE = [
    "aramco_material_number^8",
    "asmo_material_number^8",
    "sku_name_de^7",
    "product_name_de^7",
    "product_description_de^3",
    "category_name_de^1",
    "sub_category_name_de^1",
    "product_type_de^1",
    "content^1",
]
KEYWORD_FIELDS_BY_LANGUAGE: dict[str, list[str]] = {"en": KEYWORD_FIELDS, "de": KEYWORD_FIELDS_DE}
# Seller fields live inside the nested ``seller_details`` objects, so they are searched by a
# separate nested query (a normal multi_match can't reach nested fields).
SELLER_FIELDS = [
    "seller_details.seller_name^5",
    "seller_details.seller_id^5",
]
# Field the autocomplete matches on (and shows), per language.
SUGGEST_FIELD_BY_LANGUAGE: dict[str, str] = {"en": "product_name", "de": "product_name_de"}


class SearchService:
    def __init__(self, os_client: OpenSearchClientService) -> None:
        self._os = os_client
        # Set by setup_pipeline(): False when the cluster rejects weighted RRF,
        # in which case text + image falls back to text-as-filter.
        self._text_image_fusion = False
        logger.info("Search — SearchService created")

    async def setup_pipeline(self) -> None:
        """Create the RRF search pipelines (called once at startup)."""
        # 1. Weighted text + image pipeline — always (re)written so weight changes apply.
        #    Optional: if the cluster rejects it, text + image search falls back to text-as-filter.
        w = _cfg.text_image_image_weight
        weights = [(1 - w) / 2, (1 - w) / 2, w]  # BM25, text kNN, image kNN
        logger.info(f"Search — setup: image weight={w} -> RRF weights [BM25, text kNN, image kNN]={weights}")
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
        logger.info(f"Search — setup: checking whether pipeline '{PIPELINE_ID}' exists")
        if await self._os.pipeline_exists(PIPELINE_ID):
            logger.info(f"Search — pipeline '{PIPELINE_ID}' already exists")
            return
        logger.info(f"Search — setup: pipeline '{PIPELINE_ID}' missing, creating it")
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
        logger.info(
            f"Search — start: query={query!r}, language={language}, has_text_vector={query_vector is not None}, "
            f"has_image_vector={has_image}, filters={len(filters)}, sort_by={sort_by}, sort_order={sort_order}, "
            f"page={page}, page_size={page_size}",
            extra={"search_query": query, "language": language,
                   "input_data": {"filters": filters, "sort_by": sort_by, "sort_order": sort_order,
                                  "page": page, "page_size": page_size, "request_id": request_id}},
        )

        if not query and not has_image:
            mode = "filter-browse"
            query_clause: dict[str, Any] = {"bool": {"filter": filters}} if filters else {"match_all": {}}
            logger.info(f"Search — step: no query and no image -> filter-browse "
                        f"({'bool filter with ' + str(len(filters)) + ' filters' if filters else 'match_all'})")
        else:
            # Per-query HNSW candidate count (Lucene ignores the index-level ef_search setting).
            knn_params = {"method_parameters": {"ef_search": _cfg.knn_ef_search}} if _cfg.knn_ef_search else {}
            depth = max(_cfg.hybrid_retrieval_k, page * page_size)  # deep enough to reach the requested page
            logger.info(f"Search — step: kNN ef_search={_cfg.knn_ef_search or 'index default'}, "
                        f"depth={depth} (max of retrieval_k={_cfg.hybrid_retrieval_k} and page*page_size={page * page_size})")

            image_clause: dict[str, Any] = {}
            image_knn: dict[str, Any] = {}
            if has_image:
                image_knn = {"vector": image_vector, "k": depth, **knn_params}
                image_clause = {"nested": {
                    "path": IMAGE_PATH, "score_mode": "max",
                    "query": {"knn": {IMAGE_VECTOR_FIELD: image_knn}},
                }}
                logger.info(f"Search — step: image kNN built on '{IMAGE_VECTOR_FIELD}' (vector dim={len(image_vector)}, k={depth})")
                if _cfg.image_min_score > 0:
                    # function_score with no functions keeps the kNN score; min_score drops weak matches.
                    image_clause = {"function_score": {"query": image_clause, "min_score": _cfg.image_min_score}}
                    logger.info(f"Search — step: image results below score {_cfg.image_min_score} will be dropped")
                else:
                    logger.info("Search — step: no image min_score (all image kNN hits kept)")

            if not query:
                mode = "image"
                # Filters go inside the kNN so it pre-filters instead of post-filtering. One kNN
                # list needs no RRF, so no hybrid query / pipeline.
                logger.info("Search — step: image only -> plain nested kNN, no RRF pipeline")
                if filters:
                    image_knn["filter"] = {"bool": {"filter": filters}}
                    logger.info(f"Search — step: {len(filters)} filters applied inside the image kNN (pre-filter)")
                query_clause = image_clause
            else:
                # BM25: match the weighted product fields OR any seller's name / id.
                fields = KEYWORD_FIELDS_BY_LANGUAGE.get(language, KEYWORD_FIELDS)
                product_match = {"multi_match": {"query": query, "fields": fields, "type": "best_fields", "lenient": True}}
                seller_match = {"multi_match": {"query": query, "fields": SELLER_FIELDS, "type": "best_fields", "lenient": True}}
                if _cfg.text_min_words_match:
                    # e.g. "2<75%": queries of 1-2 words need every word; longer ones 75% of the words.
                    product_match["multi_match"]["minimum_should_match"] = _cfg.text_min_words_match
                    seller_match["multi_match"]["minimum_should_match"] = _cfg.text_min_words_match
                    logger.info(f"Search — step: a product must match {_cfg.text_min_words_match} of the query words")
                bm25 = {"bool": {"should": [
                    product_match,
                    {"nested": {"path": "seller_details", "score_mode": "max", "ignore_unmapped": True, "query": seller_match}},
                ]}}
                logger.info(f"Search — step: BM25 on language={language} fields={fields} "
                            f"+ nested seller fields={SELLER_FIELDS}")
                text_knn = {"knn": {_cfg.opensearch_vector_field: {
                    "vector": query_vector, "k": _cfg.hybrid_retrieval_k, **knn_params,
                }}}
                logger.info(f"Search — step: text kNN on '{_cfg.opensearch_vector_field}' "
                            f"(vector dim={len(query_vector or [])}, k={_cfg.hybrid_retrieval_k})")
                # Which products may be returned at all. Without a gate the text vector search would
                # add its 100 nearest products, related or not (e.g. fans for "pump"). The vectors
                # then only rank what passes the gate.
                #   text         -> the product contains the query words
                #   text + image -> the product contains the query words OR its image matches the
                #                   uploaded one (score >= image_min_score), so a photo still finds
                #                   look-alikes that the text doesn't describe
                gated = False
                if _cfg.text_require_keyword_match:
                    if has_image:
                        gate: dict[str, Any] = {"bool": {"should": [bm25, image_clause], "minimum_should_match": 1}}
                        # The text kNN may only rank keyword matches: otherwise it also votes for the
                        # image-only look-alikes (e.g. a stapler for "battery" + a scissors photo), which
                        # then get two votes (image + text kNN) and outrank the real keyword matches.
                        text_knn["knn"][_cfg.opensearch_vector_field]["filter"] = bm25
                        logger.info("Search — step: results limited to keyword matches OR image matches; "
                                    "text kNN ranks keyword matches only")
                    else:
                        gate = bm25
                        # Does any product (within the filters) contain the query words? A cheap
                        # size-0 search that stops at the first match.
                        check = await self._os.search(_cfg.opensearch_index_name, {
                            "size": 0, "track_total_hits": 1, "terminate_after": 1,
                            "query": {"bool": {"filter": [bm25, *filters]}},
                        })
                        if check["hits"]["total"]["value"] == 0:
                            logger.info(f"Search — no product contains the query words {query!r} -> returning 0 results",
                                        extra={"search_query": query, "status": "no_results"})
                            return SearchApiResponse(request_id=request_id, total_results=0, page=page,
                                                     page_size=page_size, products=[], facets=SearchFacets())
                        logger.info("Search — step: products contain the query words -> results limited to "
                                    "keyword matches; the vector search only re-ranks them")
                    filters.append(gate)
                    gated = True

                if not has_image:
                    mode = "text (keyword matches only)" if gated else "text"
                    subqueries = [bm25, text_knn]
                    pipeline_id = PIPELINE_ID
                    logger.info(f"Search — step: text only -> hybrid [BM25, text kNN] fused by '{pipeline_id}'")
                elif self._text_image_fusion:
                    mode = "text+image (keyword or image matches)" if gated else "text+image"
                    subqueries = [bm25, text_knn, image_clause]  # order matches the pipeline weights
                    pipeline_id = TEXT_IMAGE_PIPELINE_ID
                    logger.info(f"Search — step: text + image -> hybrid [BM25, text kNN, image kNN] "
                                f"fused by weighted '{pipeline_id}'")
                else:
                    mode = "text+image (keyword or image matches, plain RRF)" if gated else "text+image (plain RRF)"
                    # No weighted RRF: plain RRF over the same 3 lists. The text kNN only ranks keyword
                    # matches (see the gate above), so a keyword match gets 2 votes, an image-only
                    # look-alike 1, and a product matching both 3. With only [BM25, image] the top
                    # keyword match and the top image match would tie (e.g. a stapler for "battery").
                    subqueries = [bm25, text_knn, image_clause]
                    pipeline_id = PIPELINE_ID
                    logger.info("Search — step: text + image, weighted RRF unavailable -> hybrid "
                                f"[BM25, text kNN, image kNN] fused by plain '{pipeline_id}'")
                # A hybrid query must be the top-level query, so filters go in its own `filter`
                # (applied to every sub-query) rather than an outer bool.
                hybrid: dict[str, Any] = {"queries": subqueries, "pagination_depth": depth}
                if filters:
                    hybrid["filter"] = {"bool": {"filter": filters}}
                    logger.info(f"Search — step: {len(filters)} filters applied to every hybrid sub-query")
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
            logger.info(f"Search — step: sorting by price {sort_order}")
        else:
            logger.info("Search — step: sorting by relevance score")
        logger.info(
            f"Search — mode={mode}, pipeline={pipeline_id}, index={_cfg.opensearch_index_name}, "
            f"from={body['from']}, size={body['size']}",
            extra={"search_query": query, "search_mode": mode, "pipeline": pipeline_id},
        )

        start = time.perf_counter()
        raw = await self._os.search(_cfg.opensearch_index_name, body, pipeline_id=pipeline_id)
        wall_ms = (time.perf_counter() - start) * 1000
        took_ms = raw.get("took", 0)
        hits = raw["hits"]["hits"]
        total = raw["hits"]["total"]["value"]
        logger.info(
            f"Search — {mode}: {total} total results, {len(hits)} returned on page {page}, "
            f"max_score={raw['hits'].get('max_score')} "
            f"(OpenSearch wall={wall_ms:.1f}ms, server={took_ms}ms, network={wall_ms - took_ms:.1f}ms, pipeline={pipeline_id})",
            extra={"status": "success", "search_query": query, "search_mode": mode, "latency": f"{wall_ms:.1f}ms",
                   "response_payload": {"total_results": total, "returned": len(hits),
                                        "max_score": raw["hits"].get("max_score"), "took_ms": took_ms}},
        )
        # One line per result: its rank, score and identifiers. In text / text+image modes the
        # score is the RRF fusion score (rank-based), not the raw BM25 score.
        for rank, hit in enumerate(hits, start=(page - 1) * page_size + 1):
            src = hit.get("_source", {})
            sellers = [sd.get("seller_name") for sd in src.get("seller_details") or []]
            logger.info(
                f"Search — result #{rank}: score={hit.get('_score')}, id={hit.get('_id')}, "
                f"sku_id={src.get('sku_id')}, name={src.get('sku_name') or src.get('product_name')!r}, "
                f"aramco_material_number={src.get('aramco_material_number')}, "
                f"asmo_material_number={src.get('asmo_material_number')}, sellers={sellers}",
                extra={"search_query": query, "search_mode": mode,
                       "result": {"rank": rank, "score": hit.get("_score"), "id": hit.get("_id"),
                                  "sku_id": src.get("sku_id"),
                                  "name": src.get("sku_name") or src.get("product_name"),
                                  "aramco_material_number": src.get("aramco_material_number"),
                                  "asmo_material_number": src.get("asmo_material_number"),
                                  "sellers": sellers, "price": src.get("price")}},
            )
        if not hits:
            logger.info(f"Search — no results for query={query!r} (mode={mode}, filters={len(filters)})",
                        extra={"search_query": query, "search_mode": mode, "status": "no_results"})
        # Facet counts returned with the results.
        aggs = raw.get("aggregations", {})
        facet_counts = {
            key: len(aggs.get(key, {}).get("values", aggs.get(key, {})).get("buckets", []))
            for key in FACETS
        }
        logger.info(f"Search — facets: {facet_counts}", extra={"search_query": query, "facets": facet_counts})
        # Map the OpenSearch response to the frontend contract (2.4): products + facets.
        logger.info(f"Search — mapping {len(raw['hits']['hits'])} hits to the response")
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
            logger.info(f"Search — mapped product sku_id={src.get('sku_id')}: {len(sellers)} sellers, "
                        f"prices={[sd.price for sd in sellers]}, "
                        f"name_from={'sku_name' if src.get('sku_name') else 'product_name'}")

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
            logger.info(f"Search — facet '{key}': {[(f.value, f.count) for f in facets[key]]}")

        response = SearchApiResponse(
            request_id=request_id,
            total_results=raw["hits"]["total"]["value"],
            page=page,
            page_size=page_size,
            products=products,
            facets=SearchFacets(**facets),
        )
        logger.info(f"Search — done: returning {len(response.products)} products (total={response.total_results}, "
                    f"request_id={response.request_id})", extra={"search_query": query, "status": "success"})
        return response

    async def get_suggestions(
        self, query: str, query_limit: int = 5, product_limit: int = 5, language: str = "en",
    ) -> SuggestionsResponse:
        """Autocomplete on what the user is typing. Matches any of:

          * product name (in ``language``) — the typed words in order, the last one may be unfinished
          * Aramco material number / ASMO material number — starts with the typed text (any case)
          * seller name (words, like the product name) / seller id (starts with) — any of the product's sellers

        ``query_suggestions`` are the matched values (a product name, a material number or a
        seller name / id); ``product_suggestions`` are the matching products.
        """
        name_field = SUGGEST_FIELD_BY_LANGUAGE.get(language, "product_name")
        typed = query.strip()
        body = {
            "size": max(product_limit, query_limit) * 3,  # extra hits so duplicates still fill both lists
            "_source": [name_field, "sku_id", "aramco_material_number", "asmo_material_number"],
            # Each clause has a "_name", so OpenSearch tells us which one matched (hit["matched_queries"]).
            "query": {"bool": {"should": [
                {"match_phrase_prefix": {name_field: {"query": typed, "max_expansions": 20, "boost": 7,
                                                      "_name": "product_name"}}},
                {"prefix": {"aramco_material_number": {"value": typed, "case_insensitive": True, "boost": 8,
                                                       "_name": "aramco_material_number"}}},
                {"prefix": {"asmo_material_number": {"value": typed, "case_insensitive": True, "boost": 8,
                                                     "_name": "asmo_material_number"}}},
                # Sellers are nested objects: inner_hits returns the seller that matched.
                {"nested": {
                    "path": "seller_details", "score_mode": "max", "ignore_unmapped": True, "boost": 5, "_name": "seller",
                    "query": {"bool": {"should": [
                        {"match_phrase_prefix": {"seller_details.seller_name": {"query": typed, "max_expansions": 20}}},
                        {"prefix": {"seller_details.seller_id": {"value": typed, "case_insensitive": True}}},
                    ]}},
                    "inner_hits": {"size": 1},
                }},
            ]}},
        }
        logger.info(f"Search — suggestions start: q={query!r}, language={language}, fields=[{name_field}, "
                    f"aramco_material_number, asmo_material_number, seller name, seller id], "
                    f"query_limit={query_limit}, product_limit={product_limit}", extra={"search_query": query})
        raw = await self._os.search(_cfg.opensearch_index_name, body)
        logger.info(f"Search — suggestions: {raw['hits']['total']['value']} total matches, "
                    f"{len(raw['hits']['hits'])} returned (server={raw.get('took', 0)}ms)",
                    extra={"search_query": query})

        query_texts: list[str] = []
        products: list[ProductSuggestion] = []
        for rank, hit in enumerate(raw["hits"]["hits"], start=1):
            src = hit.get("_source", {})
            matched = set(hit.get("matched_queries") or [])
            seller_hits = hit.get("inner_hits", {}).get("seller_details", {}).get("hits", {}).get("hits", [])
            seller = seller_hits[0].get("_source", {}) if seller_hits else {}

            # The value to suggest is whatever matched — product name first, then material numbers, then seller.
            if "product_name" in matched or not matched:
                matched_by, value = name_field, src.get(name_field)
            elif "aramco_material_number" in matched:
                matched_by, value = "aramco_material_number", src.get("aramco_material_number")
            elif "asmo_material_number" in matched:
                matched_by, value = "asmo_material_number", src.get("asmo_material_number")
            elif str(seller.get("seller_id", "")).lower().startswith(typed.lower()):
                matched_by, value = "seller_id", seller.get("seller_id")
            else:
                matched_by, value = "seller_name", seller.get("seller_name")

            logger.info(f"Search — suggestion #{rank}: score={hit.get('_score')}, matched_by={matched_by}, "
                        f"value={value!r}, sku_id={src.get('sku_id')}, name={src.get(name_field)!r}",
                        extra={"search_query": query})
            if value:
                query_texts.append(str(value))
            if src.get(name_field) or value:
                products.append(ProductSuggestion(text=src.get(name_field) or str(value), sku_id=src.get("sku_id")))
            else:
                logger.info(f"Search — suggestion #{rank} skipped: nothing to show")

        # Query suggestions: the distinct matched values, in order.
        distinct_texts = list(dict.fromkeys(query_texts))
        logger.info(f"Search — suggestions for '{query}': {len(distinct_texts)} distinct values, {len(products)} products; "
                    f"returning {min(len(distinct_texts), query_limit)} queries and {min(len(products), product_limit)} "
                    f"products", extra={"status": "success"})
        return SuggestionsResponse(
            query=query,
            query_suggestions=[QuerySuggestion(text=text) for text in distinct_texts[:query_limit]],
            product_suggestions=products[:product_limit],
        )


# The one object the app uses — import it where needed:  from services.search import search_service
search_service = SearchService(os_client)
