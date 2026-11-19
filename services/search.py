"""Builds the OpenSearch queries for /v1/search and /v1/search/suggestions and shapes the results.

    from services.search import search_service
    response = await search_service.search(query=..., query_vector=..., image_vector=..., filters=[...], ...)

Search modes (picked from what the request sends):
  * filter-browse -> no query and no image: just the filters (or match_all)
  * image         -> nested kNN on images.image_vector (each SKU scores by its best image)
  * text          -> hybrid [BM25, text kNN] fused with RRF; only products containing the query words
  * text + image  -> hybrid [BM25, text kNN, image kNN] fused with weighted RRF; products containing
                     the query words OR whose image matches (>= image_min_score)

Language: ``en`` runs BM25 on the English fields; ``de`` on the ``*_de`` fields + ``content``.

The tables below (filters, facets, text fields) are the only things to edit to add a filter
key, a facet or a searched field.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from config import settings as _cfg
from models.response import (
    ApiError,
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

# RRF search pipelines (created at startup by setup_pipeline).
PIPELINE_ID = "hybrid-rrf-pipeline"                       # plain RRF
TEXT_IMAGE_PIPELINE_ID = "hybrid-rrf-text-image-pipeline"  # weighted: [BM25, text kNN, image kNN]

# ------------------------------------------------------------------------------------------------
# Filters — every key the `filters` JSON may hold (any other key -> 400 INVALID_FILTER_KEY).
#
#   key: (index field(s), value type, nested path, multi-select facet)
#
#   index field(s) — the field the filter matches; with two fields, a value matching either counts.
#                    Text fields use their ``.keyword`` sub-field so the match is exact.
#   value type     — "text"   a non-empty string, or a list of them (match any)
#                    "single" exactly one string (or a one-item list): category is single-select
#                    "number" a number, a list of numbers, or a range {"gt"/"gte"/"lt"/"lte": n}
#                    "bool"   true / false
#   nested path    — the field lives in nested objects (a product's sellers): the clause is wrapped
#                    in a nested query, so it matches if ANY seller matches
#   multi-select   — the facet this filter selects in; that facet is counted WITHOUT this filter
#                    (but with every other one), so the user can still tick more of its values
# ------------------------------------------------------------------------------------------------
FILTERS: dict[str, tuple[tuple[str, ...], str, str | None, str | None]] = {
    # Facet filters: the frontend sends back the facet's ``id`` (see FACETS below).
    "category":               (("category_ids",), "single", None, None),
    "sub_category":           (("sub_category_id",), "text", None, "sub_category"),
    "product_type":           (("product_type_id", "product_type.keyword"), "text", None, "product_type"),  # id or name
    "seller":                 (("seller_details.seller_id",), "text", "seller_details", None),
    # Earlier names of the facet filters, still accepted.
    "category_name":          (("category_ids",), "single", None, None),
    "sub_category_name":      (("sub_category_id",), "text", None, "sub_category"),
    "sellername":             (("seller_details.seller_id", "seller_details.seller_name.keyword"), "text", "seller_details", None),
    # Seller by its own fields (any of the product's sellers, unlike sold_by = the primary one).
    "seller_id":              (("seller_details.seller_id",), "text", "seller_details", None),
    "seller_name":            (("seller_details.seller_name.keyword",), "text", "seller_details", None),
    "lead_time":              (("seller_details.lead_time",), "number", "seller_details", None),
    # Other filters.
    "category_ids":           (("category_ids",), "single", None, None),
    "category_id":            (("category_ids",), "single", None, None),
    "sub_category_id":        (("sub_category_id",), "text", None, "sub_category"),
    "product_type_id":        (("product_type_id",), "text", None, "product_type"),
    "material_type":          (("material_type",), "text", None, None),
    "availability":           (("availability",), "text", None, None),
    "price":                  (("price",), "number", None, None),
    "moq":                    (("moq",), "number", None, None),
    "currency":               (("currency",), "text", None, None),
    "sold_by":                (("sold_by.keyword",), "text", None, None),
    "supplier_count":         (("supplier_count",), "number", None, None),
    "contract_indicator":     (("contract_indicator",), "bool", None, None),
    "temporary_unavailable":  (("temporary_unavailable",), "bool", None, None),
    "under_agreement":        (("under_agreement",), "bool", None, None),
    "sku_id":                 (("sku_id",), "text", None, None),
    "product_key":            (("product_key",), "text", None, None),
    "asmo_material_number":   (("asmo_material_number",), "text", None, None),
    "aramco_material_number": (("aramco_material_number",), "text", None, None),
}

# ------------------------------------------------------------------------------------------------
# Facets returned with every search, counted over ALL matching products (not just the page).
# Each value is {id, value, count}: ``id`` is what the frontend sends back as the filter with the
# same key, ``value`` the label to show.
#   key: (id field, label field or None to show the id, nested path or None)
# ------------------------------------------------------------------------------------------------
FACETS: dict[str, tuple[str, str | None, str | None]] = {
    "category":     ("category_ids", "category_name.keyword", None),
    "sub_category": ("sub_category_id", "sub_category_name.keyword", None),
    "product_type": ("product_type_id", "product_type.keyword", None),
    "seller":       ("seller_details.seller_id", "seller_details.seller_name.keyword", "seller_details"),
}
# The OpenSearch aggregations that count the FACETS above (top 50 values each; the label is the
# value's most common name; the nested seller facet counts products, not seller offers).
FACET_AGGS: dict[str, Any] = {
    "category": {"terms": {"field": "category_ids", "size": 50},
                 "aggs": {"label": {"terms": {"field": "category_name.keyword", "size": 1}}}},
    "sub_category": {"terms": {"field": "sub_category_id", "size": 50},
                     "aggs": {"label": {"terms": {"field": "sub_category_name.keyword", "size": 1}}}},
    "product_type": {"terms": {"field": "product_type_id", "size": 50},
                     "aggs": {"label": {"terms": {"field": "product_type.keyword", "size": 1}}}},
    "seller": {"nested": {"path": "seller_details"}, "aggs": {"values": {
        "terms": {"field": "seller_details.seller_id", "size": 50},
        "aggs": {"label": {"terms": {"field": "seller_details.seller_name.keyword", "size": 1}},
                 "products": {"reverse_nested": {}}},
    }}},
}

# Nested image field written by the ingest pipeline: [{image_url, image_vector}, ...].
IMAGE_PATH = "images"
IMAGE_VECTOR_FIELD = "images.image_vector"

# ------------------------------------------------------------------------------------------------
# Text fields searched by BM25, per request language. "field^N" = weight N (a match there ranks
# higher). Fields missing from the index are skipped (``lenient``), so German search still works on
# ``content`` until the index carries the ``*_de`` fields.
# ------------------------------------------------------------------------------------------------
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
# The typo-tolerant match searches the same fields WITHOUT the material numbers, so a mistyped
# number never returns a different, near-identical number.
FUZZY_FIELDS_BY_LANGUAGE: dict[str, list[str]] = {
    "en": ["sku_name^7", "product_name^7", "product_description^3", "category_name^1",
           "sub_category_name^1", "product_type^1", "content^1"],
    "de": ["sku_name_de^7", "product_name_de^7", "product_description_de^3", "category_name_de^1",
           "sub_category_name_de^1", "product_type_de^1", "content^1"],
}
# Seller fields live in the nested ``seller_details`` objects, so a separate nested query searches them.
SELLER_FIELDS = [
    "seller_details.seller_name^5",
    "seller_details.seller_id^5",
]
# Field the autocomplete matches on (and shows), per language.
SUGGEST_FIELD_BY_LANGUAGE: dict[str, str] = {"en": "product_name", "de": "product_name_de"}

# Identifier variant fields written by the ingest: each holds every form a user may type for the
# number — as stored, without leading zeros, split by a space, with its label ("asmo", "aramco",
# "customer") — e.g. ["4600005071", "customer4600005071", "customer 4600005071"]. A query (or a
# number word in it) equal to one of them finds the product exactly.
IDENTIFIER_VARIANT_FIELDS = (
    "asmo_material_number_trim_0",
    "aramco_material_number_trim_0",
    "asmo_purchase_aggrement_number_trim_0",
    "aramco_purchase_aggrement_number_trim_0",
)


class SearchService:
    def __init__(self, os_client: OpenSearchClientService) -> None:
        self._os = os_client
        # Set by setup_pipeline(): False when the cluster rejects weighted RRF; text + image search
        # then fuses the same 3 lists with the plain RRF pipeline.
        self._text_image_fusion = False

    async def setup_pipeline(self) -> None:
        """Create the RRF search pipelines (called once at startup)."""
        # 1. Weighted text + image pipeline — always (re)written so weight changes apply.
        #    Optional: if the cluster rejects it, text + image search uses the plain pipeline.
        w = _cfg.text_image_image_weight
        weights = [(1 - w) / 2, (1 - w) / 2, w]  # BM25, text kNN, image kNN
        try:
            await self._os.create_search_pipeline(TEXT_IMAGE_PIPELINE_ID, rank_constant=60, weights=weights)
            self._text_image_fusion = True
            logger.info(f"Search — pipeline '{TEXT_IMAGE_PIPELINE_ID}' ready (weights [BM25, text kNN, image kNN]={weights})")
        except Exception as exc:
            self._text_image_fusion = False
            logger.warning(
                f"Search — weighted RRF not available ({type(exc).__name__}: {exc}); "
                "text + image search uses plain RRF",
                extra={"status": "error", "error_details": str(exc)},
            )

        # 2. Plain RRF pipeline — required, created only if missing.
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
        multi_select_filters: dict[str, list[dict]] | None = None,
    ) -> SearchApiResponse:
        """Run a text / image / text + image / filter-browse search (the mode is picked from the inputs).

        ``filters`` are ready-made OpenSearch clauses. ``sort_by="price"`` sorts by price; anything
        else keeps relevance order. ``language`` picks the BM25 text fields (``de`` -> ``*_de`` + content).
        ``multi_select_filters`` maps a multi-select facet (sub_category, product_type) to its own
        clauses in ``filters``: that facet is counted without them, so it still lists the values the
        user can add (the other facets are counted with every filter).

        Steps:
          1. Start the multi-select facet recounts (they run alongside the main search).
          2. Build the query for the mode:
               2a. filter-browse  — no query, no image: just the filters
               2b. image          — nested kNN on the product images
               2c. text (+ image) — find which keyword match works (as typed -> partial words ->
                   only the meaningful words -> semantic fallback), then build the final query
          3. Request body: paging, sort, facet aggregations.
          4. Run it (a page after the last result becomes an empty page).
          5. Hits -> products.
          6. Aggregations -> facets.
        """
        cfg = _cfg
        index = cfg.opensearch_index_name
        has_image = image_vector is not None
        filters = list(filters)  # never modify the caller's list
        pipeline_id: str | None = None
        logger.info(
            f"Search — start: query={query!r}, language={language}, text_vector={query_vector is not None}, "
            f"image_vector={has_image}, filters={len(filters)}, sort={sort_by}/{sort_order}, page={page}x{page_size}",
            extra={"search_query": query, "language": language,
                   "input_data": {"filters": filters, "sort_by": sort_by, "sort_order": sort_order,
                                  "page": page, "page_size": page_size, "request_id": request_id}},
        )

        # ---------------------------------------------------------------------------------------
        # 1. Multi-select facets (sub_category, product_type) with a selection are each counted by
        #    the same search WITHOUT their own filter. Started now, awaited in steps 2c / 6.
        # ---------------------------------------------------------------------------------------
        recount = {key: clauses for key, clauses in (multi_select_filters or {}).items() if key in FACETS and clauses}
        recount_task: asyncio.Future | None = None
        if recount:
            logger.info(f"Search — counting multi-select facets without their own filter: {sorted(recount)}")
            recount_task = asyncio.gather(*(
                self.search(
                    query=query, query_vector=query_vector, image_vector=image_vector,
                    filters=[f for f in filters if not any(f is c for c in clauses)],
                    sort_by=None, sort_order=sort_order, page=1, page_size=1,
                    request_id=request_id, language=language,
                )
                for clauses in recount.values()
            ))

        # ---------------------------------------------------------------------------------------
        # 2. Build the query.
        # ---------------------------------------------------------------------------------------
        # Per-query HNSW candidate count (Lucene ignores the index-level ef_search setting).
        knn_params = {"method_parameters": {"ef_search": cfg.knn_ef_search}} if cfg.knn_ef_search else {}
        # How many candidates each part of the search collects before merging and paging. FIXED for
        # every page (OpenSearch needs the same pagination_depth on every page), so the total and the
        # order stay the same across pages. Results beyond it can't be paged to.
        depth = cfg.search_depth

        # The image kNN (used by 2b and by text + image in 2c). Each SKU scores by its best image;
        # function_score with no functions keeps that score, and min_score drops weak matches.
        image_knn: dict[str, Any] = {}
        image_clause: dict[str, Any] = {}
        if has_image:
            image_knn = {"vector": image_vector, "k": depth, **knn_params}
            image_clause = {"nested": {"path": IMAGE_PATH, "score_mode": "max",
                                       "query": {"knn": {IMAGE_VECTOR_FIELD: image_knn}}}}
            if cfg.image_min_score > 0:
                image_clause = {"function_score": {"query": image_clause, "min_score": cfg.image_min_score}}

        if not query and not has_image:
            # 2a. Filter-browse.
            mode = "filter-browse"
            query_clause: dict[str, Any] = {"bool": {"filter": filters}} if filters else {"match_all": {}}

        elif not query:
            # 2b. Image only: one kNN list needs no RRF. Filters go INSIDE the kNN (pre-filter).
            mode = "image"
            if filters:
                image_knn["filter"] = {"bool": {"filter": filters}}
            query_clause = image_clause

        else:
            # 2c. Text, or text + image.
            fields = KEYWORD_FIELDS_BY_LANGUAGE.get(language, KEYWORD_FIELDS)
            fuzzy_fields = FUZZY_FIELDS_BY_LANGUAGE.get(language, FUZZY_FIELDS_BY_LANGUAGE["en"])
            min_words = cfg.text_min_words_match  # e.g. "2<75%": 1-2 words -> all, 3+ words -> 75%
            words = query.split()
            # Words with a digit are identifiers (material / part numbers, sizes): always matched exactly.
            has_digit = {w: any(ch.isdigit() for ch in w) for w in words}
            # Words that may also match as the start of a longer word ("elec" -> electrical).
            prefix_words = [w for w in words if cfg.text_prefix_min_chars
                            and len(w) >= cfg.text_prefix_min_chars and not has_digit[w]]

            # How ONE query word can match a product — built once here, reused by every clause below:
            #   exact[w]   — the word as typed, on every text field
            #   typo[w]    — with typos, text fields only (not for identifiers)
            #   partial[w] — as the start of a longer word (prefix_words only)
            exact = {w: {"multi_match": {"query": w, "fields": fields, "lenient": True}} for w in words}
            typo = {w: {"multi_match": {"query": w, "fields": fuzzy_fields, "lenient": True,
                                        "fuzziness": cfg.text_fuzziness, "prefix_length": 1}}
                    for w in words if cfg.text_fuzziness and not has_digit[w]}
            partial = {w: {"multi_match": {"query": w, "fields": fuzzy_fields, "type": "phrase_prefix",
                                           "lenient": True, "max_expansions": cfg.text_prefix_max_expansions}}
                       for w in prefix_words}

            # Material numbers: identifier words of 5+ characters match aramco/asmo_material_number
            # exactly (any case), also with ASMO's "a-" prefix. Shorter ones ("2", "10mm") never do.
            number_values = list(dict.fromkeys(
                value for w in words if len(w) >= 5 and has_digit[w]
                for value in (w, w if w.lower().startswith("a-") else f"a-{w}")
            ))
            # Identifier variants ("aramco 00000000 1000805430", "customer4600005071"): the whole query,
            # and each number word in it, matched exactly against the variant fields' ``.keyword``
            # sub-field (extra spaces collapsed; id_normalizer applies to the query term too, plus
            # case_insensitive), so a value matches only as a whole. A number word needs 5+ digits
            # after its leading zeros, so "00000000" alone never matches every zero-padded number; a
            # query without one ("10mm cable") gets no identifier clause at all.
            number_words = [w.lower() for w in words if has_digit[w] and len(w.lstrip("0")) >= 5]
            variant_values = list(dict.fromkeys([" ".join(words).lower(), *number_words])) if number_words else []
            material_match: dict[str, Any] | None = None
            if number_values or variant_values:
                material_match = {"bool": {"should": [
                    *[{"term": {field: {"value": value, "case_insensitive": True}}}
                      for field in ("aramco_material_number", "asmo_material_number") for value in number_values],
                    *[{"term": {f"{field}.keyword": {"value": value, "case_insensitive": True}}}
                      for field in IDENTIFIER_VARIANT_FIELDS for value in variant_values],
                ], "minimum_should_match": 1}}
            # The same values with typos allowed — used only when no product matches exactly (below).
            # The fuzzy score falls with the edit distance, so the closest values rank first.
            fuzzy_identifier: dict[str, Any] | None = None
            if variant_values and cfg.identifier_fuzziness:
                fuzzy_identifier = {"bool": {"should": [
                    {"fuzzy": {f"{field}.keyword": {"value": value, "fuzziness": cfg.identifier_fuzziness,
                                                    "prefix_length": 0, "max_expansions": 50}}}
                    for field in IDENTIFIER_VARIANT_FIELDS for value in variant_values
                ], "minimum_should_match": 1}}
            identifier_fuzzy = False

            # Find a keyword match that some product has. Each attempt builds the BM25 query and (text
            # only) checks with a cheap size-0 search whether any product matches it:
            #   strict  — the words as typed (exact or with a typo)
            #   prefix  — also as partial words (skipped when no word can be partial)
            #   relaxed — only the query words the products closest in meaning contain
            # Nothing matches -> semantic fallback (close products only), or 0 results.
            keyword_query = query
            semantic_fallback = False
            for attempt in ("strict", "prefix", "relaxed"):
                if attempt == "prefix" and not prefix_words:
                    continue
                attempt_words = keyword_query.split()
                use_prefix = attempt != "strict" and bool(prefix_words)  # stays on for "relaxed"

                product_match = {"multi_match": {"query": keyword_query, "fields": fields,
                                                 "type": "best_fields", "lenient": True}}
                seller_match = {"multi_match": {"query": keyword_query, "fields": SELLER_FIELDS,
                                                "type": "best_fields", "lenient": True}}
                # Typo-tolerant match. With identifiers mixed in ("JL086A powr supply") each word is
                # matched on its own (identifier exact, the rest with typos); only identifiers -> none.
                fuzzy_match: dict[str, Any] | None = None
                if cfg.text_fuzziness and not all(has_digit[w] for w in attempt_words):
                    if any(has_digit[w] for w in attempt_words):
                        fuzzy_match = {"bool": {"should": [exact[w] if has_digit[w] else typo[w] for w in attempt_words],
                                                "minimum_should_match": min_words or 1}}
                    else:
                        fuzzy_match = {"multi_match": {"query": keyword_query, "fields": fuzzy_fields,
                                                       "type": "best_fields", "lenient": True,
                                                       "fuzziness": cfg.text_fuzziness, "prefix_length": 1}}
                if min_words:
                    for clause in (product_match, seller_match, fuzzy_match):
                        if clause and "multi_match" in clause:
                            clause["multi_match"]["minimum_should_match"] = min_words
                # Partial words: each word matches exactly, with a typo, or as a prefix.
                prefix_match: dict[str, Any] | None = None
                if use_prefix:
                    prefix_match = {"bool": {"should": [
                        {"bool": {"should": [c for c in (exact[w], typo.get(w), partial.get(w)) if c],
                                  "minimum_should_match": 1}}
                        for w in attempt_words
                    ], "minimum_should_match": min_words or 1}}
                # minimum_should_match=1 is required: OpenSearch merges the hybrid `filter` into each
                # sub-query, and a bool with a filter makes its should clauses optional otherwise.
                bm25 = {"bool": {"should": [
                    product_match,
                    {"nested": {"path": "seller_details", "score_mode": "max", "ignore_unmapped": True,
                                "query": seller_match}},
                    *[c for c in (fuzzy_match, prefix_match, material_match) if c],
                ], "minimum_should_match": 1}}

                # No keyword check: text + image must still return the image-only matches.
                if not cfg.text_require_keyword_match or has_image:
                    break
                check = await self._os.search(index, {
                    "size": 0, "track_total_hits": 1, "terminate_after": 1,
                    "query": {"bool": {"filter": [bm25, *filters]}},
                })
                if check["hits"]["total"]["value"] > 0:
                    logger.info(f"Search — keyword match found ({attempt}: {keyword_query!r}); "
                                f"vectors only re-rank the keyword matches")
                    break
                if attempt == "strict" and prefix_words:
                    logger.info(f"Search — no product has {query!r} as typed -> retrying with partial words {prefix_words}")
                    continue

                # A mistyped identifier ("4600005017", "asmo400000251"): no product has it exactly, so
                # try the variant fields with up to identifier_fuzziness edits. Only as a fallback —
                # neighbouring numbers (970000352152 / ...153) are 1-2 edits apart, so a correct number
                # must never pull them in.
                if fuzzy_identifier and not identifier_fuzzy:
                    fuzzy_check = await self._os.search(index, {
                        "size": 0, "track_total_hits": 1, "terminate_after": 1,
                        "query": {"bool": {"filter": [fuzzy_identifier, *filters]}},
                    })
                    if fuzzy_check["hits"]["total"]["value"] > 0:
                        identifier_fuzzy = True
                        logger.info(f"Search — no exact identifier {variant_values} -> fuzzy match "
                                    f"(up to {cfg.identifier_fuzziness} edits)")
                        break

                if attempt != "relaxed" and len(words) > 1 and query_vector is not None and cfg.text_relax_neighbors > 0:
                    # Which query words do the products closest in meaning contain? Keep those — unless a
                    # word is in too large a share of the WHOLE catalogue ("for", "with"): it describes
                    # nothing. "i want steel valvs" -> "steel valvs".
                    word_aggs = {f"word_{i}": {"filter": {"bool": {"should": [
                        exact[w],
                        {"nested": {"path": "seller_details", "ignore_unmapped": True, "query": {
                            "multi_match": {"query": w, "fields": SELLER_FIELDS, "lenient": True}}}},
                        *[c for c in (typo.get(w), partial.get(w)) if c],
                    ], "minimum_should_match": 1}}} for i, w in enumerate(words)}
                    # The nearest products with their scores. Each word clause is named with boost 0: the
                    # score stays the kNN score, and each hit lists which words it contains
                    # (matched_queries). Only neighbours scoring at least text_relax_min_score count: the
                    # nearest are returned however far away they are, and far ones ("laptop" -> steel
                    # tubes) would keep their words ("work") and return unrelated products.
                    # The kNN only looks at products containing at least one query word: across the whole
                    # catalogue, products close in meaning but sharing no word ("I want lamps" -> other
                    # lighting) can fill every neighbour slot, so the one that has "lamps" is never seen
                    # and no word is kept.
                    any_word = {"bool": {"should": [clause["filter"] for clause in word_aggs.values()],
                                         "minimum_should_match": 1}}
                    neighbours = await self._os.search(index, {
                        "size": cfg.text_relax_neighbors,
                        "_source": ["sku_name", "product_name"],
                        "query": {"bool": {
                            "must": [{"knn": {cfg.opensearch_vector_field: {
                                "vector": query_vector, "k": cfg.text_relax_neighbors, **knn_params,
                                "filter": {"bool": {"filter": [*filters, any_word]}},
                            }}}],
                            "should": [{"constant_score": {"filter": clause["filter"], "boost": 0, "_name": name}}
                                       for name, clause in word_aggs.items()],
                        }},
                        "aggs": {"catalogue": {"global": {}, "aggs": word_aggs}},
                    })
                    neighbour_hits = neighbours.get("hits", {}).get("hits", [])
                    close = [h for h in neighbour_hits if (h.get("_score") or 0) >= cfg.text_relax_min_score]
                    found = neighbours.get("aggregations", {})
                    catalogue = found.get("catalogue", {})
                    catalogue_size = catalogue.get("doc_count", 0)
                    in_neighbours = {w: sum(f"word_{i}" in h.get("matched_queries", []) for h in close)
                                     for i, w in enumerate(words)}
                    share = {w: catalogue.get(f"word_{i}", {}).get("doc_count", 0) / catalogue_size if catalogue_size else 0.0
                             for i, w in enumerate(words)}
                    kept = [w for w in words if in_neighbours[w] > 0 and share[w] <= cfg.text_relax_max_word_share]
                    nearest = [(round(h.get("_score") or 0, 4), (h.get("_source") or {}).get("sku_name")
                                or (h.get("_source") or {}).get("product_name")) for h in neighbour_hits[:5]]
                    logger.info(f"Search — relax {query!r}: {len(close)} of the {len(neighbour_hits)} nearest products "
                                f"score >= {cfg.text_relax_min_score} (top 5: {nearest}); words in those {in_neighbours}, "
                                f"catalogue share { {w: round(s, 3) for w, s in share.items()} } "
                                f"(max {cfg.text_relax_max_word_share}) -> keep {kept}")
                    if kept and len(kept) < len(words):
                        keyword_query = " ".join(kept)  # the text vector still uses the full query
                        continue

                if query_vector is not None and cfg.semantic_fallback_min_score > 0:
                    semantic_fallback = True
                    logger.info(f"Search — no keyword match for {query!r} -> semantic fallback "
                                f"(score >= {cfg.semantic_fallback_min_score})")
                    break
                logger.info(f"Search — no product contains {query!r} -> 0 results",
                            extra={"search_query": query, "status": "no_results"})
                # The multi-select facets may still have values to pick (without their own filter).
                recounted = await recount_task if recount_task else []
                return SearchApiResponse(request_id=request_id, total_results=0, page=page, page_size=page_size,
                                         products=[], facets=SearchFacets(**{
                                             key: getattr(other.facets, key) for key, other in zip(recount, recounted)}))

            # Text only and the query names a material number some product has: rank by keyword score
            # with the exact owners pinned on top (RRF ranks by position, and a bare number's vector
            # is noise, so the exact owner could fall below near-identical numbers).
            exact_number_hits = 0
            if material_match and not has_image and not semantic_fallback and not identifier_fuzzy:
                exact_check = await self._os.search(index, {
                    "size": 0, "track_total_hits": 1, "terminate_after": 1,
                    "query": {"bool": {"filter": [material_match, *filters]}},
                })
                exact_number_hits = exact_check["hits"]["total"]["value"]

            if semantic_fallback:
                # Plain kNN on the full query's text vector, filters inside the kNN; min_score drops the
                # weak neighbours, so total_results, paging and facets count only close products.
                mode = f"text (semantic fallback, score >= {cfg.semantic_fallback_min_score})"
                fallback_knn: dict[str, Any] = {"vector": query_vector, "k": cfg.semantic_fallback_k, **knn_params}
                if filters:
                    fallback_knn["filter"] = {"bool": {"filter": filters}}
                query_clause = {"function_score": {"query": {"knn": {cfg.opensearch_vector_field: fallback_knn}},
                                                   "min_score": cfg.semantic_fallback_min_score}}

            elif identifier_fuzzy:
                # Only the products whose identifier is within the allowed edits, closest first.
                mode = f"identifier (fuzzy, up to {cfg.identifier_fuzziness} edits)"
                query_clause = {"bool": {"must": [fuzzy_identifier], "filter": filters}}

            elif exact_number_hits:
                mode = "material number (exact match first)"
                query_clause = {"bool": {
                    "must": [bm25],
                    "should": [{"constant_score": {"filter": material_match, "boost": 100000}}],
                    "filter": filters,
                }}

            else:
                # Hybrid query fused by RRF. The "gate" decides which products may be returned at all
                # (without it the text kNN would add its nearest products, related or not):
                #   text         -> the product contains the query words
                #   text + image -> contains the query words OR its image matches; the text kNN then
                #                   ranks keyword matches only, so image-only look-alikes can't outrank them
                text_knn = {"knn": {cfg.opensearch_vector_field: {
                    "vector": query_vector, "k": cfg.hybrid_retrieval_k, **knn_params,
                }}}
                gated = cfg.text_require_keyword_match
                if gated and has_image:
                    filters.append({"bool": {"should": [bm25, image_clause], "minimum_should_match": 1}})
                    text_knn["knn"][cfg.opensearch_vector_field]["filter"] = bm25
                elif gated:
                    filters.append(bm25)

                # Sub-query order matches the pipeline weights: [BM25, text kNN, image kNN]. Without
                # weighted RRF the same 3 lists use plain RRF (a keyword match gets 2 votes, an
                # image-only look-alike 1, a product matching both 3).
                if not has_image:
                    mode = "text (keyword matches only)" if gated else "text"
                    subqueries = [bm25, text_knn]
                    pipeline_id = PIPELINE_ID
                elif self._text_image_fusion:
                    mode = "text+image (keyword or image matches)" if gated else "text+image"
                    subqueries = [bm25, text_knn, image_clause]
                    pipeline_id = TEXT_IMAGE_PIPELINE_ID
                else:
                    mode = "text+image (keyword or image matches, plain RRF)" if gated else "text+image (plain RRF)"
                    subqueries = [bm25, text_knn, image_clause]
                    pipeline_id = PIPELINE_ID
                # A hybrid query must be the top-level query, so the filters go in its own `filter`.
                hybrid: dict[str, Any] = {"queries": subqueries, "pagination_depth": depth}
                if filters:
                    hybrid["filter"] = {"bool": {"filter": filters}}
                query_clause = {"hybrid": hybrid}

        # ---------------------------------------------------------------------------------------
        # 3. Request body. Embedding vectors are never returned (large and useless to clients).
        # ---------------------------------------------------------------------------------------
        body: dict[str, Any] = {
            "from": (page - 1) * page_size,
            "size": page_size,
            "_source": {"excludes": [cfg.opensearch_vector_field, IMAGE_PATH]},
            "query": query_clause,
            "aggs": FACET_AGGS,
        }
        if sort_by == "price":
            body["sort"] = [{"price": {"order": sort_order}}]
        logger.info(f"Search — mode={mode}, pipeline={pipeline_id}, from={body['from']}, size={body['size']}, "
                    f"sort={'price ' + sort_order if sort_by == 'price' else 'relevance'}",
                    extra={"search_query": query, "search_mode": mode, "pipeline": pipeline_id})

        # ---------------------------------------------------------------------------------------
        # 4. Run it. A hybrid search asked for a page after its last result gets "Reached end of
        #    search result" (400) from OpenSearch: return an empty page with page 1's total + facets.
        # ---------------------------------------------------------------------------------------
        start = time.perf_counter()
        try:
            raw = await self._os.search(index, body, pipeline_id=pipeline_id)
        except ApiError as exc:
            if "Reached end of search result" not in str(exc.detail):
                raise
            logger.info(f"Search — page {page} is after the last result -> empty page")
            raw = await self._os.search(index, {**body, "from": 0}, pipeline_id=pipeline_id)
            raw["hits"]["hits"] = []
        wall_ms = (time.perf_counter() - start) * 1000
        hits = raw["hits"]["hits"]
        total = raw["hits"]["total"]["value"]
        logger.info(
            f"Search — {mode}: {total} results, {len(hits)} on page {page} "
            f"(OpenSearch {wall_ms:.1f}ms, server {raw.get('took', 0)}ms)",
            extra={"status": "success" if hits else "no_results", "search_query": query, "search_mode": mode,
                   "latency": f"{wall_ms:.1f}ms",
                   "response_payload": {"total_results": total, "returned": len(hits),
                                        "max_score": raw["hits"].get("max_score"), "took_ms": raw.get("took", 0)}},
        )

        # ---------------------------------------------------------------------------------------
        # 5. Hits -> products (frontend contract 2.4), one log line per result.
        # ---------------------------------------------------------------------------------------
        products: list[ProductItem] = []
        for rank, hit in enumerate(hits, start=(page - 1) * page_size + 1):
            src = hit.get("_source", {})
            sellers: list[SellerDetail] = []
            for i, seller in enumerate(src.get("seller_details") or []):
                # Older indexes keep temporary_unavailable / under_agreement on the root document (the
                # primary, cheapest seller's offer): used for seller 0 when its offer has none.
                root = src if i == 0 else {}
                sellers.append(SellerDetail(
                    seller_id=seller.get("seller_id"),
                    name=seller.get("seller_name"),
                    lead_time=seller.get("lead_time"),
                    unit_type=seller.get("unit_type"),
                    price=seller.get("price"),
                    moq=seller["moq"] if seller.get("moq") is not None else "",  # the frontend expects "" for none
                    currency=seller.get("currency"),
                    temporary_unavailable=(seller["temporary_unavailable"] if seller.get("temporary_unavailable") is not None
                                           else root.get("temporary_unavailable")),
                    under_agreement=(seller["under_agreement"] if seller.get("under_agreement") is not None
                                     else root.get("under_agreement")),
                ))
            product = ProductItem(
                sku_id=src.get("sku_id"),
                sku_name=src.get("sku_name") or src.get("product_name"),  # older indexes only have product_name
                primary_image_url=src.get("primary_image_url"),
                asmo_material_number=src.get("asmo_material_number"),
                aramco_material_number=src.get("aramco_material_number"),
                seller_details=sellers,
            )
            products.append(product)
            logger.info(
                f"Search — #{rank}: score={hit.get('_score')}, sku_id={product.sku_id}, name={product.sku_name!r}, "
                f"aramco={product.aramco_material_number}, asmo={product.asmo_material_number}, "
                f"sellers={[s.name for s in sellers]}, prices={[s.price for s in sellers]}",
                extra={"search_query": query, "search_mode": mode,
                       "result": {"rank": rank, "score": hit.get("_score"), "id": hit.get("_id"),
                                  "sku_id": product.sku_id, "name": product.sku_name,
                                  "aramco_material_number": product.aramco_material_number,
                                  "asmo_material_number": product.asmo_material_number,
                                  "sellers": [s.name for s in sellers], "price": src.get("price")}},
            )

        # ---------------------------------------------------------------------------------------
        # 6. Aggregations -> facets {id, value, count}. Nested facets (seller) count products, not
        #    offers. Multi-select facets take the counts made without their own filter (step 1).
        # ---------------------------------------------------------------------------------------
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
        if recount_task:
            for key, other in zip(recount, await recount_task):
                facets[key] = getattr(other.facets, key)
        logger.info(f"Search — done: {len(products)} products, total={total}, facets="
                    f"{ {key: len(values) for key, values in facets.items()} }",
                    extra={"search_query": query, "status": "success"})
        return SearchApiResponse(
            request_id=request_id,
            total_results=total,
            page=page,
            page_size=page_size,
            products=products,
            facets=SearchFacets(**facets),
        )

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
