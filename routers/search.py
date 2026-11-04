"""HTTP endpoints: POST /v1/search, GET /v1/search/suggestions and GET /indices.

The endpoints validate the request against the frontend contract and map
backend failures to the error contract ({"code", "message"}); the search
logic itself lives in services/.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from tempfile import SpooledTemporaryFile
from typing import Annotated, Any

import starlette.formparsers as _starlette_formparsers
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from models.response import (
    ApiError,
    ErrorResponse,
    IndexInfo,
    IndicesResponse,
    SearchApiResponse,
    SuggestionsResponse,
)
from services.embedding_store import EmbeddingStoreUnavailable, RequestIdNotFound, embedding_store
from services.image_processor import MAX_IMAGE_BYTES, process_image
from services.opensearch import IndexNotFoundError, OpenSearchQueryError, os_client
from services.search import ALLOWED_FILTER_KEYS, FILTER_ALT_FIELDS, FILTER_KEY_MAP, NESTED_FILTER_PATHS, search_service
from utils.keycloak_auth import require_auth
from utils.logger import logger

index_router = APIRouter(prefix="/indices", tags=["Indices"])
v1_router = APIRouter(prefix="/v1", tags=["V1 Search API"])

# Contract-level enums (2.1). Checked by hand so unsupported values get a 400
# with the standard error body, rather than FastAPI's default 422.
SUPPORTED_CHANNELS = frozenset({"web", "mobile"})
SUPPORTED_LANGUAGES = frozenset({"en", "de"})  # en -> English fields, de -> *_de fields + content
SUPPORTED_SORT_BY = frozenset({"relevance", "price"})
SUPPORTED_SORT_ORDER = frozenset({"asc", "desc"})

# Filter value types (5: an invalid filter value -> 400). Keys not listed take strings.
NUMERIC_FILTER_KEYS = frozenset({"price", "moq", "supplier_count", "lead_time"})
BOOLEAN_FILTER_KEYS = frozenset({"contract_indicator", "temporary_unavailable", "under_agreement"})
RANGE_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})

# Multipart parts may be up to this size (above MAX_IMAGE_BYTES so an oversized
# image reaches our 413 check rather than failing in the parser).
MAX_PART_BYTES = 2 * MAX_IMAGE_BYTES


class _InMemorySpool(SpooledTemporaryFile):
    """Keeps uploaded parts (up to MAX_PART_BYTES) in memory instead of spilling them to a temp file."""

    def __init__(self, max_size: int = 0, *args: Any, **kwargs: Any) -> None:
        super().__init__(max(max_size, MAX_PART_BYTES), *args, **kwargs)


_starlette_formparsers.SpooledTemporaryFile = _InMemorySpool

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Bad request — unsupported channel/language, invalid filter, unknown request_id, or empty request"},
    401: {"model": ErrorResponse, "description": "Unauthorized — missing, malformed, invalid or expired Keycloak bearer token"},
    404: {"model": ErrorResponse, "description": "Not found — the target index does not exist"},
    413: {"model": ErrorResponse, "description": "Payload too large — uploaded image exceeds the size limit"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}
_SEARCH_ERROR_RESPONSES = {
    **_ERROR_RESPONSES,
    503: {"model": ErrorResponse, "description": "Service unavailable — a request_id was sent but the search-context store (Redis) is unreachable"},
}


class SearchForm(BaseModel):
    """Text fields of the multipart search request (2.1)."""

    channel: str = Field(description="Client channel — 'web' or 'mobile'")
    language: str = Field(description="Search language — 'en' or 'de' (German searches the *_de fields and content)")
    search_query: str | None = Field(default=None, description="Free-text query. Omit for filter-only browse")
    sort_by: str | None = Field(default=None, description="Sort field — 'relevance' or 'price'")
    sort_order: str = Field(default="desc", description="Sort direction — 'asc' or 'desc'")
    filters: str | None = Field(default=None, description=(
        "Filter object encoded as a JSON string, e.g. {\"category\": [\"<category id>\"], \"price\": {\"gte\": 50, \"lte\": 150}}. "
        "Every facet key is a filter key and takes the facet's id: category, sub_category, product_type, seller "
        "(category_name, sub_category_name and sellername are still accepted). Also: price, moq, lead_time (number or range), "
        "material_type, availability, category_ids, currency, contract_indicator, "
        "temporary_unavailable, under_agreement, supplier_count, sub_category_id, product_type_id, sold_by, sku_id, product_key, "
        "asmo_material_number, aramco_material_number. A value may be a list (match any); numeric keys also take "
        "{gt, gte, lt, lte}. Any other key -> 400 INVALID_FILTER_KEY"
    ))
    page: int = Field(default=1, ge=1, description="1-based page number")
    page_size: int = Field(default=60, ge=1, le=200, description="Results per page (frontend default 60)")
    request_id: str | None = Field(default=None, description="request_id of a previous response: refines that search — its query and/or image are reused for whichever of search_query / image is omitted")


# The multipart body is parsed in the handler (not by FastAPI) so the image part is
# accepted as raw binary whether or not the client gives it a filename. This documents
# the same form for OpenAPI / Swagger.
_SEARCH_REQUEST_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": ["channel", "language"],
                "properties": {
                    **SearchForm.model_json_schema()["properties"],
                    "image": {"type": "string", "format": "binary", "description": "Optional image for image / text+image search, sent as binary in a normal multipart image part: a file part with Content-Type image/jpeg, image/png or image/webp, max 10 MiB (10485760 bytes); the bytes must be a JPEG, PNG or WebP image. In the browser: form.append('image', file). A text field (e.g. Base64), any other Content-Type (application/octet-stream, image/gif, ...) or non-image bytes get 400 UNSUPPORTED_IMAGE_TYPE, larger files 413 IMAGE_TOO_LARGE"},
                },
            }
        }
    },
}


# Request validation


async def _check_allowed(value: str, allowed: frozenset[str], field: str, code: str) -> None:
    """400 ``code`` unless ``value`` is one of ``allowed``."""
    if value not in allowed:
        logger.warning(f"Rejected unsupported {field}: '{value}'")
        raise ApiError(400, code, f"Unsupported {field} '{value}'. Allowed: {', '.join(sorted(allowed))}")


async def _read_search_form(request: Request) -> tuple[SearchForm, tuple[bytes, str] | None]:
    """Parse and validate the multipart search body.

    Returns ``(form, image)``; ``image`` is the preprocessed ``(bytes, mime)`` or
    None when no (or an empty) image was sent.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.split(";")[0].strip().lower() != "multipart/form-data":
        logger.warning(f"POST /v1/search — rejected Content-Type '{content_type or 'none'}'")
        raise ApiError(400, "INVALID_CONTENT_TYPE", "The request must be sent as multipart/form-data.")
    try:
        multipart = await request.form(max_part_size=MAX_PART_BYTES)
    except StarletteHTTPException as exc:
        if "maximum size" in str(exc.detail):
            raise ApiError(413, "IMAGE_TOO_LARGE", f"Image exceeds the maximum size of {MAX_IMAGE_BYTES} bytes.")
        raise ApiError(400, "VALIDATION_ERROR", f"Malformed multipart body: {exc.detail}")

    # 1. Text fields. Empty fields count as absent, as with FastAPI's own form handling.
    fields = {k: v for k, v in multipart.multi_items() if k != "image" and isinstance(v, str) and v != ""}
    try:
        form = SearchForm(**fields)
    except ValidationError as exc:
        raise RequestValidationError([{**e, "loc": ("body", *e["loc"])} for e in exc.errors()])

    await _check_allowed(form.channel, SUPPORTED_CHANNELS, "channel", "UNSUPPORTED_CHANNEL")
    await _check_allowed(form.language, SUPPORTED_LANGUAGES, "language", "UNSUPPORTED_LANGUAGE")
    if form.sort_by is not None:
        await _check_allowed(form.sort_by, SUPPORTED_SORT_BY, "sort_by", "INVALID_SORT_BY")
    await _check_allowed(form.sort_order, SUPPORTED_SORT_ORDER, "sort_order", "INVALID_SORT_ORDER")

    # 2. The image: validate + EXIF fix + downscale + re-encode, in one step (see services/image_processor.py).
    #    Pillow is CPU-heavy, so it runs in a worker thread to keep the event loop free.
    image = await asyncio.to_thread(process_image, multipart.get("image"))
    return form, image


async def _build_filters(raw: str | None) -> list[dict]:
    """Turn the ``filters`` form field (a JSON object string, 2.1) into OpenSearch filter clauses.

    Numeric keys accept a number, a non-empty list of numbers, or a range such as
    {"gte": 50, "lte": 150}; boolean keys accept true/false; every other key accepts
    a non-empty string or a non-empty list of them. A list matches any of its values.
    """
    if not raw or not raw.strip():
        return []
    try:
        filters = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(400, "INVALID_FILTER", f"filters must be a valid JSON object: {exc.msg}")
    if not isinstance(filters, dict):
        raise ApiError(400, "INVALID_FILTER", "filters must be a JSON object")

    clauses: list[dict] = []
    for key, value in filters.items():
        if key not in ALLOWED_FILTER_KEYS:
            logger.warning(f"POST /v1/search — rejected unknown filter key '{key}'")
            raise ApiError(400, "INVALID_FILTER_KEY", f"The filter key '{key}' is not supported.")

        # Check the value's shape and the type of every scalar in it.
        is_range = isinstance(value, dict)
        scalars = list(value.values()) if is_range else value if isinstance(value, list) else [value]
        if key in NUMERIC_FILTER_KEYS:
            types_ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in scalars)
        elif key in BOOLEAN_FILTER_KEYS:
            types_ok = all(isinstance(v, bool) for v in scalars)
        else:
            types_ok = all(isinstance(v, str) and v != "" for v in scalars)
        range_ok = not is_range or (key in NUMERIC_FILTER_KEYS and set(value) <= RANGE_OPERATORS)
        if not (scalars and types_ok and range_ok):
            logger.warning(f"POST /v1/search — rejected invalid value for filter '{key}': {value!r}")
            raise ApiError(400, "INVALID_FILTER_VALUE", f"Invalid value for filter '{key}': {json.dumps(value)}")

        # {"lte": 500} -> range, ["A", "B"] -> terms (match any), scalar -> term.
        kind = "range" if is_range else "terms" if isinstance(value, list) else "term"
        fields = FILTER_ALT_FIELDS.get(key, (FILTER_KEY_MAP[key],))
        if len(fields) == 1:
            clause: dict = {kind: {fields[0]: value}}
        else:
            clause = {"bool": {"should": [{kind: {f: value}} for f in fields], "minimum_should_match": 1}}
        if key in NESTED_FILTER_PATHS:
            clause = {"nested": {"path": NESTED_FILTER_PATHS[key], "query": clause}}
        clauses.append(clause)
    return clauses


# Endpoints


@index_router.get("", response_model=IndicesResponse, responses=_ERROR_RESPONSES, summary="List all indices")
@require_auth
async def list_indices(_request: Request) -> IndicesResponse:
    start = time.perf_counter()
    try:
        indices = [IndexInfo(**i) for i in await os_client.list_indices()]
    except Exception as exc:
        logger.exception(f"GET /indices failed ({type(exc).__name__}): {exc}",
                         extra={"function_name": "list_indices", "status": "error", "error_details": str(exc)})
        raise ApiError(500, "INTERNAL_ERROR", str(exc))
    logger.info(f"GET /indices — {len(indices)} indices in {(time.perf_counter() - start) * 1000:.1f}ms")
    return IndicesResponse(indices=indices, total=len(indices))


@v1_router.post(
    "/search",
    response_model=SearchApiResponse,
    responses=_SEARCH_ERROR_RESPONSES,
    summary="Search products (text / image / text+image / filter-browse)",
    openapi_extra={"requestBody": _SEARCH_REQUEST_BODY},
)
@require_auth
async def v1_search(request: Request) -> SearchApiResponse:
    # This HTTP request's id (set by the app middleware); a new search context is stored under it.
    request_id: str = getattr(request.state, "request_id", "") or str(uuid.uuid4())
    start = time.perf_counter()
    try:
        form, image = await _read_search_form(request)
        filters = await _build_filters(form.filters)
        query = form.search_query.strip() if form.search_query and form.search_query.strip() else None
        if not query and image is None and not form.request_id and not filters:
            raise ApiError(400, "EMPTY_REQUEST", "At least one of search_query, image, filters or request_id must be provided")
        logger.info(
            "POST /v1/search — received",
            extra={"search_query": form.search_query, "language": form.language, "channel": form.channel,
                   "input_data": {"request_id": request_id, "page": form.page, "page_size": form.page_size,
                                  "sort_by": form.sort_by, "sort_order": form.sort_order,
                                  "filter_count": len(filters), "has_image": image is not None,
                                  "previous_request_id": form.request_id}},
        )

        # One Redis entry per search holds its query/image embeddings under its request_id.
        # Sending a previous request_id refines that search.
        embed_start = time.perf_counter()
        ctx = await embedding_store.resolve(request_id, query, image, form.request_id)
        embed_ms = (time.perf_counter() - embed_start) * 1000

        result = await search_service.search(
            query=ctx.query,
            query_vector=ctx.query_vector,
            image_vector=ctx.image_vector,
            filters=filters,
            sort_by=form.sort_by,
            sort_order=form.sort_order,
            page=form.page,
            page_size=form.page_size,
            request_id=ctx.request_id,
            language=form.language,
        )
    except (HTTPException, RequestValidationError):
        raise  # already in the error contract
    except RequestIdNotFound as exc:
        logger.warning(f"POST /v1/search — request_id not found: {exc}", extra={"status": "error"})
        raise ApiError(400, "REQUEST_ID_NOT_FOUND", str(exc))
    except EmbeddingStoreUnavailable as exc:
        logger.warning(f"POST /v1/search — Redis unavailable while resolving a request_id: {exc}", extra={"status": "error"})
        raise ApiError(503, "EMBEDDING_STORE_UNAVAILABLE", f"request_id cannot be resolved ({exc}); send search_query / image instead.")
    except IndexNotFoundError as exc:
        raise ApiError(404, "INDEX_NOT_FOUND", str(exc))
    except OpenSearchQueryError as exc:
        raise ApiError(400, "QUERY_ERROR", str(exc))
    except Exception as exc:
        logger.exception(f"POST /v1/search — failed ({type(exc).__name__}): {exc}",
                         extra={"function_name": "v1_search", "status": "error", "error_details": str(exc)})
        raise ApiError(500, "INTERNAL_ERROR", str(exc))

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"POST /v1/search — completed in {elapsed_ms:.1f}ms (embeddings {embed_ms:.1f}ms, search {elapsed_ms - embed_ms:.1f}ms)",
        extra={"status": "success", "latency": f"{elapsed_ms:.1f}ms", "search_query": form.search_query,
               "response_payload": {"request_id": request_id, "context_request_id": result.request_id,
                                    "total_results": result.total_results, "returned": len(result.products)}},
    )
    return result


@v1_router.get(
    "/search/suggestions",
    response_model=SuggestionsResponse,
    responses=_ERROR_RESPONSES,
    summary="Autocomplete suggestions",
)
@require_auth
async def v1_suggestions(
    _request: Request,
    q: Annotated[str, Query(min_length=1, description="Partial text entered by the user")],
    channel: Annotated[str, Query(description="Client channel — 'web' or 'mobile'")],
    language: Annotated[str, Query(description="Search language — 'en' or 'de' (German suggests from product_name_de)")],
    query_limit: Annotated[int, Query(ge=1, le=50, description="Max query suggestions to return")] = 5,
    product_limit: Annotated[int, Query(ge=1, le=50, description="Max product suggestions to return")] = 5,
) -> SuggestionsResponse:
    start = time.perf_counter()
    await _check_allowed(channel, SUPPORTED_CHANNELS, "channel", "UNSUPPORTED_CHANNEL")
    await _check_allowed(language, SUPPORTED_LANGUAGES, "language", "UNSUPPORTED_LANGUAGE")
    try:
        result = await search_service.get_suggestions(
            query=q, query_limit=query_limit, product_limit=product_limit, language=language,
        )
    except IndexNotFoundError as exc:
        raise ApiError(404, "INDEX_NOT_FOUND", str(exc))
    except OpenSearchQueryError as exc:
        raise ApiError(400, "QUERY_ERROR", str(exc))
    except Exception as exc:
        logger.exception(f"GET /v1/search/suggestions — failed ({type(exc).__name__}): {exc}",
                         extra={"function_name": "v1_suggestions", "status": "error", "error_details": str(exc)})
        raise ApiError(500, "INTERNAL_ERROR", str(exc))

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"GET /v1/search/suggestions — completed in {elapsed_ms:.1f}ms",
        extra={"status": "success", "search_query": q, "latency": f"{elapsed_ms:.1f}ms",
               "response_payload": {"queries": len(result.query_suggestions), "products": len(result.product_suggestions)}},
    )
    return result
