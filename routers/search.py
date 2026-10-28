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
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import settings as _cfg
from utils.keycloak_auth import require_auth
from utils.logger import logger
from models.response import (
    ErrorResponse,
    IndicesResponse,
    IndexInfo,
    SearchApiResponse,
    SuggestionsResponse,
)
from services.opensearch import IndexNotFoundError, OpenSearchQueryError
from services.embedding_store import (
    EmbeddingStoreUnavailable,
    RequestIdNotFound,
    embedding_store,
)
from services.search import ALLOWED_FILTER_KEYS, FILTER_ALT_FIELDS, FILTER_KEY_MAP, NESTED_FILTER_PATHS, search_service

index_router = APIRouter(prefix="/indices", tags=["Indices"])
v1_router = APIRouter(prefix="/v1", tags=["V1 Search API"])

_RETRIEVAL_K = _cfg.hybrid_retrieval_k
_EF_SEARCH = _cfg.knn_ef_search

# Contract-level enums (§2.1). Validated manually so unsupported values return a
# 400 with the standard error body , rather than FastAPI's default 422.
SUPPORTED_CHANNELS = frozenset({"web", "mobile"})
# The index holds English text only, so English is the only supported language.
SUPPORTED_LANGUAGES = frozenset({"en"})
SUPPORTED_SORT_BY = frozenset({"relevance", "price"})
SUPPORTED_SORT_ORDER = frozenset({"asc", "desc"})

# Image constraints (agreed with the frontend): the image is sent as binary in a
# normal multipart image part — a file part with Content-Type image/jpeg,
# image/png or image/webp. A plain text field (e.g. Base64) or any other
# Content-Type -> 400; over 10 MiB -> 413; bytes that are not a JPEG, PNG or WebP
# image (GIF included) -> 400. The bytes decide the format sent to the embedder.
IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
NOT_AN_IMAGE_PART = (
    "The image must be sent as binary in a multipart file part with Content-Type "
    "image/jpeg, image/png or image/webp."
)
UNSUPPORTED_IMAGE = "Unsupported image type. Only JPEG, PNG and WebP images are accepted."
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB (10485760 bytes)
# Multipart parts may be up to this size (above MAX_IMAGE_BYTES so oversized
# images reach the 413 check rather than a parser error).
MAX_PART_BYTES = 2 * MAX_IMAGE_BYTES
class _InMemorySpool(SpooledTemporaryFile):
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


def _image_mime(data: bytes) -> str | None:
    """Return the MIME type for JPEG/PNG/WebP bytes, else None (GIF is not supported)."""
    if data[:8].startswith(b"\x89PNG"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _api_error(http_status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=http_status, detail={"code": code, "message": message})


def _validate_enum(value: str, allowed: frozenset[str], field: str, code: str) -> None:
    if value not in allowed:
        logger.warning(f"POST /v1/search — unsupported {field}: '{value}'")
        raise _api_error(400, code, f"Unsupported {field} '{value}'. Allowed: {', '.join(sorted(allowed))}")


def _parse_filters(raw: str | None) -> dict[str, Any]:
    """Parse the `filters` form field (a JSON string per §2.1) into a dict."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _api_error(400, "INVALID_FILTER", f"filters must be a valid JSON object: {exc.msg}")
    if not isinstance(parsed, dict):
        raise _api_error(400, "INVALID_FILTER", "filters must be a JSON object")
    return parsed


# Value types per filter key (§5: invalid filter value -> 400). Keys not listed
# here are keyword fields and take strings.
NUMERIC_FILTER_KEYS = frozenset({"price", "moq", "supplier_count", "lead_time"})
BOOLEAN_FILTER_KEYS = frozenset({"contract_indicator", "temporary_unavailable"})
RANGE_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})


def _valid_scalar(key: str, value: Any) -> bool:
    if key in NUMERIC_FILTER_KEYS:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if key in BOOLEAN_FILTER_KEYS:
        return isinstance(value, bool)
    return isinstance(value, str) and value != ""


def _validate_filter_value(key: str, value: Any) -> None:
    """Raise 400 INVALID_FILTER_VALUE unless the value fits the field's type.

    Numeric fields accept a number, a non-empty list of numbers, or a range
    object such as {"gte": 50, "lte": 150}; boolean fields accept true/false;
    every other field accepts a string or a non-empty list of strings.
    """
    if isinstance(value, dict):
        ok = (
            key in NUMERIC_FILTER_KEYS
            and value
            and set(value) <= RANGE_OPERATORS
            and all(_valid_scalar(key, v) for v in value.values())
        )
    elif isinstance(value, list):
        ok = bool(value) and all(_valid_scalar(key, v) for v in value)
    else:
        ok = _valid_scalar(key, value)
    if not ok:
        logger.warning(f"POST /v1/search — rejected invalid value for filter '{key}': {value!r}")
        raise _api_error(400, "INVALID_FILTER_VALUE", f"Invalid value for filter '{key}': {json.dumps(value)}")


def _build_os_filters(filters: dict[str, Any]) -> list[dict]:
    os_filters: list[dict] = []
    for k, v in filters.items():
        if k not in ALLOWED_FILTER_KEYS:
            logger.warning(
                f"POST /v1/search — rejected invalid filter key '{k}'",
                extra={"status": "error", "error_details": f"Filter key '{k}' not in allowed list"},
            )
            raise _api_error(400, "INVALID_FILTER_KEY", f"The filter key '{k}' is not supported.")
        _validate_filter_value(k, v)

        def clause_for(os_field: str) -> dict:
            # {"lte": 500} -> range, ["A", "B"] -> terms (match any), scalar -> term.
            if isinstance(v, dict):
                return {"range": {os_field: v}}
            if isinstance(v, list):
                return {"terms": {os_field: v}}
            return {"term": {os_field: v}}

        fields = FILTER_ALT_FIELDS.get(k, (FILTER_KEY_MAP[k],))
        clause = clause_for(fields[0]) if len(fields) == 1 else {
            "bool": {"should": [clause_for(f) for f in fields], "minimum_should_match": 1}
        }
        if k in NESTED_FILTER_PATHS:
            clause = {"nested": {"path": NESTED_FILTER_PATHS[k], "query": clause}}
        os_filters.append(clause)
        logger.debug(f"POST /v1/search — filter: {k} -> {fields}")
    return os_filters


@index_router.get("", response_model=IndicesResponse, responses=_ERROR_RESPONSES, summary="List all indices")
@require_auth
async def list_indices(_request: Request) -> IndicesResponse:
    logger.info("GET /indices")
    try:
        raw = await search_service._os.list_indices()
        indices = [IndexInfo(**i) for i in raw]
        logger.info(f"GET /indices — found {len(indices)} indices")
        return IndicesResponse(indices=indices, total=len(indices))
    except Exception as exc:
        logger.exception(f"GET /indices failed: {exc}")
        raise _api_error(500, "INTERNAL_ERROR", str(exc))


class SearchForm(BaseModel):
    """Text fields of the multipart search request (§2.1)."""

    channel: str = Field(description="Client channel — 'web' or 'mobile'")
    language: str = Field(description="Response language — 'en'")
    search_query: str | None = Field(default=None, description="Free-text query. Omit for filter-only browse")
    sort_by: str | None = Field(default=None, description="Sort field — 'relevance' or 'price'")
    sort_order: str = Field(default="desc", description="Sort direction — 'asc' or 'desc'")
    filters: str | None = Field(default=None, description=(
        "Filter object encoded as a JSON string, e.g. {\"category\": [\"<category id>\"], \"price\": {\"gte\": 50, \"lte\": 150}}. "
        "Every facet key is a filter key and takes the facet's id: category, sub_category, product_type, seller "
        "(category_name, sub_category_name and sellername are still accepted). Also: price, moq, lead_time (number or range), "
        "material_type, availability, category_ids, currency, contract_indicator, "
        "temporary_unavailable, supplier_count, sub_category_id, product_type_id, sold_by, sku_id, product_key, "
        "asmo_material_number, aramco_material_number. A value may be a list (match any); numeric keys also take "
        "{gt, gte, lt, lte}. Any other key -> 400 INVALID_FILTER_KEY"
    ))
    page: int = Field(default=1, ge=1, description="1-based page number")
    page_size: int = Field(default=60, ge=1, le=200, description="Results per page (frontend default 60)")
    request_id: str | None = Field(default=None, description="request_id of a previous response: refines that search — its query and/or image are reused for whichever of search_query / image is omitted")


# The multipart body is parsed in the handler (not by FastAPI) so the image part
# is accepted as raw binary whether or not the client gives it a filename. This
# documents the same form for OpenAPI / Swagger.
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


async def _read_search_form(request: Request) -> tuple[SearchForm, tuple[bytes, str | None] | None]:
    """Parse the multipart body into the text fields and the image part.

    Returns:
        ``(fields, image)``; ``image`` is None when no (or an empty) image part
        was sent, else ``(bytes, declared_content_type)``. The content type is
        None when ``image`` came as a text field rather than a file part (no
        filename, e.g. Base64), which the caller rejects.
    """
    # The request must be multipart/form-data: text fields and the image are separate parts.
    content_type = request.headers.get("content-type", "")
    if content_type.split(";")[0].strip().lower() != "multipart/form-data":
        logger.warning(f"POST /v1/search — rejected Content-Type '{content_type or 'none'}'")
        raise _api_error(400, "INVALID_CONTENT_TYPE", "The request must be sent as multipart/form-data.")
    try:
        form = await request.form(max_part_size=MAX_PART_BYTES)
    except StarletteHTTPException as exc:
        if "maximum size" in str(exc.detail):
            raise _api_error(413, "IMAGE_TOO_LARGE", f"Image exceeds the maximum size of {MAX_IMAGE_BYTES} bytes.")
        raise _api_error(400, "VALIDATION_ERROR", f"Malformed multipart body: {exc.detail}")

    # Empty text fields count as absent, as with FastAPI's own form handling.
    fields = {k: v for k, v in form.multi_items() if k != "image" and isinstance(v, str) and v != ""}
    try:
        parsed = SearchForm(**fields)
    except ValidationError as exc:
        raise RequestValidationError([{**e, "loc": ("body", *e["loc"])} for e in exc.errors()])

    part = form.get("image")
    if isinstance(part, UploadFile):
        data = await part.read()
        return parsed, ((data, (part.content_type or "").split(";")[0].strip().lower()) if data else None)
    if isinstance(part, str) and part:
        return parsed, (part.encode("utf-8"), None)
    return parsed, None


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
    _t0 = time.perf_counter()

    form, image_part = await _read_search_form(request)
    channel, language = form.channel, form.language
    search_query, sort_by, sort_order = form.search_query, form.sort_by, form.sort_order
    filters, page, page_size = form.filters, form.page, form.page_size

    # Contract validation 
    _validate_enum(channel, SUPPORTED_CHANNELS, "channel", "UNSUPPORTED_CHANNEL")
    _validate_enum(language, SUPPORTED_LANGUAGES, "language", "UNSUPPORTED_LANGUAGE")
    if sort_by is not None:
        _validate_enum(sort_by, SUPPORTED_SORT_BY, "sort_by", "INVALID_SORT_BY")
    _validate_enum(sort_order, SUPPORTED_SORT_ORDER, "sort_order", "INVALID_SORT_ORDER")

    query_text = search_query.strip() if search_query and search_query.strip() else None
    previous_request_id = form.request_id

    logger.info(
        "POST /v1/search — received",
        extra={
            "search_query": search_query,
            "language": language,
            "channel": channel,
            "input_data": {
                "request_id": request_id,
                "page": page,
                "page_size": page_size,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "has_filters": bool(filters),
                "has_image": image_part is not None,
                "previous_request_id": previous_request_id,
            },
        },
    )

    # Image validation : not an image file part -> 400, oversized -> 413,
    # bytes not JPEG/PNG/WebP -> 400
    image_input: tuple[bytes, str] | None = None
    if image_part is not None:
        image_bytes, declared_type = image_part
        if declared_type not in IMAGE_CONTENT_TYPES:
            logger.warning(
                "POST /v1/search — rejected image: "
                + ("sent as a text field, not a file part" if declared_type is None else f"Content-Type '{declared_type}'")
            )
            # Another image type (e.g. image/gif) is an unsupported format; anything
            # else (text field, application/octet-stream, ...) is not an image part.
            is_other_image = declared_type is not None and declared_type.startswith("image/")
            raise _api_error(400, "UNSUPPORTED_IMAGE_TYPE", UNSUPPORTED_IMAGE if is_other_image else NOT_AN_IMAGE_PART)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            logger.warning(f"POST /v1/search — image too large: {len(image_bytes)} bytes")
            raise _api_error(413, "IMAGE_TOO_LARGE", f"Image exceeds the maximum size of {MAX_IMAGE_BYTES} bytes.")
        # The bytes decide the format sent to the embedder.
        detected = _image_mime(image_bytes)
        if detected is None:
            logger.warning("POST /v1/search — uploaded file is not a JPEG/PNG/WebP image")
            raise _api_error(400, "UNSUPPORTED_IMAGE_TYPE", UNSUPPORTED_IMAGE)
        image_input = (image_bytes, detected)

    # 'relevance' means default (no explicit sort field).
    effective_sort_by = None if sort_by in (None, "relevance") else sort_by

    logger.debug("POST /v1/search — parsing filters and building OpenSearch clauses")
    os_filters = _build_os_filters(_parse_filters(filters))

    has_filters = bool(os_filters)

    # Operation decision (§2.2) --
    if not query_text and image_input is None and not previous_request_id and not has_filters:
        logger.warning("POST /v1/search — rejected: no query, no image and no filters provided")
        raise _api_error(400, "EMPTY_REQUEST", "At least one of search_query, image, filters or request_id must be provided")

    embed_ms: float | None = None
    try:
        # One Redis entry per search — its query and/or image embeddings — under
        # its request_id. Sending a previous request_id refines that search; the
        # mode follows from the resolved context.
        _t_emb = time.perf_counter()
        ctx = await embedding_store.resolve(request_id, query_text, image_input, previous_request_id)
        embed_ms = (time.perf_counter() - _t_emb) * 1000
        logger.info(
            f"POST /v1/search — search context {ctx.request_id} resolved in {embed_ms:.1f}ms "
            f"({'reused from request_id' if ctx.reused else 'new'})"
        )
        has_query = bool(ctx.query)
        has_image = ctx.image_vector is not None
        mode = (
            "text+image" if has_query and has_image
            else "image" if has_image
            else "text" if has_query
            else "filter-browse"
        )
        logger.info(f"POST /v1/search — dispatching: mode={mode} filters={len(os_filters)}")

        if has_query or has_image:
            result = await search_service.search_v1(
                index=_cfg.opensearch_index_name,
                query=ctx.query,
                query_vector=ctx.query_vector,
                image_vector=ctx.image_vector,
                vector_field=_cfg.opensearch_vector_field,
                language=language,
                filters=os_filters,
                sort_by=effective_sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size,
                retrieval_k=_RETRIEVAL_K,
                ef_search=_EF_SEARCH,
                image_min_score=_cfg.image_min_score,
                request_id=ctx.request_id,
            )
        else:
            result = await search_service.filter_browse(
                index=_cfg.opensearch_index_name,
                filters=os_filters,
                sort_by=effective_sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size,
                request_id=ctx.request_id,
            )

        elapsed_ms = (time.perf_counter() - _t0) * 1000
        logger.info(
            f"POST /v1/search — completed in {elapsed_ms:.1f}ms"
            + (f" (embeddings {embed_ms:.1f}ms, search {elapsed_ms - embed_ms:.1f}ms)" if embed_ms is not None else ""),
            extra={
                "status": "success",
                "latency": f"{elapsed_ms:.1f}ms",
                "search_query": search_query,
                "language": language,
                "channel": channel,
                "response_payload": {
                    "request_id": request_id,
                    "context_request_id": result.request_id,
                    "total_results": result.total_results,
                    "returned": len(result.products),
                    "api_latency_ms": round(elapsed_ms, 1),
                    "embedding_ms": round(embed_ms, 1) if embed_ms is not None else None,
                },
            },
        )
        return result
    except HTTPException:
        raise
    except RequestIdNotFound as exc:
        logger.warning(f"POST /v1/search — request_id not found: {exc}")
        raise _api_error(400, "REQUEST_ID_NOT_FOUND", str(exc))
    except EmbeddingStoreUnavailable as exc:
        logger.warning(f"POST /v1/search — Redis unavailable while resolving a request_id: {exc}")
        raise _api_error(503, "EMBEDDING_STORE_UNAVAILABLE", f"request_id cannot be resolved ({exc}); send search_query / image instead.")
    except IndexNotFoundError as exc:
        logger.warning(f"POST /v1/search — index not found: {exc}", extra={"status": "error", "error_details": str(exc)})
        raise _api_error(404, "INDEX_NOT_FOUND", str(exc))
    except OpenSearchQueryError as exc:
        logger.warning(f"POST /v1/search — query error: {exc}", extra={"status": "error", "error_details": str(exc)})
        raise _api_error(400, "QUERY_ERROR", str(exc))
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - _t0) * 1000
        # exception(): include the traceback so the failing call is visible in the logs.
        logger.exception(
            f"POST /v1/search — failed after {elapsed_ms:.1f}ms: {exc}",
            extra={
                "status": "error",
                "latency": f"{elapsed_ms:.1f}ms",
                "search_query": search_query,
                "error_details": str(exc),
            },
        )
        raise _api_error(500, "INTERNAL_ERROR", str(exc))


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
    language: Annotated[str, Query(description="Response language — 'en'")],
    query_limit: Annotated[int, Query(ge=1, le=50, description="Max query suggestions to return")] = 5,
    product_limit: Annotated[int, Query(ge=1, le=50, description="Max product suggestions to return")] = 5,
) -> SuggestionsResponse:
    _t0 = time.perf_counter()

    _validate_enum(channel, SUPPORTED_CHANNELS, "channel", "UNSUPPORTED_CHANNEL")
    _validate_enum(language, SUPPORTED_LANGUAGES, "language", "UNSUPPORTED_LANGUAGE")

    logger.info(
        "GET /v1/search/suggestions — received",
        extra={
            "search_query": q,
            "language": language,
            "input_data": {
                "q": q,
                "channel": channel,
                "language": language,
                "query_limit": query_limit,
                "product_limit": product_limit,
            },
        },
    )

    try:
        result = await search_service.get_suggestions(
            index=_cfg.opensearch_index_name,
            query=q,
            language=language,
            query_limit=query_limit,
            product_limit=product_limit,
        )
        elapsed_ms = (time.perf_counter() - _t0) * 1000
        logger.info(
            f"GET /v1/search/suggestions — completed in {elapsed_ms:.1f}ms",
            extra={
                "status": "success",
                "search_query": q,
                "latency": f"{elapsed_ms:.1f}ms",
                "response_payload": {
                    "queries": len(result.query_suggestions),
                    "products": len(result.product_suggestions),
                    "api_latency_ms": round(elapsed_ms, 1),
                },
            },
        )
        return result
    except HTTPException:
        raise
    except IndexNotFoundError as exc:
        logger.warning(f"GET /v1/search/suggestions — index not found: {exc}", extra={"status": "error", "error_details": str(exc)})
        raise _api_error(404, "INDEX_NOT_FOUND", str(exc))
    except OpenSearchQueryError as exc:
        logger.warning(f"GET /v1/search/suggestions — query error: {exc}", extra={"status": "error", "error_details": str(exc)})
        raise _api_error(400, "QUERY_ERROR", str(exc))
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - _t0) * 1000
        logger.error(
            f"GET /v1/search/suggestions — failed after {elapsed_ms:.1f}ms: {exc}",
            extra={
                "status": "error",
                "search_query": q,
                "latency": f"{elapsed_ms:.1f}ms",
                "error_details": str(exc),
            },
        )
        raise _api_error(500, "INTERNAL_ERROR", str(exc))
