from __future__ import annotations

import json
import time
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from config import settings as _cfg
from utils.logger import logger
from models.response import (
    ErrorResponse,
    IndicesResponse,
    IndexInfo,
    SearchApiResponse,
    SuggestionsResponse
)
from services.opensearch import IndexNotFoundError, OpenSearchQueryError
from services.search import ALLOWED_FILTER_KEYS, FILTER_KEY_MAP, search_service

index_router = APIRouter(prefix="/indices", tags=["Indices"])
v1_router = APIRouter(prefix="/v1", tags=["V1 Search API"])

_RETRIEVAL_K = _cfg.hybrid_retrieval_k

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Bad request — invalid filter key, empty request, or malformed query"},
    404: {"model": ErrorResponse, "description": "Not found — the target index does not exist"},
    422: {"model": ErrorResponse, "description": "Validation error — form fields failed validation"},
    501: {"model": ErrorResponse, "description": "Not implemented — image / text+image search"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}


def _api_error(http_status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=http_status, detail={"code": code, "message": message})


@index_router.get("", response_model=IndicesResponse, responses=_ERROR_RESPONSES, summary="List all indices")
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


@v1_router.post("/search", response_model=SearchApiResponse, responses=_ERROR_RESPONSES, summary="Search products (text / filter-browse)")
async def v1_search(
    request: Request,
    channel: Annotated[str, Form(description="Client channel — 'web' or 'mobile'")],
    language: Annotated[str, Form(description="Response language — 'en' or 'de'")],
    search_query: Annotated[str | None, Form(description="Free-text query. Omit for filter-only browse")] = None,
    sort_by: Annotated[str | None, Form(description="Sort field — 'relevance' or 'price'")] = None,
    sort_order: Annotated[str, Form(description="Sort direction — 'asc' or 'desc'")] = "desc",
    filters: Annotated[str | None, Form(description="Filter object as a JSON string")] = None,
    page: Annotated[int, Form(description="1-based page number")] = 1,
    page_size: Annotated[int, Form(description="Results per page")] = 60,
    image: Annotated[UploadFile | None, File(description="Optional image for image / text+image search")] = None,
) -> SearchApiResponse:
    request_id: str = getattr(request.state, "request_id", "")
    _t0 = time.perf_counter()

    has_image = image is not None

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
                "has_image": has_image,
            },
        },
    )

    # Validate required/enumerated form fields (multipart fields aren't type-validated by FastAPI).
    if channel not in ("web", "mobile"):
        raise _api_error(422, "VALIDATION_ERROR", "channel must be 'web' or 'mobile'")
    if language not in ("en", "de"):
        raise _api_error(422, "VALIDATION_ERROR", "language must be 'en' or 'de'")
    if sort_by not in (None, "", "relevance", "price"):
        raise _api_error(422, "VALIDATION_ERROR", "sort_by must be 'relevance' or 'price'")
    if sort_order not in ("asc", "desc"):
        raise _api_error(422, "VALIDATION_ERROR", "sort_order must be 'asc' or 'desc'")
    if page < 1:
        raise _api_error(422, "VALIDATION_ERROR", "page must be >= 1")
    if page_size < 1 or page_size > 200:
        raise _api_error(422, "VALIDATION_ERROR", "page_size must be between 1 and 200")

    # 'relevance' means default (no explicit sort field).
    effective_sort_by = None if sort_by in (None, "", "relevance") else sort_by

    # Parse the filters JSON string into a dict.
    filters_dict: dict = {}
    if filters:
        try:
            parsed = json.loads(filters)
        except json.JSONDecodeError as exc:
            raise _api_error(422, "VALIDATION_ERROR", f"filters must be a valid JSON string: {exc}")
        if not isinstance(parsed, dict):
            raise _api_error(422, "VALIDATION_ERROR", "filters must be a JSON object")
        filters_dict = parsed

    # Image / text+image search is not implemented yet (form-only phase).
    if has_image:
        logger.warning("POST /v1/search — image provided but image search is not implemented")
        raise _api_error(501, "NOT_IMPLEMENTED", "Image and text+image search are not yet implemented")

    logger.debug("POST /v1/search — building OpenSearch filter clauses")
    os_filters: list[dict] = []
    for k, v in filters_dict.items():
        if k not in ALLOWED_FILTER_KEYS:
            logger.warning(
                f"POST /v1/search — rejected invalid filter key '{k}'",
                extra={"status": "error", "error_details": f"Filter key '{k}' not in allowed list"},
            )
            raise _api_error(400, "INVALID_FILTER_KEY", f"Filter key '{k}' is not supported")
        os_field = FILTER_KEY_MAP[k]
        clause = {"range": {os_field: v}} if isinstance(v, dict) else {"term": {os_field: v}}
        os_filters.append(clause)
        logger.debug(f"POST /v1/search — filter: {k} -> {os_field}")

    has_query = bool(search_query and search_query.strip())
    has_filters = bool(os_filters)

    if not has_query and not has_filters:
        logger.warning("POST /v1/search — rejected: no query, no image and no filters provided")
        raise _api_error(400, "EMPTY_REQUEST", "At least one of search_query, image or filters must be provided")

    logger.info(
        f"POST /v1/search — dispatching: mode={'hybrid' if has_query else 'filter-browse'} filters={len(os_filters)}",
    )

    try:
        if has_query:
            result = await search_service.text_search_v1(
                index=_cfg.opensearch_index_name,
                query=search_query.strip(),
                vector_field=_cfg.opensearch_vector_field,
                language=language,
                filters=os_filters,
                sort_by=effective_sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size,
                retrieval_k=_RETRIEVAL_K,
                request_id=request_id,
            )
        else:
            result = await search_service.filter_browse(
                index=_cfg.opensearch_index_name,
                filters=os_filters,
                sort_by=effective_sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size,
            )

        elapsed_ms = (time.perf_counter() - _t0) * 1000
        logger.info(
            f"POST /v1/search — completed in {elapsed_ms:.1f}ms",
            extra={
                "status": "success",
                "latency": f"{elapsed_ms:.1f}ms",
                "search_query": search_query,
                "language": language,
                "channel": channel,
                "response_payload": {
                    "request_id": request_id,
                    "total": result.total,
                    "returned": len(result.results),
                    "total_pages": result.total_pages,
                    "search_type": result.search_type,
                    "took_ms": result.took_ms,
                    "api_latency_ms": round(elapsed_ms, 1),
                    "facets": len(result.facets),
                },
            },
        )
        return result
    except IndexNotFoundError as exc:
        logger.warning(f"POST /v1/search — index not found: {exc}", extra={"status": "error", "error_details": str(exc)})
        raise _api_error(404, "INDEX_NOT_FOUND", str(exc))
    except OpenSearchQueryError as exc:
        logger.warning(f"POST /v1/search — query error: {exc}", extra={"status": "error", "error_details": str(exc)})
        raise _api_error(400, "QUERY_ERROR", str(exc))
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - _t0) * 1000
        logger.error(
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
async def v1_suggestions(
    _request: Request,
    q: Annotated[str, Query(min_length=1, description="Typed prefix to match against product names and category names")],
    language: Annotated[Literal["en", "de"], Query(description="Response language")] = "en",
    product_limit: Annotated[int, Query(ge=1, le=20, description="Max product suggestions to return")] = 5,
    category_limit: Annotated[int, Query(ge=1, le=10, description="Max category suggestions to return")] = 3,
    query_limit: Annotated[int | None, Query(ge=1, le=20, description="When set, overrides product_limit")] = None,
) -> SuggestionsResponse:
    _t0 = time.perf_counter()

    logger.info(
        "GET /v1/search/suggestions — received",
        extra={
            "search_query": q,
            "language": language,
            "input_data": {
                "q": q,
                "language": language,
                "product_limit": product_limit,
                "category_limit": category_limit,
                "query_limit": query_limit,
            },
        },
    )

    eff_product_limit = query_limit or product_limit
    try:
        result = await search_service.get_suggestions(
            index=_cfg.opensearch_index_name,
            query=q,
            language=language,
            product_limit=eff_product_limit,
            category_limit=category_limit,
        )
        elapsed_ms = (time.perf_counter() - _t0) * 1000
        logger.info(
            f"GET /v1/search/suggestions — completed in {elapsed_ms:.1f}ms",
            extra={
                "status": "success",
                "search_query": q,
                "latency": f"{elapsed_ms:.1f}ms",
                "response_payload": {
                    "products": len(result.product_suggestions),
                    "categories": len(result.category_suggestions),
                    "api_latency_ms": round(elapsed_ms, 1),
                },
            },
        )
        return result
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
