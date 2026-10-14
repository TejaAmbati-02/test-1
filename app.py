from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

import uuid

from config import settings
from utils.logger import logger, setup_logging, set_log_context, clear_log_context
from models.response import ErrorDetail, ErrorResponse, HealthResponse
from routers.search import index_router, v1_router
from services.embedder import embedding_service
from services.opensearch import os_client
from services.reranker import reranker_service
from services.search import search_service

setup_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application startup")
    try:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        await embedding_service.load()
        logger.info("Embedding model ready")

        logger.info(f"Loading reranker model: {settings.reranker_model_name}")
        await reranker_service.load()
        logger.info("Reranker model ready")

        logger.info(f"Connecting to OpenSearch at {settings.opensearch_host}:{settings.opensearch_port}")
        connected = await os_client.ping()
        if not connected:
            logger.warning("OpenSearch ping returned False — searches may fail")
        else:
            logger.info("OpenSearch connected")

        logger.info("Setting up hybrid search pipeline (RRF)")
        await search_service.setup_pipeline()
        logger.info("Search pipeline ready")

    except Exception as exc:
        logger.exception(f"Startup failed: {exc}")
        raise

    logger.info("Startup complete — API is ready")
    yield

    logger.info("Shutting down")
    await os_client.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        set_log_context(request_id)

        client_ip = request.client.host if request.client else "unknown"
        query_string = str(request.url.query) if request.url.query else ""
        user_agent = request.headers.get("user-agent", "")
        content_type = request.headers.get("content-type", "")

        start = time.perf_counter()
        logger.info(
            f"REQUEST {request.method} {request.url.path}",
            extra={
                "function_name": f"{request.method} {request.url.path}",
                "input_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "query": query_string,
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "content_type": content_type,
                },
                "status": "received",
            },
        )

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            status = "success" if response.status_code < 400 else "error"
            logger.info(
                f"RESPONSE {request.method} {request.url.path} — {response.status_code} in {elapsed_ms:.1f}ms",
                extra={
                    "function_name": f"{request.method} {request.url.path}",
                    "status": status,
                    "latency": f"{elapsed_ms:.1f}ms",
                    "response_payload": {
                        "http_status": response.status_code,
                        "latency_ms": round(elapsed_ms, 1),
                    },
                },
            )
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"REQUEST {request.method} {request.url.path} — unhandled exception after {elapsed_ms:.1f}ms: {exc}",
                extra={
                    "function_name": f"{request.method} {request.url.path}",
                    "status": "error",
                    "latency": f"{elapsed_ms:.1f}ms",
                    "error_details": str(exc),
                },
            )
            raise
        finally:
            clear_log_context()


app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(index_router)
app.include_router(v1_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    logger.info("GET /health")
    try:
        connected = await os_client.ping()
        cluster = await os_client.cluster_name() if connected else None
        model_loaded = embedding_service.is_loaded
        status = "ok" if connected and model_loaded else "degraded"
        logger.info(f"Health: status={status} opensearch={connected} model={model_loaded}")
        return HealthResponse(
            status=status,
            opensearch_connected=connected,
            opensearch_cluster=cluster,
            embedding_model_loaded=model_loaded,
            embedding_model_name=embedding_service.model_name,
        )
    except Exception as exc:
        logger.exception(f"Health check failed: {exc}")
        raise


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=ErrorDetail(code="HTTP_ERROR", message=str(detail))).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning(f"Request validation failed: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request validation failed",
                detail=jsonable_encoder(exc.errors()),
            )
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=ErrorDetail(code="INTERNAL_ERROR", message=str(exc))).model_dump(),
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        # Don't restart on log writes — the app writes logs into ./logs inside the watched tree.
        reload_excludes=["logs/*", "*.log", "*.log.*"],
    )
