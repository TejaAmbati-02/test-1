"""ASMO AI Search API — FastAPI app: startup/shutdown, middleware, health checks and error handlers."""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Awaitable, Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import settings
from utils.logger import logger, request_id_var, setup_logging

# Set up logging (incl. the log file) BEFORE importing the services below: they log while
# being imported, and those lines would otherwise reach the console but not the file.
setup_logging(level=settings.log_level, to_file=settings.log_to_file)

from models.response import HealthResponse, LatencyResponse  # noqa: E402
from routers.search import index_router, v1_router  # noqa: E402
from services.embedder import embedding_service  # noqa: E402
from services.opensearch import os_client  # noqa: E402
from services.redis_client import redis_client  # noqa: E402
from services.search import search_service  # noqa: E402
from utils.keycloak_auth import keycloak_auth  # noqa: E402

# How often to ping OpenSearch so the pooled TCP/TLS connection stays warm. Must stay below
# the server/aiohttp idle-reap window (~15s), otherwise the next request pays a full handshake (~1s).
OPENSEARCH_HEARTBEAT_INTERVAL = 5

# How often to send a tiny embed to OCI GenAI so the on-demand model stays warm (a cold
# model costs ~1.5s on the first embed). Each tick is a billable call, so keep it only as
# frequent as needed. Set OCI_EMBED_HEARTBEAT_INTERVAL (seconds) in .env.
OCI_EMBED_HEARTBEAT_INTERVAL = settings.oci_embed_heartbeat_interval

# Throwaway query used for the startup warmup and the keep-alive heartbeat.
_EMBED_WARMUP_TEXT = "warmup"


async def _heartbeat(name: str, interval: int, call: Callable[..., Awaitable[object]], *args: object) -> None:
    """Run ``await call(*args)`` every ``interval`` seconds until cancelled. A failed call is logged, never fatal."""
    logger.info(f"{name} heartbeat started — every {interval}s")
    while True:
        try:
            await asyncio.sleep(interval)
            start = time.perf_counter()
            await call(*args)
            logger.debug(f"{name} heartbeat ok in {(time.perf_counter() - start) * 1000:.1f}ms")
        except asyncio.CancelledError:
            logger.info(f"{name} heartbeat cancelled")
            raise
        except Exception as exc:
            logger.warning(f"{name} heartbeat failed ({type(exc).__name__}): {exc}",
                           extra={"status": "error", "error_details": str(exc)})


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application startup")
    try:
        await embedding_service.load()

        # Warm the on-demand OCI model so the first real search doesn't pay the cold start.
        # Not fatal: searches still work, they just warm it lazily.
        try:
            await embedding_service.encode_query(_EMBED_WARMUP_TEXT)
        except Exception as exc:
            logger.warning(f"Embedding warmup failed (continuing) ({type(exc).__name__}): {exc}",
                           extra={"status": "error", "error_details": str(exc)})

        logger.info(f"Connecting to OpenSearch at {settings.opensearch_host}:{settings.opensearch_port}")
        if not await os_client.ping():
            logger.warning("OpenSearch ping returned False — searches may fail")

        if redis_client.enabled:
            await redis_client.connect()  # one shared client, reused by every request
            if not await redis_client.ping():
                logger.warning("Redis unreachable — searches still run, but their request_id can't be reused")
        else:
            logger.info("Redis disabled (REDIS_ENABLED=false or REDIS_HOST unset) — request_id refinement off")

        await search_service.setup_pipeline()
    except Exception as exc:
        logger.exception(f"Startup failed ({type(exc).__name__}): {exc}",
                         extra={"status": "error", "error_details": str(exc)})
        raise

    # Keep the warm connection / warm model alive so no request hits a cold start.
    heartbeats = [
        asyncio.create_task(_heartbeat("OpenSearch", OPENSEARCH_HEARTBEAT_INTERVAL, os_client.ping)),
        asyncio.create_task(_heartbeat("OCI embed", OCI_EMBED_HEARTBEAT_INTERVAL,
                                       embedding_service.encode_query, _EMBED_WARMUP_TEXT)),
    ]
    logger.info("Startup complete — API is ready")
    yield

    logger.info("Shutting down")
    for task in heartbeats:
        task.cancel()
    await asyncio.gather(*heartbeats, return_exceptions=True)
    await os_client.close()
    await redis_client.close()
    await keycloak_auth.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Give every request an id (put on all its log lines) and log the request and response."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    where = f"{request.method} {request.url.path}"
    logger.info(f"REQUEST {where}", extra={
        "function_name": where,
        "status": "received",
        "input_data": {
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", ""),
            "content_type": request.headers.get("content-type", ""),
        },
    })
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"RESPONSE {where} — {response.status_code} in {elapsed_ms:.1f}ms", extra={
            "function_name": where,
            "status": "success" if response.status_code < 400 else "error",
            "latency": f"{elapsed_ms:.1f}ms",
            "response_payload": {"http_status": response.status_code, "latency_ms": round(elapsed_ms, 1)},
        })
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(f"REQUEST {where} — unhandled {type(exc).__name__} after {elapsed_ms:.1f}ms: {exc}",
                         extra={"function_name": where, "status": "error", "error_details": str(exc)})
        raise
    finally:
        request_id_var.reset(token)


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(index_router)
app.include_router(v1_router)


# Health checks

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    cluster = await os_client.cluster_name()  # one round-trip: a name back means connected
    redis_connected = await redis_client.ping() if redis_client.enabled else None
    healthy = cluster is not None and embedding_service.is_loaded and redis_connected is not False
    return HealthResponse(
        status="ok" if healthy else "degraded",
        opensearch_connected=cluster is not None,
        opensearch_cluster=cluster,
        embedding_model_loaded=embedding_service.is_loaded,
        embedding_model_name=embedding_service.model_name,
        redis_connected=redis_connected,
    )


async def _measure_latency(target: str, endpoint: str, samples: int,
                           call: Callable[[], Awaitable[object]]) -> LatencyResponse:
    """Await ``call`` ``samples`` times (clamped to 1..20) and report the round-trip times. Stops at the first failure."""
    samples = max(1, min(samples, 20))
    times: list[float] = []
    error: str | None = None
    for i in range(samples):
        start = time.perf_counter()
        try:
            await call()
        except Exception as exc:
            error = f"sample {i + 1} failed: {exc}"
            logger.warning(f"{target} latency {error}", extra={"status": "error", "error_details": str(exc)})
            break
        times.append(round((time.perf_counter() - start) * 1000, 1))

    result = LatencyResponse(
        target=target,
        endpoint=endpoint,
        samples=samples,
        samples_ms=times,
        min_ms=min(times) if times else 0.0,
        avg_ms=round(sum(times) / len(times), 1) if times else 0.0,
        max_ms=max(times) if times else 0.0,
        ok=error is None and len(times) == samples,
        error=error,
    )
    logger.info(f"{target} latency — min={result.min_ms}ms avg={result.avg_ms}ms max={result.max_ms}ms ok={result.ok}")
    return result


# One sample of each latency check.
async def _opensearch_info() -> None:
    await (await os_client.get_client()).info()


async def _redis_ping() -> None:
    await redis_client.client.ping()


async def _oci_embed() -> None:
    await embedding_service.encode_query("latency check")


@app.get("/health/latency/opensearch", response_model=LatencyResponse, tags=["Health"])
async def latency_opensearch(samples: int = 5) -> LatencyResponse:
    """Measure round-trip latency to OpenSearch (one GET / per sample)."""
    return await _measure_latency("opensearch", f"{settings.opensearch_host}:{settings.opensearch_port}",
                                  samples, _opensearch_info)


@app.get("/health/latency/redis", response_model=LatencyResponse, tags=["Health"])
async def latency_redis(samples: int = 5) -> LatencyResponse:
    """Measure round-trip latency to Redis (one PING per sample)."""
    endpoint = redis_client.endpoint
    if not redis_client.enabled:
        return LatencyResponse(
            target="redis", endpoint=endpoint, samples=max(1, min(samples, 20)), samples_ms=[],
            min_ms=0.0, avg_ms=0.0, max_ms=0.0, ok=False,
            error="Redis is disabled (REDIS_ENABLED=false or REDIS_HOST unset)",
        )
    return await _measure_latency("redis", endpoint, samples, _redis_ping)


@app.get("/health/latency/oci", response_model=LatencyResponse, tags=["Health"])
async def latency_oci(samples: int = 5) -> LatencyResponse:
    """Measure round-trip latency to OCI GenAI (one embed call per sample)."""
    return await _measure_latency("oci", settings.oci_genai_endpoint, samples, _oci_embed)


# Error handlers — every error leaves the API as {"code", "message"} (5.1)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    log = logger.error if exc.status_code >= 500 else logger.warning
    log(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}",
        extra={"status": "error", "error_details": str(exc.detail)})
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(status_code=exc.status_code, content={"code": "HTTP_ERROR", "message": str(detail)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Missing/malformed form or query fields map to the contract's 400 Bad Request.
    errors = exc.errors()
    logger.warning(f"Request validation failed on {request.method} {request.url.path}: {errors}",
                   extra={"status": "error", "error_details": str(errors)})
    first = errors[0] if errors else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    msg = first.get("msg", "Request validation failed")
    return JSONResponse(status_code=400, content={"code": "VALIDATION_ERROR", "message": f"{loc}: {msg}" if loc else msg})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled {type(exc).__name__} on {request.method} {request.url.path}: {exc}",
                     extra={"status": "error", "error_details": str(exc)})
    return JSONResponse(status_code=500, content={"code": "INTERNAL_ERROR", "message": str(exc)})


if __name__ == "__main__":
    # Auto-reload is OFF by default. Enable it only for local dev with RELOAD=true, and never in a
    # OneDrive/synced folder — background file syncing triggers endless reloads.
    reload_enabled = settings.reload
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_enabled,
        # With reload on, ignore log writes under ./logs (they'd otherwise trigger restarts).
        reload_excludes=["logs/*", "*.log", "*.log.*"] if reload_enabled else None,
    )
