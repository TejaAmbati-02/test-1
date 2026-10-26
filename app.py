from __future__ import annotations
 
from contextlib import asynccontextmanager
from typing import AsyncGenerator
 
import asyncio
import os
import time
 
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
 
import uuid
 
from config import settings
from utils.logger import logger, setup_logging, set_log_context, clear_log_context
from models.response import HealthResponse, LatencyResponse
from routers.search import index_router, v1_router
from services.embedder import embedding_service
from services.embedding_store import embedding_store
from services.opensearch import os_client
from services.search import search_service
 
setup_logging(level=settings.log_level, to_file=settings.log_to_file)
 
# How often the background heartbeat pings OpenSearch to keep the pooled TCP/TLS
# connection warm. Must stay below the server/aiohttp idle-reap window (~15s) so the
# socket is never torn down — otherwise the next request pays a full handshake (~1s).
OPENSEARCH_HEARTBEAT_INTERVAL = 5
 
# How often the background heartbeat sends a tiny embed to OCI GenAI to keep the
# on-demand model warm. The first embed after the model goes idle pays a cold
# spin-up (~1.5s observed); a periodic warmup keeps real searches off that path.
# Override with OCI_EMBED_HEARTBEAT_INTERVAL (seconds). Each tick is a billable
# on-demand call, so keep it only as frequent as needed to beat the idle window.
OCI_EMBED_HEARTBEAT_INTERVAL = int(os.environ.get("OCI_EMBED_HEARTBEAT_INTERVAL", "60"))
 
# Throwaway query used for both the startup warmup and the keep-alive heartbeat.
_EMBED_WARMUP_TEXT = "warmup"
 
 
async def _opensearch_heartbeat(interval: int) -> None:
    """Periodically ping OpenSearch so the keep-alive connection never goes idle-cold."""
    logger.info(f"OpenSearch heartbeat started — every {interval}s")
    while True:
        try:
            await asyncio.sleep(interval)
            await os_client.ping()
        except asyncio.CancelledError:
            logger.info("OpenSearch heartbeat cancelled")
            raise
        except Exception as exc:
            # Never let a transient ping failure kill the heartbeat loop.
            logger.warning(f"OpenSearch heartbeat ping failed: {exc}")
 
 
async def _oci_embed_heartbeat(interval: int) -> None:
    """Periodically embed a throwaway query so the on-demand OCI model never goes idle-cold."""
    logger.info(f"OCI embed heartbeat started — every {interval}s")
    while True:
        try:
            await asyncio.sleep(interval)
            await embedding_service.encode_query(_EMBED_WARMUP_TEXT)
        except asyncio.CancelledError:
            logger.info("OCI embed heartbeat cancelled")
            raise
        except Exception as exc:
            # Never let a transient embed failure kill the heartbeat loop.
            logger.warning(f"OCI embed heartbeat failed: {exc}")
 
 
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Application startup")
    heartbeat_task: asyncio.Task | None = None
    embed_heartbeat_task: asyncio.Task | None = None
    try:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        await embedding_service.load()
        logger.info("Embedding model ready")
 
        # Warm the on-demand OCI model now so the first real search doesn't pay the
        # cold spin-up (~1.5s). A failure here must not block startup — searches can
        # still run, they'll just warm it lazily.
        logger.info("Warming up embedding model with a throwaway query")
        try:
            await embedding_service.encode_query(_EMBED_WARMUP_TEXT)
            logger.info("Embedding model warmed up")
        except Exception as exc:
            logger.warning(f"Embedding warmup failed (continuing): {exc}")
 
        logger.info(f"Connecting to OpenSearch at {settings.opensearch_host}:{settings.opensearch_port}")
        connected = await os_client.ping()
        if not connected:
            logger.warning("OpenSearch ping returned False — searches may fail")
        else:
            logger.info("OpenSearch connected")
 
        if embedding_store.enabled:
            logger.info(f"Connecting to Redis at {settings.redis_host}:{settings.redis_port}")
            if await embedding_store.ping():
                logger.info("Redis connected — embedding cache enabled")
            else:
                logger.warning("Redis unreachable — searches will embed without the cache")
        else:
            logger.info("Redis disabled (REDIS_ENABLED=false or REDIS_HOST unset) — embedding cache off")

        logger.info("Setting up hybrid search pipeline (RRF)")
        await search_service.setup_pipeline()
        logger.info("Search pipeline ready")
 
        # Keep the warmed-up connection alive so no request ever hits a cold handshake.
        heartbeat_task = asyncio.create_task(_opensearch_heartbeat(OPENSEARCH_HEARTBEAT_INTERVAL))
        # Keep the on-demand embedding model warm so no request ever hits a cold spin-up.
        embed_heartbeat_task = asyncio.create_task(_oci_embed_heartbeat(OCI_EMBED_HEARTBEAT_INTERVAL))
 
    except Exception as exc:
        logger.exception(f"Startup failed: {exc}")
        raise
 
    logger.info("Startup complete — API is ready")
    yield
 
    logger.info("Shutting down")
    for task in (heartbeat_task, embed_heartbeat_task):
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    await os_client.close()
    await embedding_store.close()
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
        # Single OpenSearch round-trip: cluster_name() already confirms connectivity.
        cluster = await os_client.cluster_name()
        connected = cluster is not None
        model_loaded = embedding_service.is_loaded
        redis_connected = await embedding_store.ping() if embedding_store.enabled else None
        status = "ok" if connected and model_loaded and redis_connected is not False else "degraded"
        logger.info(f"Health: status={status} opensearch={connected} model={model_loaded} redis={redis_connected}")
        return HealthResponse(
            status=status,
            opensearch_connected=connected,
            opensearch_cluster=cluster,
            embedding_model_loaded=model_loaded,
            embedding_model_name=embedding_service.model_name,
            redis_connected=redis_connected,
        )
    except Exception as exc:
        logger.exception(f"Health check failed: {exc}")
        raise
 
 
async def _measure_latency(call, samples: int) -> tuple[list[float], str | None]:
    """Run an async no-arg call `samples` times, returning per-call ms and any error."""
    times: list[float] = []
    error: str | None = None
    for i in range(samples):
        t0 = time.perf_counter()
        try:
            await call()
            times.append(round((time.perf_counter() - t0) * 1000, 1))
        except Exception as exc:
            error = f"sample {i + 1} failed: {exc}"
            break
    return times, error
 
 
def _latency_response(target: str, endpoint: str, samples: int, times: list[float], error: str | None) -> LatencyResponse:
    return LatencyResponse(
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
 
 
@app.get("/health/latency/opensearch", response_model=LatencyResponse, tags=["Health"])
async def latency_opensearch(samples: int = 5) -> LatencyResponse:
    """Measure round-trip latency to OpenSearch (one GET / per sample)."""
    samples = max(1, min(samples, 20))
    logger.info(f"GET /health/latency/opensearch — {samples} samples")
 
    async def _call() -> None:
        client = await os_client.get_client()
        await client.info()
 
    times, error = await _measure_latency(_call, samples)
    endpoint = f"{settings.opensearch_host}:{settings.opensearch_port}"
    result = _latency_response("opensearch", endpoint, samples, times, error)
    logger.info(f"OpenSearch latency — min={result.min_ms}ms avg={result.avg_ms}ms max={result.max_ms}ms ok={result.ok}")
    return result
 
 
@app.get("/health/latency/redis", response_model=LatencyResponse, tags=["Health"])
async def latency_redis(samples: int = 5) -> LatencyResponse:
    """Measure round-trip latency to Redis (one PING per sample)."""
    samples = max(1, min(samples, 20))
    logger.info(f"GET /health/latency/redis — {samples} samples")
    endpoint = f"{settings.redis_host}:{settings.redis_port}"
    if not embedding_store.enabled:
        return _latency_response("redis", endpoint, samples, [], "Redis is disabled (REDIS_ENABLED=false or REDIS_HOST unset)")

    async def _call() -> None:
        await embedding_store._redis().ping()

    times, error = await _measure_latency(_call, samples)
    result = _latency_response("redis", endpoint, samples, times, error)
    logger.info(f"Redis latency — min={result.min_ms}ms avg={result.avg_ms}ms max={result.max_ms}ms ok={result.ok}")
    return result


@app.get("/health/latency/oci", response_model=LatencyResponse, tags=["Health"])
async def latency_oci(samples: int = 5) -> LatencyResponse:
    """Measure round-trip latency to OCI GenAI (one embed call per sample)."""
    samples = max(1, min(samples, 20))
    logger.info(f"GET /health/latency/oci — {samples} samples")
 
    async def _call() -> None:
        await embedding_service.encode_query("latency check")
 
    times, error = await _measure_latency(_call, samples)
    result = _latency_response("oci", settings.oci_genai_endpoint, samples, times, error)
    logger.info(f"OCI latency — min={result.min_ms}ms avg={result.avg_ms}ms max={result.max_ms}ms ok={result.ok}")
    return result
 
 
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": "HTTP_ERROR", "message": str(detail)},
    )
 
 
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # Missing/malformed form or query fields map to the contract's 400 Bad Request (§5).
    errors = exc.errors()
    logger.warning(f"Request validation failed: {errors}")
    first = errors[0] if errors else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    msg = first.get("msg", "Request validation failed")
    message = f"{loc}: {msg}" if loc else msg
    return JSONResponse(status_code=400, content={"code": "VALIDATION_ERROR", "message": message})
 
 
@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"code": "INTERNAL_ERROR", "message": str(exc)})
 
 
if __name__ == "__main__":
    # Auto-reload is OFF by default. Enable only for local dev with RELOAD=true, and
    # never in a OneDrive/synced folder — background file syncing triggers endless reloads.
    reload_enabled = os.environ.get("RELOAD", "false").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_enabled,
        # If reload is on, ignore log writes under ./logs (they'd otherwise trigger restarts).
        reload_excludes=["logs/*", "*.log", "*.log.*"] if reload_enabled else None,
    )