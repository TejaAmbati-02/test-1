"""Thin async wrapper around the OpenSearch client (one shared connection pool).

Use the ready-made object:

    from services.opensearch import os_client
    result = await os_client.search(index, body, pipeline_id)

Each method is self-contained (it calls no other method).
"""
import asyncio
import json
import re
import time
from typing import Any

from opensearchpy import AsyncOpenSearch, NotFoundError, RequestError

from config import Settings, settings
from models.response import ApiError
from utils.logger import logger


class OpenSearchClientService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # The shared client. Creating it opens no connection; the pool connects on the first request.
        self.client = AsyncOpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=(settings.opensearch_user, settings.opensearch_password),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            http_compress=True,
            timeout=settings.opensearch_timeout,
        )

    async def close(self) -> None:
        """Close the connection pool (on app shutdown); never raises."""
        try:
            await self.client.close()
            logger.info("OpenSearch — connection closed", extra={"status": "success"})
        except Exception as exc:
            logger.error("OpenSearch — error while closing", extra={"status": "error", "error_details": str(exc)})

    async def ping(self) -> bool:
        """True when the cluster answers; never raises."""
        try:
            ok = await self.client.ping()
            logger.info(f"OpenSearch ping — {'OK' if ok else 'FAILED'}",
                        extra={"status": "success" if ok else "error", "response_payload": {"reachable": ok}})
            return ok
        except Exception as exc:
            logger.error("OpenSearch ping — failed", extra={"status": "error", "error_details": str(exc)})
            return False

    async def keep_warm(self, interval: int) -> None:
        """Ping OpenSearch every ``interval`` seconds so the pooled TCP/TLS connection stays warm.

        Runs until cancelled (app shutdown). A failed ping is logged, never fatal.
        """
        logger.info(f"OpenSearch heartbeat started — every {interval}s")
        try:
            while True:
                await asyncio.sleep(interval)
                start = time.perf_counter()
                try:
                    await self.client.ping()
                    logger.debug(f"OpenSearch heartbeat ok in {(time.perf_counter() - start) * 1000:.1f}ms")
                except Exception as exc:
                    logger.warning(f"OpenSearch heartbeat failed ({type(exc).__name__}): {exc}",
                                   extra={"status": "error", "error_details": str(exc)})
        except asyncio.CancelledError:
            logger.info("OpenSearch heartbeat cancelled")
            raise

    async def cluster_name(self) -> str | None:
        """The cluster's name, or None when it can't be reached; never raises."""
        try:
            info = await self.client.info()
            return info.get("cluster_name")
        except Exception as exc:
            logger.error("OpenSearch cluster_name — failed", extra={"status": "error", "error_details": str(exc)})
            return None

    async def list_indices(self) -> list[dict[str, Any]]:
        """Every non-system index with its document count, size and status."""
        stats = await self.client.cat.indices(format="json", bytes="b")
        indices = [
            {
                "name": idx["index"],
                "doc_count": int(idx.get("docs.count") or 0),
                "size_bytes": int(idx.get("store.size") or 0),
                "status": idx.get("status", "unknown"),
            }
            for idx in stats
            if not idx["index"].startswith(".")
        ]
        logger.info(f"OpenSearch — listed {len(indices)} indices", extra={"status": "success"})
        return indices

    async def pipeline_exists(self, pipeline_id: str) -> bool:
        """True when the search pipeline exists; never raises."""
        try:
            await self.client.transport.perform_request("GET", f"/_search/pipeline/{pipeline_id}")
            return True
        except Exception as exc:
            logger.info(f"OpenSearch — pipeline '{pipeline_id}' not found ({type(exc).__name__}: {exc})")
            return False

    async def create_search_pipeline(
        self, pipeline_id: str, rank_constant: int = 60, weights: list[float] | None = None,
    ) -> dict[str, Any]:
        """Create (or overwrite) an RRF search pipeline. ``weights`` = one per hybrid sub-query, in order."""
        combination: dict[str, Any] = {"technique": "rrf", "rank_constant": rank_constant}
        if weights:
            combination["parameters"] = {"weights": weights}
        body = {
            "description": "RRF hybrid search pipeline" + (" (weighted)" if weights else ""),
            "phase_results_processors": [{"score-ranker-processor": {"combination": combination}}],
        }
        response = await self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=body)
        logger.info(f"OpenSearch — created pipeline '{pipeline_id}'", extra={"status": "success"})
        return response

    async def search(self, index: str, body: dict[str, Any], pipeline_id: str | None = None) -> dict[str, Any]:
        """Run a search. Raises ApiError 404 INDEX_NOT_FOUND (unknown index) or 400 QUERY_ERROR (bad query)."""
        # 1. Log the full query sent to OpenSearch, with embedding vectors (lists of more than 16
        #    numbers) shortened to "<vector dim=N>" so the log stays readable.
        logged_body = re.sub(
            r"\[(?:\s*-?[\d.eE+-]+\s*,){16,}\s*-?[\d.eE+-]+\s*\]",
            lambda match: f'"<vector dim={match.group(0).count(",") + 1}>"',
            json.dumps(body, default=str),
        )
        logger.info(f"OpenSearch — search on '{index}' (pipeline={pipeline_id})",
                    extra={"input_data": {"body": logged_body}})

        # 2. Search.
        params = {"search_pipeline": pipeline_id} if pipeline_id else {}
        start = time.perf_counter()
        try:
            result = await self.client.search(index=index, body=body, params=params)
        except NotFoundError as exc:
            logger.error(f"OpenSearch — index '{index}' not found", extra={"status": "error", "error_details": str(exc)})
            raise ApiError(404, "INDEX_NOT_FOUND", f"Index '{index}' not found") from exc
        except RequestError as exc:
            logger.error(f"OpenSearch — query error on '{index}'", extra={"status": "error", "error_details": str(exc.info)})
            raise ApiError(400, "QUERY_ERROR", f"OpenSearch query error: {exc.info}") from exc

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            f"OpenSearch search on '{index}' — {result['hits']['total']['value']} hits in {elapsed_ms}ms",
            extra={"status": "success", "response_payload": {
                "opensearch_elapsed_ms": elapsed_ms,
                "server_took_ms": result.get("took", 0),
                "returned": len(result["hits"]["hits"]),
                "pipeline": pipeline_id,
            }},
        )
        return result


# The one object the app uses — import it where needed:  from services.opensearch import os_client
os_client = OpenSearchClientService(settings)
