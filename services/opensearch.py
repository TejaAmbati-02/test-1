"""Thin async wrapper around the OpenSearch client (one shared connection pool)."""
from __future__ import annotations

import time
from typing import Any

from opensearchpy import AsyncOpenSearch, NotFoundError, RequestError

from config import Settings, settings as _settings
from utils.logger import logger


class IndexNotFoundError(Exception):
    """A search targeted an index that does not exist (the API returns 404)."""


class OpenSearchQueryError(Exception):
    """OpenSearch rejected the query as malformed (the API returns 400)."""


async def _redact_vectors(obj: Any) -> Any:
    """Copy of a query body for logging, with embedding vectors (long float lists) replaced by a short summary."""
    if isinstance(obj, dict):
        return {k: await _redact_vectors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 16 and all(isinstance(v, (int, float)) for v in obj):
            return f"<vector dim={len(obj)}>"
        return [await _redact_vectors(v) for v in obj]
    return obj


class OpenSearchClientService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenSearch | None = None

    async def get_client(self) -> AsyncOpenSearch:
        """The shared client, created on first use."""
        if self._client is None:
            cfg = self._settings
            logger.debug(f"OpenSearch — creating client for {cfg.opensearch_host}:{cfg.opensearch_port}")
            self._client = AsyncOpenSearch(
                hosts=[{"host": cfg.opensearch_host, "port": cfg.opensearch_port}],
                http_auth=(cfg.opensearch_user, cfg.opensearch_password),
                use_ssl=True,
                verify_certs=False,
                ssl_show_warn=False,
                http_compress=True,
                timeout=cfg.opensearch_timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.close()
            self._client = None
            logger.info("OpenSearch — connection closed", extra={"status": "success"})
        except Exception as exc:
            logger.error("OpenSearch — error while closing", extra={"status": "error", "error_details": str(exc)})

    async def ping(self) -> bool:
        """True when the cluster answers; never raises."""
        try:
            ok = await (await self.get_client()).ping()
            logger.info(f"OpenSearch ping — {'OK' if ok else 'FAILED'}",
                        extra={"status": "success" if ok else "error", "response_payload": {"reachable": ok}})
            return ok
        except Exception as exc:
            logger.error("OpenSearch ping — failed", extra={"status": "error", "error_details": str(exc)})
            return False

    async def cluster_name(self) -> str | None:
        """The cluster's name, or None when it can't be reached; never raises."""
        try:
            info = await (await self.get_client()).info()
            return info.get("cluster_name")
        except Exception as exc:
            logger.error("OpenSearch cluster_name — failed", extra={"status": "error", "error_details": str(exc)})
            return None

    async def list_indices(self) -> list[dict[str, Any]]:
        """Every non-system index with its document count, size and status."""
        stats = await (await self.get_client()).cat.indices(format="json", bytes="b")
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
        try:
            await (await self.get_client()).transport.perform_request("GET", f"/_search/pipeline/{pipeline_id}")
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
        client = await self.get_client()
        response = await client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=body)
        logger.info(f"OpenSearch — created pipeline '{pipeline_id}'", extra={"status": "success"})
        return response

    async def search(self, index: str, body: dict[str, Any], pipeline_id: str | None = None) -> dict[str, Any]:
        """Run a search. Raises IndexNotFoundError (unknown index) or OpenSearchQueryError (bad query)."""
        # The full query sent to OpenSearch (embedding vectors shortened to "<vector dim=N>").
        logger.info(f"OpenSearch — search on '{index}' (pipeline={pipeline_id})",
                    extra={"input_data": {"body": await _redact_vectors(body)}})
        client = await self.get_client()
        params = {"search_pipeline": pipeline_id} if pipeline_id else {}
        start = time.perf_counter()
        try:
            result = await client.search(index=index, body=body, params=params)
        except NotFoundError as exc:
            logger.error(f"OpenSearch — index '{index}' not found", extra={"status": "error", "error_details": str(exc)})
            raise IndexNotFoundError(f"Index '{index}' not found") from exc
        except RequestError as exc:
            logger.error(f"OpenSearch — query error on '{index}'", extra={"status": "error", "error_details": str(exc.info)})
            raise OpenSearchQueryError(f"OpenSearch query error: {exc.info}") from exc

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


os_client = OpenSearchClientService(_settings)
