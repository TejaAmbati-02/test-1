from __future__ import annotations

from typing import Any

from opensearchpy import AsyncOpenSearch, NotFoundError, RequestError

from config import Settings, settings as _settings
from utils.logger import logger


class IndexNotFoundError(Exception):
    """Raised when a search targets an index that does not exist (maps to HTTP 404)."""


class OpenSearchQueryError(Exception):
    """Raised when OpenSearch rejects a malformed query (maps to HTTP 400)."""


class OpenSearchClientService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenSearch | None = None

    async def get_client(self) -> AsyncOpenSearch:
        if self._client is None:
            cfg = self._settings
            logger.debug(
                "OpenSearchClientService.get_client — creating client",
                extra={"input_data": {"host": cfg.opensearch_host, "port": cfg.opensearch_port}},
            )
            try:
                self._client = AsyncOpenSearch(
                    hosts=[{"host": cfg.opensearch_host, "port": cfg.opensearch_port}],
                    http_auth=(cfg.opensearch_user, cfg.opensearch_password),
                    use_ssl=True,
                    verify_certs=False,
                    ssl_show_warn=False,
                    http_compress=True,
                    timeout=cfg.opensearch_timeout,
                )
                logger.debug("OpenSearchClientService.get_client — client created")
            except Exception as exc:
                logger.error(
                    "OpenSearchClientService.get_client — failed to create client",
                    extra={"status": "error", "error_details": str(exc)},
                )
                raise
        return self._client

    async def close(self) -> None:
        if self._client is None:
            logger.debug("OpenSearchClientService.close — client already closed")
            return
        logger.info("OpenSearchClientService.close — closing connection")
        try:
            await self._client.close()
            self._client = None
            logger.info("OpenSearchClientService.close — connection closed", extra={"status": "success"})
        except Exception as exc:
            logger.error(
                "OpenSearchClientService.close — error while closing",
                extra={"status": "error", "error_details": str(exc)},
            )

    async def ping(self) -> bool:
        logger.debug("OpenSearchClientService.ping — pinging cluster")
        try:
            client = await self.get_client()
            result = await client.ping()
            logger.info(
                f"OpenSearchClientService.ping — {'OK' if result else 'FAILED'}",
                extra={"status": "success" if result else "error", "response_payload": {"reachable": result}},
            )
            return result
        except Exception as exc:
            logger.error(
                "OpenSearchClientService.ping — exception during ping",
                extra={"status": "error", "error_details": str(exc)},
            )
            return False

    async def cluster_name(self) -> str | None:
        logger.debug("OpenSearchClientService.cluster_name — fetching cluster info")
        try:
            client = await self.get_client()
            info = await client.info()
            name = info.get("cluster_name")
            logger.debug(f"OpenSearchClientService.cluster_name — got '{name}'")
            return name
        except Exception as exc:
            logger.error(
                "OpenSearchClientService.cluster_name — failed",
                extra={"status": "error", "error_details": str(exc)},
            )
            return None

    async def list_indices(self) -> list[dict[str, Any]]:
        logger.info("OpenSearchClientService.list_indices — fetching index list")
        try:
            client = await self.get_client()
            stats = await client.cat.indices(format="json", bytes="b")
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
            logger.info(
                "OpenSearchClientService.list_indices — done",
                extra={"status": "success", "response_payload": {"count": len(indices)}},
            )
            return indices
        except Exception as exc:
            logger.error(
                "OpenSearchClientService.list_indices — failed",
                extra={"status": "error", "error_details": str(exc)},
            )
            raise

    async def index_exists(self, index: str) -> bool:
        logger.debug(f"OpenSearchClientService.index_exists — checking '{index}'")
        try:
            client = await self.get_client()
            exists = await client.indices.exists(index=index)
            logger.debug(f"OpenSearchClientService.index_exists — '{index}' exists={exists}")
            return exists
        except Exception as exc:
            logger.error(
                f"OpenSearchClientService.index_exists — failed for '{index}'",
                extra={"status": "error", "error_details": str(exc)},
            )
            return False

    async def pipeline_exists(self, pipeline_id: str) -> bool:
        logger.debug(f"OpenSearchClientService.pipeline_exists — checking '{pipeline_id}'")
        try:
            client = await self.get_client()
            await client.transport.perform_request("GET", f"/_search/pipeline/{pipeline_id}")
            logger.info(f"OpenSearchClientService.pipeline_exists — pipeline '{pipeline_id}' found")
            return True
        except Exception:
            logger.info(f"OpenSearchClientService.pipeline_exists — pipeline '{pipeline_id}' not found")
            return False

    async def create_search_pipeline(self, pipeline_id: str, rank_constant: int = 60) -> dict[str, Any]:
        logger.info(
            f"OpenSearchClientService.create_search_pipeline — creating '{pipeline_id}'",
            extra={"input_data": {"pipeline_id": pipeline_id, "rank_constant": rank_constant}},
        )
        try:
            client = await self.get_client()
            body = {
                "description": "RRF hybrid search pipeline",
                "phase_results_processors": [
                    {"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": rank_constant}}}
                ],
            }
            response = await client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=body)
            logger.info(
                f"OpenSearchClientService.create_search_pipeline — created '{pipeline_id}'",
                extra={"status": "success"},
            )
            return response
        except Exception as exc:
            logger.error(
                f"OpenSearchClientService.create_search_pipeline — failed for '{pipeline_id}'",
                extra={"status": "error", "error_details": str(exc)},
            )
            raise

    async def search(self, index: str, body: dict[str, Any], pipeline_id: str | None = None) -> dict[str, Any]:
        logger.debug(f"OpenSearchClientService.search — executing on '{index}' pipeline={pipeline_id}")
        try:
            client = await self.get_client()
            params = {"search_pipeline": pipeline_id} if pipeline_id else {}
            result = await client.search(index=index, body=body, params=params)
            total = result["hits"]["total"]["value"]
            took = result.get("took", 0)
            logger.info(
                f"OpenSearchClientService.search — done on '{index}'",
                extra={
                    "status": "success",
                    "response_payload": {"total_hits": total, "took_ms": took, "returned": len(result["hits"]["hits"]), "pipeline": pipeline_id},
                },
            )
            return result
        except NotFoundError as exc:
            logger.error(
                f"OpenSearchClientService.search — index '{index}' not found",
                extra={"status": "error", "error_details": str(exc)},
            )
            raise IndexNotFoundError(f"Index '{index}' not found") from exc
        except RequestError as exc:
            logger.error(
                f"OpenSearchClientService.search — query error on '{index}'",
                extra={"status": "error", "error_details": str(exc.info)},
            )
            raise OpenSearchQueryError(f"OpenSearch query error: {exc.info}") from exc
        except Exception as exc:
            logger.error(
                f"OpenSearchClientService.search — unexpected error on '{index}'",
                extra={"status": "error", "error_details": str(exc)},
            )
            raise


os_client = OpenSearchClientService(_settings)
