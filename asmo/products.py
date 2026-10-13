import aiohttp
from dataclasses import dataclass, field
from .auth import AuthClient, requires_token
from .config import Config
from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


@dataclass
class Product:
    """Lightweight representation of a CommerceTools product projection."""

    id: str
    name: str
    raw: dict = field(repr=False)

    @classmethod
    def from_raw(cls, raw: dict) -> "Product":
        try:
            name = raw.get("name", {})
            display_name = name.get("en") or name.get("de") or "N/A"
            logger.info(
                f"Parsed product id={raw.get('id')!r} name={display_name!r}",
                extra={
                    "function_name": "Product.from_raw",
                    "input_data": {"id": raw.get("id"), "name_field": name},
                    "response_payload": {"id": raw["id"], "name": display_name},
                },
            )
            return cls(id=raw["id"], name=display_name, raw=raw)
        except Exception as exc:
            logger.error(
                f"Failed to parse product payload: {exc}",
                extra={
                    "function_name": "Product.from_raw",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"id": raw.get("id")},
                },
            )
            raise


class ProductsClient:
    """Fetches product projections from CommerceTools using cursor-based pagination."""

    PAGE_SIZE = 500

    def __init__(self, config: Config, auth: AuthClient) -> None:
        self._config = config
        self._auth = auth

    @log_function
    @requires_token
    async def get_all(self) -> list[Product]:
        """Return every product projection using cursor-based pagination (id > last_id)."""
        results: list[dict] = []
        last_id: str | None = None
        page_num = 0

        logger.info("Starting product fetch", extra={"function_name": "get_all"})
        async with aiohttp.ClientSession() as session:
            while True:
                page = await self._fetch_page(session, last_id=last_id)
                batch: list[dict] = page["results"]
                page_num += 1
                if not batch:
                    break
                results.extend(batch)
                logger.info(
                    f"Products page {page_num}: {len(batch)} records (total so far: {len(results)})",
                    extra={"function_name": "get_all", "input_data": {"page": page_num, "batch_size": len(batch)}},
                )
                if len(batch) < self.PAGE_SIZE:
                    break
                last_id = batch[-1]["id"]

        logger.info(
            f"Fetched {len(results)} products in {page_num} page(s)",
            extra={"function_name": "get_all", "status": "success", "input_data": {"total": len(results)}},
        )
        return [Product.from_raw(r) for r in results]

    async def _fetch_page(self, session: aiohttp.ClientSession, last_id: str | None) -> dict:
        from datetime import datetime
        params = {
            "limit": self.PAGE_SIZE,
            "sort": "id asc",
            "expand": "categories[*].ancestors[*]",
        }
        if last_id:
            params["where"] = f'id > "{last_id}"'
        url = f"{self._config.api_base}/product-projections"
        logger.info(
            f"GET {url} last_id={last_id!r} params={params}",
            extra={"function_name": "_fetch_page", "input_data": {"url": url, "params": params, "last_id": last_id, "limit": self.PAGE_SIZE}},
        )
        t0 = datetime.now()
        try:
            async with session.get(
                url,
                headers=self._auth.headers,
                params=params,
                ssl=SSL_CONTEXT,
            ) as resp:
                full_url = str(resp.url)
                status = resp.status
                body = await resp.json()
                elapsed = (datetime.now() - t0).total_seconds()
                count = len(body.get("results", []))
                logger.info(
                    f"HTTP {status} ← {full_url} → {count} records (total={body.get('total')}) in {elapsed:.3f}s",
                    extra={
                        "function_name": "_fetch_page",
                        "latency": f"{elapsed:.3f}s",
                        "status": "success" if resp.ok else "error",
                        "input_data": {"full_url": full_url, "http_status": status, "returned": count, "total": body.get("total")},
                        "response_payload": body,
                    },
                )
                resp.raise_for_status()
                return body
        except Exception as exc:
            elapsed = (datetime.now() - t0).total_seconds()
            logger.error(
                f"Request failed after {elapsed:.3f}s: GET {url} last_id={last_id!r}: {exc}",
                extra={
                    "function_name": "_fetch_page",
                    "latency": f"{elapsed:.3f}s",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"url": url, "params": params, "last_id": last_id},
                },
            )
            raise
