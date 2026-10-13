import asyncio
import aiohttp
from dataclasses import dataclass, field
from .auth import AuthClient, requires_token
from .config import Config
from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


@dataclass
class Price:
    """Lightweight representation of a CommerceTools standalone price."""

    id: str
    sku: str
    currency: str
    amount: float
    scope: str | None
    seller_name_en: str | None
    seller_name_de: str | None
    contract_id: str | None
    lead_time_days: int | None
    price_unit: str | None
    moq: int | None
    raw: dict = field(repr=False)

    @classmethod
    def from_raw(cls, raw: dict) -> "Price":
        try:
            return cls._from_raw(raw)
        except Exception as exc:
            logger.error(
                f"Failed to parse price payload: {exc}",
                extra={
                    "request_id": raw.get("sku"),
                    "function_name": "Price.from_raw",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"id": raw.get("id"), "sku": raw.get("sku")},
                },
            )
            raise

    @classmethod
    def _from_raw(cls, raw: dict) -> "Price":
        value = raw.get("value", {})
        custom_fields = raw.get("custom", {}).get("fields", {})
        cent_amount = value.get("centAmount", 0)
        fraction_digits = value.get("fractionDigits", 2)
        price = cls(
            id=raw["id"],
            sku=raw.get("sku", "N/A"),
            currency=value.get("currencyCode", "N/A"),
            amount=cent_amount / (10 ** fraction_digits),
            scope=custom_fields.get("scope"),
            seller_name_en=_channel_name(raw, "en"),
            seller_name_de=_channel_name(raw, "de"),
            contract_id=custom_fields.get("contractId"),
            lead_time_days=custom_fields.get("leadTimeDays"),
            price_unit=value.get("currencyCode"),
            moq=raw.get("minimumQuantity"),
            raw=raw,
        )
        logger.info(
            f"Parsed price id={price.id!r} sku={price.sku!r} amount={price.amount} {price.currency} scope={price.scope!r}",
            extra={
                "request_id": price.sku,
                "function_name": "Price.from_raw",
                "input_data": {"value": value, "custom_fields": custom_fields},
                "response_payload": {
                    "id": price.id, "sku": price.sku, "amount": price.amount,
                    "currency": price.currency, "scope": price.scope,
                    "seller_name_en": price.seller_name_en, "contract_id": price.contract_id,
                    "lead_time_days": price.lead_time_days, "moq": price.moq,
                },
            },
        )
        return price


def _channel_name(raw: dict, locale: str) -> str | None:
    channel_obj = raw.get("channel", {}).get("obj", {})
    name = channel_obj.get("name")
    if isinstance(name, dict):
        return name.get(locale) or None
    return name or None if locale == "en" else None


class PricesClient:
    """Fetches standalone prices from CommerceTools using cursor-based pagination."""

    def __init__(self, config: Config, auth: AuthClient) -> None:
        self._config = config
        self._auth = auth

    @log_function
    @requires_token
    async def get_all(self) -> list[Price]:
        """Return every standalone price using parallel offset-based pagination."""
        logger.info("Starting price fetch", extra={"function_name": "get_all"})
        async with aiohttp.ClientSession() as session:
            first = await self._fetch_page(session, offset=0)
            total: int = first["total"]
            results: list[dict] = first["results"]

            if total > self._config.page_size:
                offsets = range(self._config.page_size, total, self._config.page_size)
                logger.info(
                    f"Fetching {len(offsets)} remaining price pages concurrently (total={total})",
                    extra={"function_name": "get_all", "input_data": {"pages": len(offsets) + 1}},
                )
                pages = await asyncio.gather(
                    *[self._fetch_page(session, offset=off) for off in offsets]
                )
                for page in pages:
                    results.extend(page["results"])

        logger.info(
            f"Fetched {len(results)} prices in {1 + (total - 1) // self._config.page_size} page(s)",
            extra={"function_name": "get_all", "status": "success", "input_data": {"total": len(results)}},
        )
        return [Price.from_raw(r) for r in results]

    async def _fetch_page(self, session: aiohttp.ClientSession, offset: int) -> dict:
        from datetime import datetime
        params = {
            "limit": self._config.page_size,
            "sort": "id asc",
            "offset": offset,
            "expand": "channel.id",
        }
        url = f"{self._config.api_base}/standalone-prices"
        logger.info(
            f"GET {url} offset={offset} params={params}",
            extra={"function_name": "_fetch_page", "input_data": {"url": url, "params": params, "offset": offset, "limit": self._config.page_size}},
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
                    f"HTTP {status} ← {full_url} offset={offset} → {count} records (total={body.get('total')}) in {elapsed:.3f}s",
                    extra={
                        "function_name": "_fetch_page",
                        "latency": f"{elapsed:.3f}s",
                        "status": "success" if resp.ok else "error",
                        "input_data": {"full_url": full_url, "http_status": status, "offset": offset, "returned": count, "total": body.get("total")},
                        "response_payload": body,
                    },
                )
                resp.raise_for_status()
                return body
        except Exception as exc:
            elapsed = (datetime.now() - t0).total_seconds()
            logger.error(
                f"Request failed after {elapsed:.3f}s: GET {url} offset={offset}: {exc}",
                extra={
                    "function_name": "_fetch_page",
                    "latency": f"{elapsed:.3f}s",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"url": url, "params": params, "offset": offset},
                },
            )
            raise
