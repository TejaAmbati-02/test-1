import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from .products import Product
from .prices import Price
from .logging_service import logger, log_function


@dataclass
class ExportPaths:
    products: Path
    prices: Path
    transformed: Path


class Exporter:
    """Writes fetched and transformed data as JSON files into the configured output folders."""

    PRODUCTS_DIR = Path("out_products")
    PRICES_DIR = Path("out_prices")
    TRANSFORMED_DIR = Path("out_transformed")

    def __init__(self, base_dir: Path = Path(".")) -> None:
        self._products_dir = base_dir / self.PRODUCTS_DIR
        self._prices_dir = base_dir / self.PRICES_DIR
        self._transformed_dir = base_dir / self.TRANSFORMED_DIR


    @log_function
    async def export(
        self,
        products: list[Product],
        prices: list[Price],
        transformed: dict[str, dict],
    ) -> ExportPaths:
        logger.info(
            f"Exporting {len(products)} products, {len(prices)} prices, {len(transformed)} transformed rows",
            extra={"function_name": "export"},
        )
        for folder in (self._products_dir, self._prices_dir, self._transformed_dir):
            folder.mkdir(parents=True, exist_ok=True)

        await self._write_each(self._products_dir, [p.raw for p in products])
        await self._write_each(self._prices_dir, [p.raw for p in prices])
        await self._write_transformed(transformed)

        paths = ExportPaths(
            products=self._products_dir,
            prices=self._prices_dir,
            transformed=self._transformed_dir,
        )
        logger.info("Export complete", extra={"function_name": "export", "status": "success"})
        return paths

    async def _write_each(self, folder: Path, records: list[dict]) -> None:
        from datetime import datetime
        logger.info(
            f"Writing {len(records)} records to {folder}/",
            extra={"function_name": "_write_each", "input_data": {"count": len(records), "folder": str(folder)}},
        )
        t0 = datetime.now()
        await asyncio.to_thread(self._write_each_sync, folder, records)
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f"{len(records):>6} records → {folder}/ in {elapsed:.3f}s",
            extra={
                "function_name": "_write_each",
                "latency": f"{elapsed:.3f}s",
                "status": "success",
                "input_data": {"count": len(records), "folder": str(folder)},
            },
        )

    async def _write_transformed(self, transformed: dict[str, dict]) -> None:
        from datetime import datetime
        logger.info(
            f"Writing {len(transformed)} transformed records to {self._transformed_dir}/",
            extra={"function_name": "_write_transformed", "input_data": {"count": len(transformed)}},
        )
        t0 = datetime.now()
        await asyncio.to_thread(self._write_transformed_sync, transformed)
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f"{len(transformed):>6} records → {self._transformed_dir}/ in {elapsed:.3f}s",
            extra={
                "function_name": "_write_transformed",
                "latency": f"{elapsed:.3f}s",
                "status": "success",
                "input_data": {"count": len(transformed)},
            },
        )

    @staticmethod
    def _write_each_sync(folder: Path, records: list[dict]) -> None:
        for record in records:
            try:
                file_path = folder / f"{record['id'].replace('-', '')}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.error(
                    f"Failed to write record id={record.get('id')!r} to {folder}/: {exc}",
                    extra={
                        "function_name": "_write_each_sync",
                        "status": "error",
                        "error_details": str(exc),
                        "input_data": {"id": record.get("id"), "folder": str(folder)},
                    },
                )
                raise

    def _write_transformed_sync(self, transformed: dict[str, dict]) -> None:
        for sku, row in transformed.items():
            try:
                file_path = self._transformed_dir / f"{sku.replace('-', '')}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(row, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.error(
                    f"Failed to write transformed row sku={sku!r}: {exc}",
                    extra={
                        "request_id": sku,
                        "function_name": "_write_transformed_sync",
                        "status": "error",
                        "error_details": str(exc),
                        "input_data": {"sku": sku, "folder": str(self._transformed_dir)},
                    },
                )
                raise
