"""Persists fetched and transformed data to the local filesystem as JSON.

This module is the output stage of the pipeline. Given the raw products and
prices fetched from CommerceTools and the flattened SKU rows produced by the
transformer, it writes one JSON file per record into dedicated output folders
(``out_products``, ``out_prices`` and ``out_transformed``).

The blocking file I/O is executed on worker threads via
:func:`asyncio.to_thread` so that writing does not stall the event loop in the
otherwise asynchronous pipeline. The public entry point is
:meth:`Exporter.export`, which returns an :class:`ExportPaths` describing where
each category of data was written.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from .products import Product
from .prices import Price
from .logging_service import logger, log_function


@dataclass
class ExportPaths:
    """Locations of the folders that each category of export was written to.

    Attributes:
        products: Directory containing the raw product JSON files.
        prices: Directory containing the raw price JSON files.
        transformed: Directory containing the transformed SKU-row JSON files.
    """

    products: Path
    prices: Path
    transformed: Path


class Exporter:
    """Writes fetched and transformed data as JSON files into the configured output folders.

    Each exporter instance resolves its three output folders relative to a base
    directory supplied at construction time. Records are written one file per
    record, named after the record id (or SKU) with hyphens stripped.
    """

    PRODUCTS_DIR = Path("out_products")
    PRICES_DIR = Path("out_prices")
    TRANSFORMED_DIR = Path("out_transformed")

    def __init__(self, base_dir: Path = Path(".")) -> None:
        """Initialize the exporter and resolve its output folders.

        Args:
            base_dir: Base directory under which the ``out_products``,
                ``out_prices`` and ``out_transformed`` folders are placed.
                Defaults to the current working directory.
        """
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
        """Write all products, prices and transformed rows to disk.

        Ensures the three output folders exist, then writes the raw product
        documents, the raw price documents and the transformed SKU rows, each
        into its own folder.

        Args:
            products: The fetched products whose raw documents are written.
            prices: The fetched prices whose raw documents are written.
            transformed: Mapping of SKU to transformed row to write.

        Returns:
            An :class:`ExportPaths` describing the folders written to.
        """
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
        """Write a list of raw records to a folder, one JSON file per record.

        Offloads the blocking writes to a worker thread and logs the elapsed
        time once complete.

        Args:
            folder: Destination directory for the JSON files.
            records: Raw record dicts to write; each must contain an ``id``.
        """
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
        """Write the transformed SKU rows to the transformed folder.

        Offloads the blocking writes to a worker thread and logs the elapsed
        time once complete.

        Args:
            transformed: Mapping of SKU to transformed row; each SKU becomes a
                JSON file in the transformed output folder.
        """
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
        """Synchronously write raw records to a folder, one JSON file per record.

        Runs on a worker thread (via :meth:`_write_each`). Each file is named
        after the record's ``id`` with hyphens removed and is written as
        indented, non-ASCII-escaped JSON.

        Args:
            folder: Destination directory for the JSON files.
            records: Raw record dicts to write; each must contain an ``id``.

        Raises:
            Exception: Re-raises any write error after logging it with the
                offending record id.
        """
        for record in records:
            try:
                file_path = folder / f"{record['id'].replace('-', '')}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                logger.debug(
                    f"Wrote record id={record.get('id')!r} to {folder}/",
                    extra={"function_name": "_write_each_sync", "input_data": {"id": record.get("id"), "folder": str(folder)}},
                )
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
        """Synchronously write transformed rows, one JSON file per SKU.

        Runs on a worker thread (via :meth:`_write_transformed`). Each file is
        named after the SKU with hyphens removed and is written as indented,
        non-ASCII-escaped JSON.

        Args:
            transformed: Mapping of SKU to transformed row to write.

        Raises:
            Exception: Re-raises any write error after logging it with the
                offending SKU.
        """
        for sku, row in transformed.items():
            try:
                file_path = self._transformed_dir / f"{sku.replace('-', '')}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(row, f, indent=2, ensure_ascii=False)
                logger.debug(
                    f"Wrote transformed row sku={sku!r} to {self._transformed_dir}/",
                    extra={
                        "request_id": sku,
                        "function_name": "_write_transformed_sync",
                        "input_data": {"sku": sku, "folder": str(self._transformed_dir)},
                    },
                )
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
