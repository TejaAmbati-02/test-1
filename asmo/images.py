"""Concurrent product-image downloading for the ASMO Data pipeline.

This module exposes :class:`ImageDownloader`, which takes the transformed SKU
rows produced upstream and downloads each row's ``imageUrl`` concurrently,
writing one image file per SKU into an output directory. Downloads are bounded
by a semaphore, identical URLs are fetched only once and shared, and failures
are tolerated and tallied rather than aborting the run.

Network calls use :mod:`aiohttp` with the project's shared SSL context and are
logged through the structured logger.
"""

import asyncio
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


class ImageDownloader:
    """Downloads product images concurrently, one file per SKU.

    Each image is saved into ``output_dir`` using the same stem as the SKU's
    transformed JSON file, so ``001000-002-chemicals`` → ``001000002chemicals.png``.
    Identical URLs are fetched only once and reused across SKUs.

    A single shared instance is created on first use via :meth:`instance` and
    reused for the lifetime of the application.
    """

    OUTPUT_DIR = Path("out_images")

    _instance: "ImageDownloader | None" = None

    def __init__(self, base_dir: Path = Path("."), concurrency: int = 10) -> None:
        """Configure the output location and concurrency limit.

        Args:
            base_dir: Directory under which the :data:`OUTPUT_DIR` image folder
                is created (defaults to the current directory).
            concurrency: Maximum number of image downloads allowed to run at
                once.
        """
        self._output_dir = base_dir / self.OUTPUT_DIR
        self._concurrency = concurrency

    @classmethod
    def instance(cls, base_dir: Path = Path("."), concurrency: int = 10) -> "ImageDownloader":
        """Return the shared downloader, creating it on the first call.

        Later calls reuse the object built on the first call and ignore their
        arguments, so the whole application shares one downloader.
        """
        if cls._instance is None:
            cls._instance = cls(base_dir=base_dir, concurrency=concurrency)
        return cls._instance

    @staticmethod
    def _filename_for(sku: str, url: str) -> str:
        """Build the image filename for a SKU, mirroring the exporter's JSON naming.

        The transformed JSON for a SKU is written as ``{sku without dashes}.json`` by
        the Exporter, so the image gets the same stem with whatever extension the URL
        carries (``.jpg`` when the URL path has none).
        """
        stem = sku.replace("-", "")
        ext = Path(urlparse(url).path).suffix.lower() or ".jpg"
        return f"{stem}{ext}"

    @log_function
    async def download_all(self, transformed: dict[str, dict]) -> dict:
        """Download the ``imageUrl`` of every transformed row concurrently.

        Returns a summary dict with ``downloaded``, ``skipped`` and ``failed`` counts.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        targets = [
            (sku, row["imageUrl"])
            for sku, row in transformed.items()
            if row.get("imageUrl")
        ]
        skipped_no_url = len(transformed) - len(targets)
        logger.info(
            f"Downloading images for {len(targets)} SKUs ({skipped_no_url} without imageUrl)",
            extra={
                "function_name": "download_all",
                "input_data": {"with_url": len(targets), "without_url": skipped_no_url},
            },
        )

        sem = asyncio.Semaphore(self._concurrency)
        url_cache: dict[str, asyncio.Task] = {}
        counters = {"downloaded": 0, "skipped": skipped_no_url, "failed": 0}

        async with aiohttp.ClientSession() as session:

            async def fetch(url: str) -> bytes | None:
                """Fetch a URL's bytes once; concurrent callers share the result."""
                task = url_cache.get(url)
                if task is None:
                    task = asyncio.create_task(self._get_bytes(session, sem, url))
                    url_cache[url] = task
                return await task

            async def handle(sku: str, url: str) -> None:
                data = await fetch(url)
                if data is None:
                    counters["failed"] += 1
                    return
                path = self._output_dir / self._filename_for(sku, url)
                try:
                    await asyncio.to_thread(path.write_bytes, data)
                    counters["downloaded"] += 1
                except Exception as exc:
                    counters["failed"] += 1
                    logger.error(
                        f"Failed to write image for sku={sku!r} to {path}: {exc}",
                        extra={
                            "function_name": "download_all",
                            "status": "error",
                            "error_details": str(exc),
                            "input_data": {"sku": sku, "path": str(path)},
                        },
                    )

            await asyncio.gather(*(handle(sku, url) for sku, url in targets))

        logger.info(
            f"Image download complete — {counters['downloaded']} downloaded, "
            f"{counters['skipped']} skipped, {counters['failed']} failed "
            f"({len(url_cache)} unique URLs)",
            extra={
                "function_name": "download_all",
                "status": "success",
                "input_data": {**counters, "unique_urls": len(url_cache)},
            },
        )
        return counters

    async def _get_bytes(
        self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, url: str
    ) -> bytes | None:
        """Fetch a single URL's bytes, returning None on any failure."""
        async with sem:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), ssl=SSL_CONTEXT) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except Exception as exc:
                logger.warning(
                    f"Image download failed for {url}: {exc}",
                    extra={
                        "function_name": "_get_bytes",
                        "status": "error",
                        "error_details": str(exc),
                        "input_data": {"url": url},
                    },
                )
                return None
