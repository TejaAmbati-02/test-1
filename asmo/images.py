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
from typing import Awaitable, Callable, Optional
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
    async def download_all(
        self,
        transformed: dict[str, dict],
        write_json: Optional[Callable[[str, dict], Awaitable]] = None,
        upload_json: Optional[Callable[[str, dict], Awaitable]] = None,
        upload_image: Optional[Callable[[str, str, bytes], Awaitable]] = None,
    ) -> dict:
        """Persist every transformed row's JSON and image, optionally to storage.

        Each SKU is handled by its own coroutine; all SKUs run concurrently,
        bounded end-to-end by the configured semaphore (so at most
        ``concurrency`` SKUs — and therefore their in-memory image bytes and
        open upload clients — are ever in flight, which caps memory use).

        Per SKU the order is:

        1. ``write_json(sku, row)`` — create the SKU's local JSON file (if given).
        2. Then, **in parallel**:
             * ``upload_json(sku, row)`` — upload that JSON to object storage, and
             * download the image, save it locally, then ``upload_image(sku,
               filename, data)`` — upload the image bytes to object storage.

        So as soon as a SKU's JSON exists, its JSON upload and its image
        download+upload proceed together, rather than running JSON and images as
        separate bulk stages. Identical image URLs are still fetched only once
        via a shared per-run cache. Upload failures for one SKU are logged and
        counted but never abort the run.

        Args:
            transformed: Mapping of SKU to transformed row. Rows without an
                ``imageUrl`` still have their JSON written/uploaded but are
                counted as skipped for the image download.
            write_json: Optional async callback ``(sku, row)`` that persists the
                SKU's JSON locally.
            upload_json: Optional async callback ``(sku, row)`` that uploads the
                SKU's JSON to object storage. When ``None``, JSON is not uploaded.
            upload_image: Optional async callback ``(sku, filename, data)`` that
                uploads the image bytes to object storage. When ``None``, images
                are not uploaded.

        Returns:
            A summary dict with ``downloaded``, ``skipped``, ``failed`` (image
            counts) plus ``json_uploaded``, ``image_uploaded`` and
            ``upload_failed`` (storage counts).
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        with_url = sum(1 for row in transformed.values() if row.get("imageUrl"))
        without_url = len(transformed) - with_url
        uploading = upload_json is not None or upload_image is not None
        logger.info(
            f"Processing {len(transformed)} SKUs — {with_url} with imageUrl, {without_url} without"
            + (" (per SKU: write+upload JSON and download+upload image in parallel)" if uploading
               else " (writing JSON then downloading each image per SKU)" if write_json else ""),
            extra={
                "function_name": "download_all",
                "input_data": {"total": len(transformed), "with_url": with_url, "without_url": without_url, "uploading": uploading},
            },
        )

        sem = asyncio.Semaphore(self._concurrency)
        url_cache: dict[str, asyncio.Task] = {}
        counters = {
            "downloaded": 0, "skipped": 0, "failed": 0,
            "json_uploaded": 0, "image_uploaded": 0, "upload_failed": 0,
        }

        async with aiohttp.ClientSession() as session:

            async def fetch(url: str) -> bytes | None:
                """Fetch a URL's bytes once; concurrent callers share the result."""
                task = url_cache.get(url)
                if task is None:
                    task = asyncio.create_task(self._get_bytes(session, url))
                    url_cache[url] = task
                return await task

            async def json_to_storage(sku: str, row: dict) -> None:
                """Upload the SKU's JSON to object storage (tolerant of failure)."""
                if upload_json is None:
                    return
                try:
                    await upload_json(sku, row)
                    counters["json_uploaded"] += 1
                    logger.debug(
                        f"Uploaded JSON for sku={sku!r}",
                        extra={"request_id": sku, "function_name": "download_all", "input_data": {"sku": sku}},
                    )
                except Exception as exc:
                    counters["upload_failed"] += 1
                    logger.error(
                        f"Failed to upload JSON for sku={sku!r}: {exc}",
                        extra={"request_id": sku, "function_name": "download_all", "status": "error",
                               "error_details": str(exc), "input_data": {"sku": sku}},
                    )

            async def image_to_storage(sku: str, row: dict) -> None:
                """Download the SKU's image, save it, then upload it (tolerant)."""
                url = row.get("imageUrl")
                if not url:
                    counters["skipped"] += 1
                    return

                data = await fetch(url)
                if data is None:
                    counters["failed"] += 1
                    return

                filename = self._filename_for(sku, url)
                path = self._output_dir / filename
                try:
                    await asyncio.to_thread(path.write_bytes, data)
                    counters["downloaded"] += 1
                    logger.debug(
                        f"Saved image for sku={sku!r} → {path}",
                        extra={"function_name": "download_all", "input_data": {"sku": sku, "path": str(path)}},
                    )
                except Exception as exc:
                    counters["failed"] += 1
                    logger.error(
                        f"Failed to write image for sku={sku!r} to {path}: {exc}",
                        extra={"function_name": "download_all", "status": "error",
                               "error_details": str(exc), "input_data": {"sku": sku, "path": str(path)}},
                    )
                    return

                if upload_image is not None:
                    try:
                        await upload_image(sku, filename, data)
                        counters["image_uploaded"] += 1
                        logger.debug(
                            f"Uploaded image for sku={sku!r} ({filename})",
                            extra={"request_id": sku, "function_name": "download_all", "input_data": {"sku": sku, "filename": filename}},
                        )
                    except Exception as exc:
                        counters["upload_failed"] += 1
                        logger.error(
                            f"Failed to upload image for sku={sku!r} ({filename}): {exc}",
                            extra={"request_id": sku, "function_name": "download_all", "status": "error",
                                   "error_details": str(exc), "input_data": {"sku": sku, "filename": filename}},
                        )

            async def process(sku: str, row: dict) -> None:
                """Write a SKU's JSON, then upload JSON and handle image in parallel.

                The whole per-SKU unit of work is bounded by the shared
                semaphore so memory and open upload clients stay capped.
                """
                async with sem:
                    if write_json is not None:
                        await write_json(sku, row)
                    # JSON upload and image download+upload run concurrently.
                    await asyncio.gather(
                        json_to_storage(sku, row),
                        image_to_storage(sku, row),
                    )

            await asyncio.gather(*(process(sku, row) for sku, row in transformed.items()))

        logger.info(
            f"Processing complete — images: {counters['downloaded']} downloaded, "
            f"{counters['skipped']} skipped, {counters['failed']} failed "
            f"({len(url_cache)} unique URLs); "
            f"storage: {counters['json_uploaded']} JSON + {counters['image_uploaded']} images uploaded, "
            f"{counters['upload_failed']} upload failures",
            extra={
                "function_name": "download_all",
                "status": "success",
                "input_data": {**counters, "unique_urls": len(url_cache)},
            },
        )
        return counters

    async def _get_bytes(
        self, session: aiohttp.ClientSession, url: str
    ) -> bytes | None:
        """Fetch a single URL's bytes, returning None on any failure.

        Concurrency is bounded by the caller (each SKU's work runs under the
        shared semaphore in :meth:`download_all`), so this method does not
        acquire the semaphore itself — doing so would deadlock when a bounded
        SKU coroutine awaits a fetch that also needs a slot.
        """
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
