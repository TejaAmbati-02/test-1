import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from .config import Config
from .auth import AuthClient
from .products import ProductsClient, Product
from .prices import PricesClient, Price
from .exporter import Exporter, ExportPaths
from .transformer import transform
from .images import ImageDownloader
from .storage import ObjectStorageClient, ObjectStorageConfig, Folder
from .logging_service import logger, log_function


@dataclass
class PipelineResult:
    """Holds the combined output of a pipeline run."""

    products: list[Product]
    prices: list[Price]
    export_paths: ExportPaths | None = field(default=None, repr=False)

    def summary(self) -> str:
        return f"Products: {len(self.products)}, Prices: {len(self.prices)}"


class Pipeline:
    """Top-level orchestrator — the only class callers need to import."""

    def __init__(self, config: Config | None = None, output_dir: Path = Path(".")) -> None:
        self._config = config or Config()
        self._auth = AuthClient(self._config)
        self._products = ProductsClient(self._config, self._auth)
        self._prices = PricesClient(self._config, self._auth)
        self._exporter = Exporter(base_dir=output_dir)
        self._images = ImageDownloader.instance(base_dir=output_dir)
        self._storage = ObjectStorageClient(ObjectStorageConfig())

    @log_function
    async def run(self) -> PipelineResult:
        """Authenticate, fetch products + prices concurrently, transform, then export."""
        logger.info("Step 1/4 — authenticating", extra={"function_name": "run"})
        await self._auth.authenticate()

        logger.info("Step 2/4 — fetching products and prices concurrently", extra={"function_name": "run"})
        products, prices = await asyncio.gather(
            self._products.get_all(),
            self._prices.get_all(),
        )
        logger.info(
            f"Fetch complete — {len(products)} products, {len(prices)} prices",
            extra={"function_name": "run", "input_data": {"products": len(products), "prices": len(prices)}},
        )

        logger.info("Step 3/4 — transforming", extra={"function_name": "run"})
        transformed = await transform(products, prices, self._config.channel_id)

        # Per-SKU storage callbacks — only wired up when uploads are enabled.
        # When enabled, each SKU's JSON and image are uploaded to object storage
        # in parallel, as they are produced, inside the per-SKU pipeline below.
        upload_json = self._make_upload_json() if self._config.upload_enabled else None
        upload_image = self._make_upload_image() if self._config.upload_enabled else None

        logger.info(
            "Step 4/4 — exporting raw dumps while, per SKU, writing the transformed "
            "JSON then downloading its image and "
            + ("uploading both to object storage in parallel" if self._config.upload_enabled
               else "storing them locally (upload disabled)"),
            extra={"function_name": "run", "input_data": {"upload_enabled": self._config.upload_enabled}},
        )
        paths, image_summary = await asyncio.gather(
            self._exporter.export_raw(products, prices),
            self._images.download_all(
                transformed,
                write_json=self._exporter.write_transformed_row,
                upload_json=upload_json,
                upload_image=upload_image,
            ),
        )
        logger.info(
            f"Export + storage — images: {image_summary['downloaded']} downloaded, "
            f"{image_summary['skipped']} skipped, {image_summary['failed']} failed; "
            f"storage: {image_summary['json_uploaded']} JSON + "
            f"{image_summary['image_uploaded']} images uploaded, "
            f"{image_summary['upload_failed']} upload failures",
            extra={"function_name": "run", "status": "success", "input_data": image_summary},
        )

        logger.info(
            f"Pipeline run complete — {len(products)} products, {len(prices)} prices, {len(transformed)} rows exported",
            extra={"function_name": "run", "status": "success"},
        )
        return PipelineResult(products=products, prices=prices, export_paths=paths)

    def _make_upload_json(self):
        """Build the async callback that uploads one SKU's JSON to storage.

        The returned callback serialises the transformed row to JSON (matching
        the local file's formatting and ``{sku-without-dashes}.json`` naming)
        and uploads it under :data:`Folder.PRODUCTS`.
        """
        async def upload_json(sku: str, row: dict) -> None:
            data = json.dumps(row, indent=2, ensure_ascii=False).encode("utf-8")
            await self._storage.upload_bytes(
                folder=Folder.PRODUCTS,
                filename=f"{sku.replace('-', '')}.json",
                data=data,
                content_type="application/json",
            )
        return upload_json

    def _make_upload_image(self):
        """Build the async callback that uploads one SKU's image to storage.

        The returned callback uploads the already-downloaded image bytes under
        :data:`Folder.IMAGES`, inferring the content type from the filename's
        extension.
        """
        content_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
            ".svg": "image/svg+xml", ".tiff": "image/tiff",
        }

        async def upload_image(sku: str, filename: str, data: bytes) -> None:
            content_type = content_types.get(Path(filename).suffix.lower(), "application/octet-stream")
            await self._storage.upload_bytes(
                folder=Folder.IMAGES,
                filename=filename,
                data=data,
                content_type=content_type,
            )
        return upload_image
