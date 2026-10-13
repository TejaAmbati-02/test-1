import asyncio
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

        logger.info("Step 3/5 — transforming", extra={"function_name": "run"})
        transformed = await transform(products, prices, self._config.channel_id)

        logger.info(
            "Step 4/5 — exporting and downloading images concurrently",
            extra={"function_name": "run"},
        )
        paths, image_summary = await asyncio.gather(
            self._exporter.export(products, prices, transformed),
            self._images.download_all(transformed),
        )
        logger.info(
            f"Image download — {image_summary['downloaded']} downloaded, "
            f"{image_summary['skipped']} skipped, {image_summary['failed']} failed",
            extra={"function_name": "run", "status": "success", "input_data": image_summary},
        )

        if self._config.upload_enabled:
            logger.info("Step 5/5 — uploading transformed files to object storage", extra={"function_name": "run"})
            summary = await self._storage.upload_from_local_folder(
                local_folder=str(paths.transformed),
                target_folder=Folder.PRODUCTS,
                validate_json=True,
                concurrency=10,
            )
            logger.info(
                f"Upload complete — {summary['uploaded']} uploaded, {summary['failed']} failed",
                extra={"function_name": "run", "status": "success", "input_data": summary},
            )
        else:
            logger.info("Step 5/5 — upload disabled, transformed files stored locally only", extra={"function_name": "run"})

        logger.info(
            f"Pipeline run complete — {len(products)} products, {len(prices)} prices, {len(transformed)} rows exported",
            extra={"function_name": "run", "status": "success"},
        )
        return PipelineResult(products=products, prices=prices, export_paths=paths)
