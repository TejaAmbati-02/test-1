"""Query and image embeddings from OCI GenAI (Cohere Embed v4.0).

Use the ready-made object:

    from services.embedder import embedding_service
    vector = await embedding_service.encode_query("ball valve")
    vector = await embedding_service.encode_image(image_bytes, "image/png")

The OCI SDK is blocking, so every SDK call runs in a worker thread (``asyncio.to_thread``)
to keep the event loop free. Each method is self-contained (it calls no other method).
"""
import asyncio
import base64
import math
import os
import time
from collections import OrderedDict

import oci
from oci.generative_ai_inference.models import (
    EmbedImageContent,
    EmbedTextDetails,
    ImageUrl,
    OnDemandServingMode,
)

from config import Settings, settings
from utils.logger import logger


class EmbeddingService:
    def __init__(self, settings: Settings, cache_size: int = 2048) -> None:
        self._settings = settings
        self._client: "oci.generative_ai_inference.GenerativeAiInferenceClient | None" = None
        # Repeated search-box queries are served from this in-process LRU cache and skip OCI entirely.
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, int, bool], list[float]] = OrderedDict()

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model_name

    async def load(self) -> None:
        """Create the OCI GenAI client from the configured profile (once, at startup)."""
        if self._client is not None:
            return
        cfg = self._settings
        config_path = os.path.expanduser(cfg.oci_config_file)
        logger.info(f"Embedder — loading OCI config '{config_path}' (profile={cfg.oci_profile}, region={cfg.oci_region})")
        try:
            config = await asyncio.to_thread(oci.config.from_file, file_location=config_path, profile_name=cfg.oci_profile)
            config["region"] = cfg.oci_region
            self._client = await asyncio.to_thread(
                oci.generative_ai_inference.GenerativeAiInferenceClient, config=config, service_endpoint=cfg.oci_genai_endpoint,
            )
        except Exception as exc:
            logger.exception(f"Embedder — failed to create the OCI GenAI client ({type(exc).__name__}): {exc}",
                             extra={"function_name": "EmbeddingService.load", "status": "error", "error_details": str(exc)})
            raise
        logger.info(f"Embedder — OCI GenAI client ready (model={cfg.embedding_model_name})", extra={"status": "success"})

    async def keep_warm(self, interval: int) -> None:
        """Every ``interval`` seconds send a tiny embed straight to OCI so the on-demand model stays warm
        (a cold model costs ~1.5s on the first embed). Each call is billable.

        Goes to OCI directly, not through encode_query — that would be answered from the cache and
        never reach OCI. Runs until cancelled (app shutdown). A failed call is logged, never fatal.
        """
        cfg = self._settings
        logger.info(f"OCI embed heartbeat started — every {interval}s")
        try:
            while True:
                await asyncio.sleep(interval)
                if self._client is None:
                    logger.warning("OCI embed heartbeat skipped: the OCI client is not loaded")
                    continue
                start = time.perf_counter()
                try:
                    await asyncio.to_thread(self._client.embed_text, embed_text_details=EmbedTextDetails(
                        compartment_id=cfg.oci_compartment_id,
                        serving_mode=OnDemandServingMode(serving_type="ON_DEMAND", model_id=cfg.embedding_model_name),
                        inputs=["warmup"],
                        input_type="SEARCH_QUERY",
                        truncate="END",
                        output_dimensions=cfg.embedding_dimension,
                        is_echo=False,
                    ))
                    logger.debug(f"OCI embed heartbeat ok in {(time.perf_counter() - start) * 1000:.1f}ms")
                except Exception as exc:
                    logger.warning(f"OCI embed heartbeat failed ({type(exc).__name__}): {exc}",
                                   extra={"status": "error", "error_details": str(exc)})
        except asyncio.CancelledError:
            logger.info("OCI embed heartbeat cancelled")
            raise

    async def encode_query(self, query: str) -> list[float]:
        """Embed a search query (input type SEARCH_QUERY). Served from the LRU cache when seen before."""
        cfg = self._settings

        # 1. Cache hit? No OCI call.
        cache_key = (query.strip().lower(), cfg.embedding_dimension, cfg.embedding_normalize)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            logger.info("Embedder — query cache hit (OCI call skipped)")
            return cached

        # 2. OCI client (created on first use if startup didn't).
        if self._client is None:
            config = await asyncio.to_thread(oci.config.from_file, file_location=os.path.expanduser(cfg.oci_config_file),
                                             profile_name=cfg.oci_profile)
            config["region"] = cfg.oci_region
            self._client = await asyncio.to_thread(
                oci.generative_ai_inference.GenerativeAiInferenceClient, config=config, service_endpoint=cfg.oci_genai_endpoint,
            )

        # 3. Ask OCI for the embedding.
        details = EmbedTextDetails(
            compartment_id=cfg.oci_compartment_id,
            serving_mode=OnDemandServingMode(serving_type="ON_DEMAND", model_id=cfg.embedding_model_name),
            inputs=[query],
            input_type="SEARCH_QUERY",
            truncate="END",
            output_dimensions=cfg.embedding_dimension,
            is_echo=False,
        )
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(self._client.embed_text, embed_text_details=details)
        except Exception as exc:
            oci_details = {}
            if isinstance(exc, oci.exceptions.ServiceError):
                oci_details = {"oci_status": exc.status, "oci_code": exc.code, "oci_message": exc.message,
                               "oci_request_id": exc.request_id}
            logger.error(f"Embedder — OCI embed (SEARCH_QUERY) failed after {(time.perf_counter() - start) * 1000:.1f}ms "
                         f"({type(exc).__name__}): {exc}",
                         extra={"function_name": "EmbeddingService.encode_query", "status": "error",
                                "error_details": str(exc), **oci_details})
            raise
        logger.info(f"Embedder — OCI embed (SEARCH_QUERY) took {(time.perf_counter() - start) * 1000:.1f}ms",
                    extra={"status": "success"})

        # 4. Read the vector: plain output is in `embeddings`, typed output in `embeddings_by_type.float`.
        vectors = getattr(response.data, "embeddings", None)
        by_type = getattr(response.data, "embeddings_by_type", None)
        if not vectors and by_type:
            vectors = by_type.get("float") if isinstance(by_type, dict) else getattr(by_type, "float", None)
        if not vectors:
            raise ValueError("OCI embed response contained no embeddings")
        vector = list(vectors[0])

        # 5. Normalise to length 1 (L2), if configured.
        if cfg.embedding_normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm == 0.0:
                logger.warning(f"Embedder — zero-norm vector (dim={len(vector)}); left unnormalised")
            else:
                vector = [v / norm for v in vector]

        # 6. Remember it (least recently used query dropped when the cache is full).
        self._cache[cache_key] = vector
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return vector

    async def encode_image(self, data: bytes, mime: str) -> list[float]:
        """Embed an uploaded image (input type IMAGE).

        OCI takes images as base64 data URIs — the same way the ingest pipeline
        sends product images, so the vectors are directly comparable.
        """
        cfg = self._settings

        # 1. OCI client (created on first use if startup didn't).
        if self._client is None:
            config = await asyncio.to_thread(oci.config.from_file, file_location=os.path.expanduser(cfg.oci_config_file),
                                             profile_name=cfg.oci_profile)
            config["region"] = cfg.oci_region
            self._client = await asyncio.to_thread(
                oci.generative_ai_inference.GenerativeAiInferenceClient, config=config, service_endpoint=cfg.oci_genai_endpoint,
            )

        # 2. Ask OCI for the embedding.
        data_uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        details = EmbedTextDetails(
            compartment_id=cfg.oci_compartment_id,
            serving_mode=OnDemandServingMode(serving_type="ON_DEMAND", model_id=cfg.embedding_model_name),
            embed_contents=[EmbedImageContent(image_url=ImageUrl(url=data_uri))],
            input_type="IMAGE",
            embedding_types=["float"],
            output_dimensions=cfg.embedding_dimension,
            is_echo=False,
        )
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(self._client.embed_text, embed_text_details=details)
        except Exception as exc:
            oci_details = {}
            if isinstance(exc, oci.exceptions.ServiceError):
                oci_details = {"oci_status": exc.status, "oci_code": exc.code, "oci_message": exc.message,
                               "oci_request_id": exc.request_id}
            logger.error(f"Embedder — OCI embed (IMAGE) failed after {(time.perf_counter() - start) * 1000:.1f}ms "
                         f"({type(exc).__name__}): {exc}",
                         extra={"function_name": "EmbeddingService.encode_image", "status": "error",
                                "error_details": str(exc), **oci_details})
            raise
        logger.info(f"Embedder — OCI embed (IMAGE) took {(time.perf_counter() - start) * 1000:.1f}ms",
                    extra={"status": "success"})

        # 3. Read the vector: plain output is in `embeddings`, typed output in `embeddings_by_type.float`.
        vectors = getattr(response.data, "embeddings", None)
        by_type = getattr(response.data, "embeddings_by_type", None)
        if not vectors and by_type:
            vectors = by_type.get("float") if isinstance(by_type, dict) else getattr(by_type, "float", None)
        if not vectors:
            raise ValueError("OCI embed response contained no embeddings")
        vector = list(vectors[0])

        # 4. Normalise to length 1 (L2), if configured.
        if cfg.embedding_normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm == 0.0:
                logger.warning(f"Embedder — zero-norm vector (dim={len(vector)}); left unnormalised")
            else:
                vector = [v / norm for v in vector]
        return vector


# The one object the app uses — import it where needed:  from services.embedder import embedding_service
embedding_service = EmbeddingService(settings)
