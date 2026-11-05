"""Query and image embeddings from OCI GenAI (Cohere Embed v4.0).

The OCI SDK is blocking, so every SDK call runs in a worker thread
(``asyncio.to_thread``) to keep the event loop free.
"""
from __future__ import annotations

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

from config import Settings, settings as _settings
from utils.logger import logger

# Repeated search-box queries are served from this in-process LRU cache and skip OCI entirely.
QUERY_CACHE_SIZE = 2048


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: "oci.generative_ai_inference.GenerativeAiInferenceClient | None" = None
        self._cache: OrderedDict[tuple[str, int, bool], list[float]] = OrderedDict()

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model_name

    async def load(self) -> None:
        """Create the OCI GenAI client from the configured profile (once)."""
        if self._client is not None:
            return
        cfg = self._settings
        config_path = os.path.expanduser(cfg.oci_config_file)
        logger.info(f"Embedder — loading OCI config '{config_path}' (profile={cfg.oci_profile}, region={cfg.oci_region})")
        try:
            config = await asyncio.to_thread(oci.config.from_file, file_location=config_path, profile_name=cfg.oci_profile)
            config["region"] = cfg.oci_region
            self._client = await asyncio.to_thread(
                oci.generative_ai_inference.GenerativeAiInferenceClient,
                config=config,
                service_endpoint=cfg.oci_genai_endpoint,
            )
        except Exception as exc:
            logger.exception(
                f"Embedder — failed to create the OCI GenAI client ({type(exc).__name__}): {exc}",
                extra={"function_name": "EmbeddingService.load", "status": "error", "error_details": str(exc)},
            )
            raise
        logger.info(f"Embedder — OCI GenAI client ready (model={cfg.embedding_model_name})", extra={"status": "success"})

    async def encode_query(self, query: str) -> list[float]:
        """Embed a search query (input type SEARCH_QUERY). Served from the LRU cache when seen before."""
        cfg = self._settings
        cache_key = (query.strip().lower(), cfg.embedding_dimension, cfg.embedding_normalize)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            logger.info("Embedder — query cache hit (OCI call skipped)")
            return cached

        vector = await self._embed(EmbedTextDetails(
            compartment_id=cfg.oci_compartment_id,
            serving_mode=OnDemandServingMode(serving_type="ON_DEMAND", model_id=cfg.embedding_model_name),
            inputs=[query],
            input_type="SEARCH_QUERY",
            truncate="END",
            output_dimensions=cfg.embedding_dimension,
            is_echo=False,
        ))

        self._cache[cache_key] = vector
        if len(self._cache) > QUERY_CACHE_SIZE:
            self._cache.popitem(last=False)  # drop the least recently used query
        return vector

    async def encode_image(self, data: bytes, mime: str) -> list[float]:
        """Embed an uploaded image (input type IMAGE).

        OCI takes images as base64 data URIs — the same way the ingest pipeline
        sends product images, so the vectors are directly comparable.
        """
        cfg = self._settings
        data_uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        return await self._embed(EmbedTextDetails(
            compartment_id=cfg.oci_compartment_id,
            serving_mode=OnDemandServingMode(serving_type="ON_DEMAND", model_id=cfg.embedding_model_name),
            embed_contents=[EmbedImageContent(image_url=ImageUrl(url=data_uri))],
            input_type="IMAGE",
            embedding_types=["float"],
            output_dimensions=cfg.embedding_dimension,
            is_echo=False,
        ))

    async def _embed(self, details: EmbedTextDetails) -> list[float]:
        """Send one embed request to OCI and return the (optionally L2-normalised) vector."""
        if self._client is None:
            await self.load()
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(self._client.embed_text, embed_text_details=details)
        except Exception as exc:
            oci_details = {}
            if isinstance(exc, oci.exceptions.ServiceError):
                oci_details = {"oci_status": exc.status, "oci_code": exc.code, "oci_message": exc.message,
                               "oci_request_id": exc.request_id}
            logger.error(
                f"Embedder — OCI embed ({details.input_type}) failed after "
                f"{(time.perf_counter() - start) * 1000:.1f}ms ({type(exc).__name__}): {exc}",
                extra={"function_name": "EmbeddingService._embed", "status": "error",
                       "error_details": str(exc), **oci_details},
            )
            raise
        logger.info(
            f"Embedder — OCI embed ({details.input_type}) took {(time.perf_counter() - start) * 1000:.1f}ms",
            extra={"status": "success"},
        )

        # Plain float output comes back in `embeddings`; when `embedding_types` is requested
        # the vectors come back in `embeddings_by_type.float` instead.
        result = response.data
        vectors = getattr(result, "embeddings", None)
        if not vectors:
            by_type = getattr(result, "embeddings_by_type", None)
            if by_type:
                vectors = by_type.get("float") if isinstance(by_type, dict) else getattr(by_type, "float", None)
        if not vectors:
            raise ValueError("OCI embed response contained no embeddings")
        vector = list(vectors[0])

        if self._settings.embedding_normalize:
            norm = math.sqrt(sum(v * v for v in vector))
            if norm == 0.0:
                logger.warning(f"Embedder — zero-norm vector (dim={len(vector)}); left unnormalised")
            else:
                vector = [v / norm for v in vector]
        return vector


embedding_service = EmbeddingService(_settings)
