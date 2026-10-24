from __future__ import annotations
 
import asyncio
import base64
import math
import time
from collections import OrderedDict
 
import oci
 
from config import Settings, settings as _settings
from services.oci_genai import build_genai_client, extract_embeddings, oci_error_fields
from utils.logger import logger
 
 
class EmbeddingService:
    """Search-query embeddings backed by OCI GenAI (Cohere Embed v4.0)."""
 
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: "oci.generative_ai_inference.GenerativeAiInferenceClient | None" = None
        # LRU cache of query -> embedding. Repeated search-box queries (the common
        # case) skip the OCI round-trip entirely, side-stepping on-demand cold starts.
        self._cache: "OrderedDict[tuple[str, int, bool], list[float]]" = OrderedDict()
        self._cache_max = getattr(settings, "embedding_cache_size", 2048)
 
    @property
    def is_loaded(self) -> bool:
        return self._client is not None
 
    @property
    def model_name(self) -> str:
        return self._settings.embedding_model_name
 
    async def load(self) -> None:
        if self._client is not None:
            logger.debug("EmbeddingService.load — client already initialized, skipping")
            return
        cfg = self._settings
        logger.info(
            "EmbeddingService.load — starting",
            extra={"input_data": {"model": cfg.embedding_model_name, "region": cfg.oci_region}},
        )
        try:
            loop = asyncio.get_running_loop()
            self._client = await loop.run_in_executor(None, build_genai_client, cfg)
            logger.info(
                "EmbeddingService.load — OCI GenAI client ready",
                extra={"status": "success", "input_data": {"model": cfg.embedding_model_name}},
            )
        except Exception as exc:
            logger.error(
                "EmbeddingService.load — failed to initialize OCI GenAI client",
                extra={"status": "error", "error_details": str(exc), "input_data": {"model": cfg.embedding_model_name}, **oci_error_fields(exc)},
            )
            raise
 
    @staticmethod
    def _l2_normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]
 
    def _embed_query_sync(self, query: str) -> list[float]:
        cfg = self._settings
        logger.debug(
            "EmbeddingService._embed_query_sync — building EmbedTextDetails",
            extra={"input_data": {
                "model": cfg.embedding_model_name,
                "dimension": cfg.embedding_dimension,
                "input_type": "SEARCH_QUERY",
                "compartment_id": cfg.oci_compartment_id,
                "query_preview": query[:120],
                "query_len": len(query),
            }},
        )
        details = oci.generative_ai_inference.models.EmbedTextDetails(
            compartment_id=cfg.oci_compartment_id,
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                serving_type="ON_DEMAND",
                model_id=cfg.embedding_model_name,
            ),
            inputs=[query],
            input_type="SEARCH_QUERY",
            truncate="END",
            output_dimensions=cfg.embedding_dimension,
            is_echo=False,
        )
        logger.debug("EmbeddingService._embed_query_sync — EmbedTextDetails built, calling OCI embed_text")
        oci_start = time.perf_counter()
        response = self._client.embed_text(embed_text_details=details)
        oci_elapsed_ms = round((time.perf_counter() - oci_start) * 1000, 2)
        logger.info(
            "EmbeddingService._embed_query_sync — OCI embed_text time taken",
            extra={"status": "success", "response_payload": {"oci_elapsed_ms": oci_elapsed_ms}},
        )
        logger.debug(
            "EmbeddingService._embed_query_sync — OCI embed_text returned",
            extra={"input_data": {
                "has_embeddings": getattr(response.data, "embeddings", None) is not None,
                "has_embeddings_by_type": getattr(response.data, "embeddings_by_type", None) is not None,
                "model_id": getattr(response.data, "model_id", None),
            }},
        )
        vectors = extract_embeddings(response.data)
        logger.debug(f"EmbeddingService._embed_query_sync — extracted {len(vectors)} vector(s)")
        vector = vectors[0]
        logger.debug(
            "EmbeddingService._embed_query_sync — vector ready",
            extra={"input_data": {"dim": len(vector), "preview": [round(v, 5) for v in vector[:5]]}},
        )
        if cfg.embedding_normalize:
            vector = self._l2_normalize(vector)
            logger.debug("EmbeddingService._embed_query_sync — vector L2-normalized")
        return vector
 
    async def encode_query(self, query: str) -> list[float]:
        """Embed a single search query (SEARCH_QUERY input type)."""
        logger.debug(
            "EmbeddingService.encode_query — starting",
            extra={"input_data": {"query_preview": query[:120], "query_len": len(query), "client_loaded": self._client is not None}},
        )
        try:
            cache_key = (
                query.strip().lower(),
                self._settings.embedding_dimension,
                self._settings.embedding_normalize,
            )
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                logger.info(
                    "EmbeddingService.encode_query — cache hit (OCI call skipped)",
                    extra={"status": "success", "input_data": {"output_dim": len(cached)}},
                )
                return cached
 
            if self._client is None:
                logger.warning("EmbeddingService.encode_query — client not initialized, triggering load")
                await self.load()
 
            logger.debug("EmbeddingService.encode_query — dispatching embed to executor thread")
            loop = asyncio.get_running_loop()
            vector = await loop.run_in_executor(None, self._embed_query_sync, query)
 
            self._cache[cache_key] = vector
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
            logger.info(
                "EmbeddingService.encode_query — done",
                extra={
                    "status": "success",
                    "input_data": {"normalize": self._settings.embedding_normalize, "output_dim": len(vector)},
                },
            )
            return vector
        except oci.exceptions.ServiceError as exc:
            logger.error(
                "EmbeddingService.encode_query — OCI service error",
                extra={"status": "error", "error_details": str(exc), **oci_error_fields(exc)},
            )
            raise
        except Exception as exc:
            logger.error(
                "EmbeddingService.encode_query — encoding failed",
                extra={"status": "error", "error_details": str(exc), **oci_error_fields(exc)},
            )
            raise

    def _embed_image_sync(self, data_uri: str) -> list[float]:
        cfg = self._settings
        details = oci.generative_ai_inference.models.EmbedTextDetails(
            compartment_id=cfg.oci_compartment_id,
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                serving_type="ON_DEMAND",
                model_id=cfg.embedding_model_name,
            ),
            # Same multimodal path, model and input type the ingest pipeline uses
            # for product images, so the vectors are directly comparable.
            embed_contents=[
                oci.generative_ai_inference.models.EmbedImageContent(
                    image_url=oci.generative_ai_inference.models.ImageUrl(url=data_uri)
                )
            ],
            input_type="IMAGE",
            embedding_types=["float"],
            output_dimensions=cfg.embedding_dimension,
            is_echo=False,
        )
        oci_start = time.perf_counter()
        response = self._client.embed_text(embed_text_details=details)
        oci_elapsed_ms = round((time.perf_counter() - oci_start) * 1000, 2)
        logger.info(
            "EmbeddingService._embed_image_sync — OCI embed_text (IMAGE) time taken",
            extra={"status": "success", "response_payload": {"oci_elapsed_ms": oci_elapsed_ms}},
        )
        vector = extract_embeddings(response.data)[0]
        if cfg.embedding_normalize:
            vector = self._l2_normalize(vector)
        return vector

    async def encode_image(self, data: bytes, mime: str) -> list[float]:
        """Embed an uploaded image (raw bytes) with input_type IMAGE.

        OCI takes images as base64 data URIs, so the binary upload is encoded
        here exactly as the ingest pipeline encodes product images.
        """
        logger.debug(
            "EmbeddingService.encode_image — starting",
            extra={"input_data": {"bytes": len(data), "mime": mime, "client_loaded": self._client is not None}},
        )
        try:
            if self._client is None:
                logger.warning("EmbeddingService.encode_image — client not initialized, triggering load")
                await self.load()
            data_uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            loop = asyncio.get_running_loop()
            vector = await loop.run_in_executor(None, self._embed_image_sync, data_uri)
            logger.info(
                "EmbeddingService.encode_image — done",
                extra={"status": "success", "input_data": {"output_dim": len(vector)}},
            )
            return vector
        except Exception as exc:
            logger.error(
                "EmbeddingService.encode_image — encoding failed",
                extra={"status": "error", "error_details": str(exc), **oci_error_fields(exc)},
            )
            raise


embedding_service = EmbeddingService(_settings)
 