from __future__ import annotations

import asyncio
import math

import oci

from config import Settings, settings as _settings
from services.oci_genai import build_genai_client, oci_error_fields
from utils.logger import logger


class EmbeddingService:
    """Search-query embeddings backed by OCI GenAI (Cohere Embed v4.0)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: "oci.generative_ai_inference.GenerativeAiInferenceClient | None" = None

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
            extra={"input_data": {"model": cfg.embedding_model_name, "dimension": cfg.embedding_dimension, "input_type": "SEARCH_QUERY"}},
        )
        details = oci.generative_ai_inference.models.EmbedTextDetails(
            compartment_id=cfg.oci_compartment_id,
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                serving_type="ON_DEMAND",
                model_id=cfg.embedding_model_name,
            ),
            inputs=[query],
            input_type="SEARCH_QUERY",
            truncate="NONE",
            embedding_types=["float"],
            output_dimensions=cfg.embedding_dimension,
            is_echo=False,
        )
        logger.debug("EmbeddingService._embed_query_sync — calling OCI embed_text")
        response = self._client.embed_text(embed_text_details=details)
        vector = list(response.data.embeddings[0])
        logger.debug(f"EmbeddingService._embed_query_sync — received vector dim={len(vector)}")
        if cfg.embedding_normalize:
            vector = self._l2_normalize(vector)
            logger.debug("EmbeddingService._embed_query_sync — vector L2-normalized")
        return vector

    async def encode_query(self, query: str) -> list[float]:
        """Embed a single search query (SEARCH_QUERY input type)."""
        logger.debug("EmbeddingService.encode_query — starting")
        try:
            if self._client is None:
                logger.warning("EmbeddingService.encode_query — client not initialized, triggering load")
                await self.load()

            logger.debug("EmbeddingService.encode_query — dispatching embed to executor")
            loop = asyncio.get_running_loop()
            vector = await loop.run_in_executor(None, self._embed_query_sync, query)
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


embedding_service = EmbeddingService(_settings)
