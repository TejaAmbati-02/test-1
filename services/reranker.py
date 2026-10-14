from __future__ import annotations

import asyncio
import math

import oci

from config import Settings, settings as _settings
from services.oci_genai import build_genai_client, oci_error_fields
from utils.logger import logger
from models.response import SearchHit

# Cohere Embed v4.0 accepts at most 96 inputs per request.
_MAX_BATCH = 96


class RerankerService:
    """Embedding-based reranking via OCI GenAI (Cohere Embed v4.0, cosine similarity).

    The query and each candidate are embedded (SEARCH_QUERY / SEARCH_DOCUMENT) and the
    candidates are reordered by cosine similarity to the query. No local model is used.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: "oci.generative_ai_inference.GenerativeAiInferenceClient | None" = None

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    async def load(self) -> None:
        if self._client is not None:
            logger.debug("RerankerService.load — client already initialized, skipping")
            return
        cfg = self._settings
        logger.info(
            "RerankerService.load — starting",
            extra={"input_data": {"model": cfg.reranker_model_name, "region": cfg.oci_region}},
        )
        try:
            loop = asyncio.get_running_loop()
            self._client = await loop.run_in_executor(None, build_genai_client, cfg)
            logger.info(
                "RerankerService.load — OCI GenAI client ready",
                extra={"status": "success", "input_data": {"model": cfg.reranker_model_name}},
            )
        except Exception as exc:
            logger.error(
                "RerankerService.load — failed to initialize OCI GenAI client",
                extra={"status": "error", "error_details": str(exc), "input_data": {"model": cfg.reranker_model_name}, **oci_error_fields(exc)},
            )
            raise

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _embed_sync(self, texts: list[str], input_type: str) -> list[list[float]]:
        cfg = self._settings
        logger.debug(
            "RerankerService._embed_sync — embedding texts",
            extra={"input_data": {"count": len(texts), "input_type": input_type, "model": cfg.reranker_model_name}},
        )
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            batch = texts[start:start + _MAX_BATCH]
            details = oci.generative_ai_inference.models.EmbedTextDetails(
                compartment_id=cfg.oci_compartment_id,
                serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                    serving_type="ON_DEMAND",
                    model_id=cfg.reranker_model_name,
                ),
                inputs=batch,
                input_type=input_type,
                truncate="END",
                embedding_types=["float"],
                output_dimensions=cfg.embedding_dimension,
                is_echo=False,
            )
            response = self._client.embed_text(embed_text_details=details)
            embeddings.extend(list(vec) for vec in response.data.embeddings)
        logger.debug(f"RerankerService._embed_sync — produced {len(embeddings)} embeddings")
        return embeddings

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int, text_field: str = "content") -> list[SearchHit]:
        logger.info(
            "RerankerService.rerank — starting",
            extra={"input_data": {"candidates": len(hits), "top_k": top_k, "text_field": text_field}},
        )
        try:
            if not hits:
                logger.info("RerankerService.rerank — no candidates, returning empty list")
                return []

            if self._client is None:
                logger.warning("RerankerService.rerank — client not initialized, triggering load")
                await self.load()

            logger.debug("RerankerService.rerank — building candidate document texts")
            documents = [
                hit.source.get(text_field) or " ".join(str(v) for v in hit.source.values())
                for hit in hits
            ]

            logger.info(f"RerankerService.rerank — embedding query + {len(documents)} candidates via OCI")
            loop = asyncio.get_running_loop()
            query_vec = (await loop.run_in_executor(None, self._embed_sync, [query], "SEARCH_QUERY"))[0]
            doc_vecs = await loop.run_in_executor(None, self._embed_sync, documents, "SEARCH_DOCUMENT")

            logger.debug("RerankerService.rerank — scoring candidates by cosine similarity")
            scored = (
                SearchHit(**{**h.model_dump(), "score": self._cosine(query_vec, dv)})
                for h, dv in zip(hits, doc_vecs)
            )
            ranked = sorted(scored, key=lambda h: h.score, reverse=True)
            result = ranked[:top_k]
            logger.info(
                "RerankerService.rerank — done",
                extra={
                    "status": "success",
                    "input_data": {"candidates_in": len(hits), "top_k": top_k},
                    "response_payload": {
                        "returned": len(result),
                        "best_score": round(result[0].score, 4) if result else None,
                        "worst_score": round(result[-1].score, 4) if result else None,
                    },
                },
            )
            return result
        except oci.exceptions.ServiceError as exc:
            logger.error(
                "RerankerService.rerank — OCI service error",
                extra={"status": "error", "error_details": str(exc), "input_data": {"candidates": len(hits), "top_k": top_k}, **oci_error_fields(exc)},
            )
            raise
        except Exception as exc:
            logger.error(
                "RerankerService.rerank — failed",
                extra={"status": "error", "error_details": str(exc), "input_data": {"candidates": len(hits), "top_k": top_k}, **oci_error_fields(exc)},
            )
            raise


reranker_service = RerankerService(_settings)
