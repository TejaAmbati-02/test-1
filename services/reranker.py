from __future__ import annotations

import asyncio
import os

from sentence_transformers import CrossEncoder

from config import Settings, settings as _settings
from utils.logger import logger
from models.response import SearchHit


class RerankerService:
    """Candidate reranking backed by a local cross-encoder (sentence-transformers)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: CrossEncoder | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def load(self) -> None:
        if self._model is not None:
            logger.debug("RerankerService.load — model already loaded, skipping")
            return
        cfg = self._settings
        logger.info(
            "RerankerService.load — starting",
            extra={"input_data": {"model": cfg.reranker_model_name, "device": cfg.reranker_device}},
        )
        try:
            cache_dir = os.path.expanduser(cfg.reranker_cache_dir) if cfg.reranker_cache_dir else None
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(cfg.reranker_model_name, device=cfg.reranker_device, cache_dir=cache_dir),
            )
            logger.info(
                "RerankerService.load — model loaded successfully",
                extra={"status": "success", "input_data": {"model": cfg.reranker_model_name}},
            )
        except Exception as exc:
            logger.error(
                "RerankerService.load — failed to load model",
                extra={"status": "error", "error_details": str(exc), "input_data": {"model": cfg.reranker_model_name}},
            )
            raise

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        logger.debug(f"RerankerService._predict_sync — scoring {len(pairs)} query-passage pairs")
        scores = self._model.predict(pairs).tolist()
        logger.debug("RerankerService._predict_sync — scoring complete")
        return scores

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int, text_field: str = "content") -> list[SearchHit]:
        logger.info(
            "RerankerService.rerank — starting",
            extra={"input_data": {"candidates": len(hits), "top_k": top_k, "text_field": text_field}},
        )
        try:
            if not hits:
                logger.info("RerankerService.rerank — no candidates, returning empty list")
                return []

            if self._model is None:
                logger.warning("RerankerService.rerank — model not loaded, triggering load")
                await self.load()

            logger.debug("RerankerService.rerank — building query-passage pairs for cross-encoder")
            pairs = [
                (query, hit.source.get(text_field) or " ".join(str(v) for v in hit.source.values()))
                for hit in hits
            ]

            logger.info(f"RerankerService.rerank — running cross-encoder inference on {len(pairs)} pairs")
            loop = asyncio.get_running_loop()
            scores: list[float] = await loop.run_in_executor(None, self._predict_sync, pairs)
            logger.debug("RerankerService.rerank — inference done, sorting by score")

            ranked = sorted(
                [SearchHit(**{**h.model_dump(), "score": s}) for h, s in zip(hits, scores)],
                key=lambda h: h.score,
                reverse=True,
            )
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
        except Exception as exc:
            logger.error(
                "RerankerService.rerank — failed",
                extra={"status": "error", "error_details": str(exc), "input_data": {"candidates": len(hits), "top_k": top_k}},
            )
            raise


reranker_service = RerankerService(_settings)
