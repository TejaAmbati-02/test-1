from __future__ import annotations

import os

import oci

from config import Settings
from utils.logger import logger


def extract_embeddings(result) -> list[list[float]]:
    """Return float embedding vectors from an EmbedTextResult, regardless of SDK field.

    `result.embeddings` is populated for the default float output; when `embedding_types`
    is requested the vectors come back under `result.embeddings_by_type` instead. Handle both.
    """
    fn = "extract_embeddings"
    logger.debug(f"[{fn}] start — result type={type(result).__name__}", extra={"function_name": fn})
    try:
        vecs = getattr(result, "embeddings", None)
        if vecs:
            out = [list(v) for v in vecs]
            logger.debug(
                f"[{fn}] {len(out)} vector(s) from 'embeddings', dim={len(out[0]) if out else 0}",
                extra={"function_name": fn, "status": "success"},
            )
            return out

        by_type = getattr(result, "embeddings_by_type", None)
        if by_type:
            floats = by_type.get("float") if isinstance(by_type, dict) else getattr(by_type, "float", None)
            if floats:
                out = [list(v) for v in floats]
                logger.debug(
                    f"[{fn}] {len(out)} vector(s) from 'embeddings_by_type.float', dim={len(out[0]) if out else 0}",
                    extra={"function_name": fn, "status": "success"},
                )
                return out

        logger.warning(
            f"[{fn}] OCI embed response contained no embeddings "
            f"(has_embeddings={vecs is not None}, has_embeddings_by_type={by_type is not None})",
            extra={"function_name": fn, "status": "error"},
        )
        raise ValueError("OCI embed response contained no embeddings")
    except ValueError:
        raise
    except Exception as exc:
        logger.exception(
            f"[{fn}] failed: {exc}",
            extra={"function_name": fn, "status": "error", "error_details": str(exc)},
        )
        raise


def oci_error_fields(exc: Exception) -> dict:
    """Extract structured OCI error details for logging.

    For an OCI ServiceError this returns the HTTP status, service error code,
    message and request id (as the reference client surfaces them). For any
    other exception it returns the exception type so logs still carry context.
    """
    try:
        if isinstance(exc, oci.exceptions.ServiceError):
            return {
                "oci_status": exc.status,
                "oci_code": exc.code,
                "oci_message": exc.message,
                "oci_request_id": exc.request_id,
            }
        return {"error_type": type(exc).__name__}
    except Exception as inner:
        # Never let error-detail extraction mask the original error.
        logger.warning(
            f"[oci_error_fields] could not extract OCI error fields from {type(exc).__name__}: {inner}",
            extra={"function_name": "oci_error_fields", "status": "error", "error_details": str(inner)},
        )
        return {"error_type": type(exc).__name__}


def build_genai_client(settings: Settings) -> "oci.generative_ai_inference.GenerativeAiInferenceClient":
    """Create an OCI GenAI inference client from the configured profile and endpoint."""
    config_path = settings.oci_config_file
    try:
        config_path = os.path.expanduser(settings.oci_config_file)
        logger.info(
            "build_genai_client — loading OCI config",
            extra={"input_data": {"config_file": config_path, "profile": settings.oci_profile, "region": settings.oci_region}},
        )
        config = oci.config.from_file(file_location=config_path, profile_name=settings.oci_profile)
        config["region"] = settings.oci_region

        logger.debug(
            "build_genai_client — creating GenerativeAiInferenceClient",
            extra={"input_data": {"endpoint": settings.oci_genai_endpoint}},
        )
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=settings.oci_genai_endpoint,
        )
        logger.info(
            "build_genai_client — client ready",
            extra={"status": "success", "input_data": {"endpoint": settings.oci_genai_endpoint}},
        )
        return client
    except oci.exceptions.ConfigFileNotFound as exc:
        logger.error(
            "build_genai_client — OCI config file not found",
            extra={"status": "error", "error_details": str(exc), "input_data": {"config_file": config_path, "profile": settings.oci_profile}},
        )
        raise
    except oci.exceptions.ServiceError as exc:
        logger.error(
            "build_genai_client — OCI service error while creating client",
            extra={"status": "error", "error_details": str(exc), **oci_error_fields(exc)},
        )
        raise
    except Exception as exc:
        logger.exception(
            f"build_genai_client — unexpected error while creating client: {exc}",
            extra={"function_name": "build_genai_client", "status": "error", "error_details": str(exc), **oci_error_fields(exc)},
        )
        raise
