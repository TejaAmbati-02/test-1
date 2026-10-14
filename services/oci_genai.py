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
    vecs = getattr(result, "embeddings", None)
    if vecs:
        return [list(v) for v in vecs]

    by_type = getattr(result, "embeddings_by_type", None)
    if by_type:
        floats = by_type.get("float") if isinstance(by_type, dict) else getattr(by_type, "float", None)
        if floats:
            return [list(v) for v in floats]

    raise ValueError("OCI embed response contained no embeddings")


def oci_error_fields(exc: Exception) -> dict:
    """Extract structured OCI error details for logging.

    For an OCI ServiceError this returns the HTTP status, service error code,
    message and request id (as the reference client surfaces them). For any
    other exception it returns the exception type so logs still carry context.
    """
    if isinstance(exc, oci.exceptions.ServiceError):
        return {
            "oci_status": exc.status,
            "oci_code": exc.code,
            "oci_message": exc.message,
            "oci_request_id": exc.request_id,
        }
    return {"error_type": type(exc).__name__}


def build_genai_client(settings: Settings) -> "oci.generative_ai_inference.GenerativeAiInferenceClient":
    """Create an OCI GenAI inference client from the configured profile and endpoint."""
    config_path = os.path.expanduser(settings.oci_config_file)
    logger.info(
        "build_genai_client — loading OCI config",
        extra={"input_data": {"config_file": config_path, "profile": settings.oci_profile, "region": settings.oci_region}},
    )
    try:
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
        logger.error(
            "build_genai_client — unexpected error while creating client",
            extra={"status": "error", "error_details": str(exc), **oci_error_fields(exc)},
        )
        raise
