"""Shared fixtures for the ASMO Search API unit tests.

The required settings are set here, before any application module is imported,
so ``config.settings`` can be built without a real ``.env``. Every external
dependency (OpenSearch, OCI GenAI, Redis) is mocked — no network is touched.
"""
from __future__ import annotations

import os

_TEST_ENV = {
    "OPENSEARCH_HOST": "opensearch.test",
    "OPENSEARCH_PORT": "9200",
    "OPENSEARCH_USER": "admin",
    "OPENSEARCH_PASSWORD": "secret",
    "OPENSEARCH_TIMEOUT": "30",
    "OCI_CONFIG_FILE": "~/.oci/config",
    "OCI_PROFILE": "DEFAULT",
    "OCI_COMPARTMENT_ID": "ocid1.compartment.test",
    "OCI_REGION": "us-chicago-1",
    "OCI_GENAI_ENDPOINT": "https://genai.test",
    "EMBEDDING_MODEL_NAME": "cohere.embed-v4.0",
    "EMBEDDING_DIMENSION": "4",
    "EMBEDDING_NORMALIZE": "true",
    # Optional settings are pinned to their defaults so a local .env cannot
    # change what the tests expect (environment variables win over .env).
    "OPENSEARCH_INDEX_NAME": "asmo-search-api-index",
    "OPENSEARCH_VECTOR_FIELD": "content_vector",
    "DEFAULT_TOP_K": "10",
    "MAX_TOP_K": "100",
    "HYBRID_SEMANTIC_WEIGHT": "0.5",
    "HYBRID_KEYWORD_WEIGHT": "0.5",
    "HYBRID_RETRIEVAL_K": "100",
    "KNN_EF_SEARCH": "500",
    "IMAGE_MIN_SCORE": "0.6",
    "TEXT_IMAGE_IMAGE_WEIGHT": "0.6",
    # Redis off by default; tests that need it enable it explicitly.
    "REDIS_ENABLED": "true",
    "REDIS_HOST": "",
    "REDIS_PORT": "443",
    "REDIS_USERNAME": "default",
    "REDIS_PASSWORD": "",
    "REDIS_DB": "0",
    "REDIS_SSL": "true",
    "REDIS_SSL_CERT_REQS": "none",
    "REDIS_TIMEOUT": "10",
    "REDIS_COLLECTION": "asmo:embeddings",
    "REDIS_TTL_SECONDS": "3600",
    "LOG_LEVEL": "WARNING",
    "LOG_TO_FILE": "false",
    "API_TITLE": "ASMO AI Search API",
    "API_VERSION": "2.1.0",
    "OCI_EMBED_HEARTBEAT_INTERVAL": "60",
    # Endpoint tests run without Keycloak; tests/test_keycloak_auth.py covers auth.
    "KEYCLOAK_AUTH_ENABLED": "false",
    "KEYCLOAK_SERVER_URL": "https://keycloak.test",
    "KEYCLOAK_REALM": "asmo",
    "KEYCLOAK_CLIENT_ID": "ai-app",
    "KEYCLOAK_CLIENT_SECRET": "test-secret",
}
os.environ.update(_TEST_ENV)

from typing import Any  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from config import settings  # noqa: E402
from models.response import SearchApiResponse  # noqa: E402

# Minimal valid magic-byte headers for each supported image type.
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32
GIF_BYTES = b"GIF89a" + b"\x00" * 32
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 32


@pytest.fixture
def test_settings():
    """A private copy of the settings that a test can change freely."""
    return settings.model_copy()


@pytest.fixture
def client() -> TestClient:
    """API client without the lifespan (startup never contacts real services)."""
    from app import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def enable_redis(monkeypatch):
    """Turn the Redis embedding cache on for the shared ``embedding_store``."""
    from services.embedding_store import embedding_store

    monkeypatch.setattr(
        embedding_store, "_settings", settings.model_copy(update={"redis_host": "redis.test"})
    )
    return embedding_store


def make_search_response(**overrides: Any) -> SearchApiResponse:
    data: dict[str, Any] = {
        "request_id": "req-1",
        "total_results": 1,
        "page": 1,
        "page_size": 60,
        "products": [{"sku_id": "SKU-1", "product_name": "Valve"}],
    }
    data.update(overrides)
    return SearchApiResponse(**data)


def os_search_result(hits: list[dict] | None = None, total: int | None = None, aggs: dict | None = None) -> dict:
    """A raw OpenSearch search response."""
    hits = hits or []
    result: dict[str, Any] = {
        "took": 5,
        "hits": {"total": {"value": len(hits) if total is None else total}, "hits": hits},
    }
    if aggs is not None:
        result["aggregations"] = aggs
    return result
