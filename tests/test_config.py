from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import Settings, settings


def test_required_settings_come_from_environment():
    assert settings.opensearch_host == "opensearch.test"
    assert settings.opensearch_port == 9200
    assert settings.embedding_dimension == 4
    assert settings.embedding_normalize is True


def test_defaults_for_optional_settings():
    s = Settings(_env_file=None)
    assert s.opensearch_index_name == "asmo-search-api-index"
    assert s.opensearch_vector_field == "content_vector"
    assert s.default_top_k == 10
    assert s.max_top_k == 100
    assert s.hybrid_retrieval_k == 100
    assert s.knn_ef_search == 500
    assert s.image_min_score == 0.6
    assert s.text_image_image_weight == 0.6
    assert s.redis_port == 443
    assert s.redis_ssl is True
    assert s.redis_collection == "asmo:embeddings"
    assert s.redis_ttl_seconds == 3600
    assert s.api_version == "2.1.0"


def test_env_values_are_type_coerced(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_PORT", "443")
    monkeypatch.setenv("EMBEDDING_NORMALIZE", "false")
    monkeypatch.setenv("HYBRID_RETRIEVAL_K", "250")
    s = Settings(_env_file=None)
    assert s.opensearch_port == 443
    assert s.embedding_normalize is False
    assert s.hybrid_retrieval_k == 250


def test_missing_required_setting_raises(monkeypatch):
    monkeypatch.delenv("OPENSEARCH_HOST")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_unknown_env_keys_are_ignored(monkeypatch):
    monkeypatch.setenv("SOMETHING_UNRELATED", "x")
    Settings(_env_file=None)  # extra="ignore" — must not raise
