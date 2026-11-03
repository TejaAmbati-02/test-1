from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import config
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
    assert s.image_max_pixels == 2_400_000
    assert s.image_jpeg_quality == 90
    assert s.redis_enabled is True
    assert s.redis_port == 443
    assert s.redis_ssl is True
    assert s.redis_ttl_seconds == 3600
    assert s.api_version == "2.1.0"


def test_image_settings_can_be_overridden(monkeypatch):
    monkeypatch.setenv("IMAGE_MAX_PIXELS", "1000000")
    monkeypatch.setenv("IMAGE_JPEG_QUALITY", "75")
    s = Settings(_env_file=None)
    assert s.image_max_pixels == 1_000_000
    assert s.image_jpeg_quality == 75


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


# --------------------------------------------------------------------------
# _load_settings
# --------------------------------------------------------------------------


@pytest.fixture
def no_env_file(monkeypatch):
    """Make _load_settings build Settings without reading a local .env."""
    real = Settings
    monkeypatch.setattr(config, "Settings", lambda: real(_env_file=None))


@pytest.fixture
def config_logger(monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(config, "logger", log)
    return log


def test_load_settings_returns_settings(no_env_file, config_logger):
    loaded = config._load_settings()
    assert isinstance(loaded, Settings)
    assert loaded.opensearch_host == "opensearch.test"
    config_logger.error.assert_not_called()
    config_logger.exception.assert_not_called()


def test_load_settings_does_not_log_secrets(no_env_file, config_logger):
    config._load_settings()
    logged = " ".join(str(c) for c in config_logger.debug.call_args_list)
    assert "secret" not in logged  # OPENSEARCH_PASSWORD / KEYCLOAK_CLIENT_SECRET values
    assert "ocid1.compartment.test" not in logged


def test_load_settings_logs_field_names_not_values_on_validation_error(monkeypatch, no_env_file, config_logger):
    monkeypatch.delenv("OPENSEARCH_HOST")
    monkeypatch.setenv("OPENSEARCH_PORT", "not-a-port-hunter2")
    with pytest.raises(ValidationError):
        config._load_settings()
    config_logger.error.assert_called_once()
    message = config_logger.error.call_args.args[0]
    assert "(2)" in message  # error count
    assert "opensearch_host: missing" in message
    assert "opensearch_port: int_parsing" in message
    assert "hunter2" not in message
    details = config_logger.error.call_args.kwargs["extra"]["error_details"]
    assert "hunter2" not in details
    assert config_logger.error.call_args.kwargs["extra"]["status"] == "error"


def test_load_settings_reraises_unexpected_errors(monkeypatch, config_logger):
    monkeypatch.setattr(config, "Settings", MagicMock(side_effect=OSError("cannot read .env")))
    with pytest.raises(OSError, match="cannot read .env"):
        config._load_settings()
    config_logger.exception.assert_called_once()
    config_logger.error.assert_not_called()
    assert "OSError" in config_logger.exception.call_args.args[0]
