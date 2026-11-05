from __future__ import annotations

import importlib.util
import logging
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
    assert s.text_image_image_weight == 0.4
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
# Module import: settings are built (and validated) once, at import time
# --------------------------------------------------------------------------


@pytest.fixture
def fresh_config(monkeypatch, tmp_path):
    """Import a private copy of config.py, so the shared ``config.settings`` is untouched.

    Runs from an empty directory so a local .env can't supply values, and spies on
    the shared "app" logger that config.py logs to.
    """
    monkeypatch.chdir(tmp_path)
    log_error = MagicMock()
    monkeypatch.setattr(logging.getLogger("app"), "error", log_error)

    def load():
        spec = importlib.util.spec_from_file_location("config_under_test", config.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    load.log_error = log_error
    return load


def test_module_builds_settings_on_import(fresh_config):
    module = fresh_config()
    assert isinstance(module.settings, module.Settings)
    assert module.settings.opensearch_host == "opensearch.test"
    fresh_config.log_error.assert_not_called()


def test_module_logger_is_the_shared_app_logger():
    assert config.logger is logging.getLogger("app")


def test_invalid_settings_log_field_names_not_values_and_reraise(monkeypatch, fresh_config):
    monkeypatch.delenv("OPENSEARCH_HOST")
    monkeypatch.setenv("OPENSEARCH_PORT", "not-a-port-hunter2")
    with pytest.raises(ValidationError):
        fresh_config()
    fresh_config.log_error.assert_called_once()
    message = fresh_config.log_error.call_args.args[0]
    assert "(2)" in message  # error count
    assert "opensearch_host: missing" in message
    assert "opensearch_port: int_parsing" in message
    assert "hunter2" not in message
    extra = fresh_config.log_error.call_args.kwargs["extra"]
    assert "hunter2" not in extra["error_details"]
    assert extra["status"] == "error"
    assert extra["function_name"] == "config"


def test_secrets_are_never_logged_on_validation_error(monkeypatch, fresh_config):
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "s3cr3t-value")
    monkeypatch.delenv("OCI_PROFILE")
    with pytest.raises(ValidationError):
        fresh_config()
    logged = str(fresh_config.log_error.call_args)
    assert "s3cr3t-value" not in logged
    assert "ocid1.compartment.test" not in logged
    assert "oci_profile: missing" in logged
