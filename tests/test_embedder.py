from __future__ import annotations

import base64
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import oci
import pytest

from services import embedder as embedder_module
from services.embedder import EmbeddingService


def _embed_response(vector):
    return SimpleNamespace(data=SimpleNamespace(embeddings=[vector], embeddings_by_type=None, model_id="m"))


@pytest.fixture
def oci_client():
    client = MagicMock()
    client.embed_text.return_value = _embed_response([3.0, 4.0, 0.0, 0.0])
    return client


@pytest.fixture
def service(test_settings, oci_client):
    svc = EmbeddingService(test_settings)
    svc._client = oci_client
    return svc


@pytest.fixture
def oci_sdk(monkeypatch, oci_client):
    """Stand-ins for oci.config.from_file and the GenAI client class."""
    from_file = MagicMock(return_value={"tenancy": "t"})
    client_cls = MagicMock(return_value=oci_client)
    monkeypatch.setattr(embedder_module.oci.config, "from_file", from_file)
    monkeypatch.setattr(embedder_module.oci.generative_ai_inference, "GenerativeAiInferenceClient", client_cls)
    return SimpleNamespace(from_file=from_file, client_cls=client_cls)


def test_properties(test_settings):
    svc = EmbeddingService(test_settings)
    assert svc.is_loaded is False
    assert svc.model_name == "cohere.embed-v4.0"
    svc._client = object()
    assert svc.is_loaded is True


class TestLoad:
    async def test_load_builds_client_from_profile(self, test_settings, oci_sdk, oci_client):
        svc = EmbeddingService(test_settings)
        await svc.load()
        assert svc._client is oci_client
        kwargs = oci_sdk.from_file.call_args.kwargs
        assert kwargs["profile_name"] == "DEFAULT"
        assert not kwargs["file_location"].startswith("~")  # ~ is expanded
        client_kwargs = oci_sdk.client_cls.call_args.kwargs
        assert client_kwargs["config"] == {"tenancy": "t", "region": "us-chicago-1"}
        assert client_kwargs["service_endpoint"] == "https://genai.test"

    async def test_load_is_skipped_when_already_loaded(self, service, oci_sdk):
        await service.load()
        oci_sdk.from_file.assert_not_called()

    async def test_load_failure_propagates(self, test_settings, oci_sdk):
        oci_sdk.from_file.side_effect = oci.exceptions.ConfigFileNotFound("no config")
        svc = EmbeddingService(test_settings)
        with pytest.raises(oci.exceptions.ConfigFileNotFound):
            await svc.load()
        assert svc.is_loaded is False


class TestEncodeQuery:
    async def test_embeds_and_normalises(self, service, oci_client):
        vector = await service.encode_query("ball valve")
        assert vector == pytest.approx([0.6, 0.8, 0.0, 0.0])
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)
        details = oci_client.embed_text.call_args.kwargs["embed_text_details"]
        assert details.inputs == ["ball valve"]
        assert details.input_type == "SEARCH_QUERY"
        assert details.truncate == "END"
        assert details.output_dimensions == 4
        assert details.compartment_id == "ocid1.compartment.test"
        assert details.serving_mode.model_id == "cohere.embed-v4.0"

    async def test_without_normalisation(self, test_settings, oci_client):
        svc = EmbeddingService(test_settings.model_copy(update={"embedding_normalize": False}))
        svc._client = oci_client
        assert await svc.encode_query("x") == [3.0, 4.0, 0.0, 0.0]

    async def test_zero_vector_is_left_unnormalised(self, service, oci_client):
        oci_client.embed_text.return_value = _embed_response([0.0, 0.0, 0.0, 0.0])
        assert await service.encode_query("x") == [0.0, 0.0, 0.0, 0.0]

    async def test_reads_embeddings_by_type(self, service, oci_client):
        for by_type in ({"float": [[0.0, 2.0, 0.0, 0.0]]}, SimpleNamespace(float=[[0.0, 2.0, 0.0, 0.0]])):
            oci_client.embed_text.return_value = SimpleNamespace(data=SimpleNamespace(embeddings=None, embeddings_by_type=by_type))
            service._cache.clear()
            assert await service.encode_query("x") == pytest.approx([0.0, 1.0, 0.0, 0.0])

    async def test_cache_hit_skips_oci(self, service, oci_client):
        first = await service.encode_query("Ball Valve")
        second = await service.encode_query("  ball valve ")  # same after strip + lower
        assert first == second
        assert oci_client.embed_text.call_count == 1

    async def test_lru_cache_evicts_oldest(self, service, oci_client, monkeypatch):
        monkeypatch.setattr(embedder_module, "QUERY_CACHE_SIZE", 2)
        await service.encode_query("a")
        await service.encode_query("b")
        await service.encode_query("a")  # refresh "a"; "b" is now the oldest
        await service.encode_query("c")  # evicts "b"
        assert [k[0] for k in service._cache] == ["a", "c"]
        await service.encode_query("b")
        assert oci_client.embed_text.call_count == 4

    async def test_lazy_loads_client(self, test_settings, oci_sdk):
        svc = EmbeddingService(test_settings)
        await svc.encode_query("q")
        assert svc.is_loaded

    async def test_oci_service_error_propagates_and_is_not_cached(self, service, oci_client):
        oci_client.embed_text.side_effect = oci.exceptions.ServiceError(500, "Internal", {}, "down")
        with pytest.raises(oci.exceptions.ServiceError):
            await service.encode_query("q")
        assert len(service._cache) == 0

    async def test_empty_response_raises(self, service, oci_client):
        oci_client.embed_text.return_value = SimpleNamespace(data=SimpleNamespace(embeddings=None, embeddings_by_type=None))
        with pytest.raises(ValueError, match="no embeddings"):
            await service.encode_query("q")
        assert service._cache == {}

    async def test_cache_key_is_normalised_query_dimension_and_normalize(self, service):
        await service.encode_query("  Ball Valve ")
        assert list(service._cache) == [("ball valve", 4, True)]

    async def test_oci_receives_the_original_query_text(self, service, oci_client):
        await service.encode_query("  Ball Valve ")
        details = oci_client.embed_text.call_args.kwargs["embed_text_details"]
        assert details.inputs == ["  Ball Valve "]
        assert details.truncate == "END"
        assert details.is_echo is False
        assert details.serving_mode.serving_type == "ON_DEMAND"

    async def test_cache_hit_does_not_need_a_client(self, service, oci_sdk):
        await service.encode_query("valve")
        service._client = None
        assert await service.encode_query("VALVE") == pytest.approx([0.6, 0.8, 0.0, 0.0])
        oci_sdk.client_cls.assert_not_called()
        assert service.is_loaded is False

    async def test_client_built_once_across_calls(self, test_settings, oci_sdk, oci_client):
        svc = EmbeddingService(test_settings)
        await svc.encode_query("a")
        await svc.encode_query("b")
        oci_sdk.client_cls.assert_called_once()
        oci_sdk.from_file.assert_called_once()
        assert oci_client.embed_text.call_count == 2

    async def test_lazy_load_failure_propagates(self, test_settings, oci_sdk):
        oci_sdk.client_cls.side_effect = RuntimeError("no config")
        svc = EmbeddingService(test_settings)
        with pytest.raises(RuntimeError, match="no config"):
            await svc.encode_query("q")
        assert svc._cache == {}
        assert svc.is_loaded is False

    async def test_vectors_from_embeddings_by_type(self, service, oci_client):
        oci_client.embed_text.return_value = SimpleNamespace(
            data=SimpleNamespace(embeddings=None, embeddings_by_type={"float": [[0.0, 2.0, 0.0, 0.0]]})
        )
        assert await service.encode_query("q") == pytest.approx([0.0, 1.0, 0.0, 0.0])

    def test_default_cache_size(self):
        assert embedder_module.QUERY_CACHE_SIZE == 2048

    async def test_service_error_is_logged_with_oci_fields(self, service, oci_client, monkeypatch):
        error = MagicMock()
        monkeypatch.setattr(embedder_module.logger, "error", error)
        oci_client.embed_text.side_effect = oci.exceptions.ServiceError(
            429, "TooManyRequests", {"opc-request-id": "req-9"}, "slow down")
        with pytest.raises(oci.exceptions.ServiceError):
            await service.encode_query("q")
        extra = error.call_args.kwargs["extra"]
        assert (extra["oci_status"], extra["oci_code"], extra["oci_message"]) == (429, "TooManyRequests", "slow down")
        assert extra["status"] == "error"

    async def test_non_oci_error_is_logged_without_oci_fields(self, service, oci_client, monkeypatch):
        error = MagicMock()
        monkeypatch.setattr(embedder_module.logger, "error", error)
        oci_client.embed_text.side_effect = TimeoutError("slow")
        with pytest.raises(TimeoutError):
            await service.encode_query("q")
        assert "oci_status" not in error.call_args.kwargs["extra"]


class TestEncodeImage:
    async def test_sends_base64_data_uri(self, service, oci_client):
        data = b"\x89PNGdata"
        vector = await service.encode_image(data, "image/png")
        assert vector == pytest.approx([0.6, 0.8, 0.0, 0.0])
        details = oci_client.embed_text.call_args.kwargs["embed_text_details"]
        assert details.input_type == "IMAGE"
        assert details.embedding_types == ["float"]
        assert details.embed_contents[0].image_url.url == "data:image/png;base64," + base64.b64encode(data).decode()

    async def test_image_is_not_cached(self, service, oci_client):
        await service.encode_image(b"x", "image/png")
        await service.encode_image(b"x", "image/png")
        assert oci_client.embed_text.call_count == 2

    async def test_lazy_loads_client(self, test_settings, oci_sdk):
        svc = EmbeddingService(test_settings)
        await svc.encode_image(b"x", "image/jpeg")
        assert svc.is_loaded

    async def test_errors_propagate(self, service, oci_client):
        oci_client.embed_text.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.encode_image(b"x", "image/png")

    async def test_oci_service_error_propagates(self, service, oci_client):
        oci_client.embed_text.side_effect = oci.exceptions.ServiceError(429, "TooManyRequests", {}, "slow down")
        with pytest.raises(oci.exceptions.ServiceError):
            await service.encode_image(b"x", "image/png")

    async def test_request_details(self, service, oci_client):
        await service.encode_image(b"x", "image/webp")
        details = oci_client.embed_text.call_args.kwargs["embed_text_details"]
        assert details.output_dimensions == 4
        assert details.compartment_id == "ocid1.compartment.test"
        assert details.serving_mode.model_id == "cohere.embed-v4.0"
        assert details.is_echo is False
        assert details.embed_contents[0].image_url.url.startswith("data:image/webp;base64,")

    async def test_without_normalisation(self, test_settings, oci_client):
        svc = EmbeddingService(test_settings.model_copy(update={"embedding_normalize": False}))
        svc._client = oci_client
        assert await svc.encode_image(b"x", "image/png") == [3.0, 4.0, 0.0, 0.0]

    async def test_vectors_from_embeddings_by_type(self, service, oci_client):
        oci_client.embed_text.return_value = SimpleNamespace(
            data=SimpleNamespace(embeddings=None, embeddings_by_type={"float": [[0.0, 0.0, 5.0, 0.0]]})
        )
        assert await service.encode_image(b"x", "image/png") == pytest.approx([0.0, 0.0, 1.0, 0.0])

    async def test_no_embeddings_raises(self, service, oci_client):
        oci_client.embed_text.return_value = SimpleNamespace(data=SimpleNamespace(embeddings=[], embeddings_by_type=None))
        with pytest.raises(ValueError, match="no embeddings"):
            await service.encode_image(b"x", "image/png")
