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


def test_properties(test_settings):
    svc = EmbeddingService(test_settings)
    assert svc.is_loaded is False
    assert svc.model_name == "cohere.embed-v4.0"
    svc._client = object()
    assert svc.is_loaded is True


class TestLoad:
    async def test_load_builds_client(self, monkeypatch, test_settings):
        build = MagicMock(return_value="client")
        monkeypatch.setattr(embedder_module, "build_genai_client", build)
        svc = EmbeddingService(test_settings)
        await svc.load()
        assert svc._client == "client"
        build.assert_called_once_with(test_settings)

    async def test_load_is_skipped_when_already_loaded(self, monkeypatch, service):
        build = MagicMock()
        monkeypatch.setattr(embedder_module, "build_genai_client", build)
        await service.load()
        build.assert_not_called()

    async def test_load_failure_propagates(self, monkeypatch, test_settings):
        monkeypatch.setattr(embedder_module, "build_genai_client", MagicMock(side_effect=RuntimeError("no config")))
        svc = EmbeddingService(test_settings)
        with pytest.raises(RuntimeError, match="no config"):
            await svc.load()
        assert svc.is_loaded is False


class TestL2Normalize:
    def test_normalises_to_unit_length(self):
        v = EmbeddingService._l2_normalize([3.0, 4.0])
        assert v == pytest.approx([0.6, 0.8])
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)

    def test_zero_vector_is_returned_unchanged(self):
        assert EmbeddingService._l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class TestEncodeQuery:
    async def test_embeds_and_normalises(self, service, oci_client):
        vector = await service.encode_query("ball valve")
        assert vector == pytest.approx([0.6, 0.8, 0.0, 0.0])
        details = oci_client.embed_text.call_args.kwargs["embed_text_details"]
        assert details.inputs == ["ball valve"]
        assert details.input_type == "SEARCH_QUERY"
        assert details.output_dimensions == 4
        assert details.compartment_id == "ocid1.compartment.test"
        assert details.serving_mode.model_id == "cohere.embed-v4.0"

    async def test_without_normalisation(self, test_settings, oci_client):
        svc = EmbeddingService(test_settings.model_copy(update={"embedding_normalize": False}))
        svc._client = oci_client
        assert await svc.encode_query("x") == [3.0, 4.0, 0.0, 0.0]

    async def test_cache_hit_skips_oci(self, service, oci_client):
        first = await service.encode_query("Ball Valve")
        second = await service.encode_query("  ball valve ")  # same after strip + lower
        assert first == second
        assert oci_client.embed_text.call_count == 1

    async def test_lru_cache_evicts_oldest(self, service, oci_client):
        service._cache_max = 2
        await service.encode_query("a")
        await service.encode_query("b")
        await service.encode_query("a")  # refresh "a"; "b" is now the oldest
        await service.encode_query("c")  # evicts "b"
        assert len(service._cache) == 2
        keys = [k[0] for k in service._cache]
        assert keys == ["a", "c"]
        await service.encode_query("b")
        assert oci_client.embed_text.call_count == 4

    async def test_lazy_loads_client(self, monkeypatch, test_settings, oci_client):
        monkeypatch.setattr(embedder_module, "build_genai_client", MagicMock(return_value=oci_client))
        svc = EmbeddingService(test_settings)
        await svc.encode_query("q")
        assert svc.is_loaded

    async def test_oci_service_error_propagates(self, service, oci_client):
        oci_client.embed_text.side_effect = oci.exceptions.ServiceError(500, "Internal", {}, "down")
        with pytest.raises(oci.exceptions.ServiceError):
            await service.encode_query("q")
        assert service._cache == {}

    async def test_other_errors_propagate(self, service, oci_client):
        oci_client.embed_text.return_value = SimpleNamespace(data=SimpleNamespace(embeddings=None, embeddings_by_type=None))
        with pytest.raises(ValueError):
            await service.encode_query("q")


class TestEncodeImage:
    async def test_sends_base64_data_uri(self, service, oci_client):
        data = b"\x89PNGdata"
        vector = await service.encode_image(data, "image/png")
        assert vector == pytest.approx([0.6, 0.8, 0.0, 0.0])
        details = oci_client.embed_text.call_args.kwargs["embed_text_details"]
        assert details.input_type == "IMAGE"
        assert details.embedding_types == ["float"]
        url = details.embed_contents[0].image_url.url
        assert url == "data:image/png;base64," + base64.b64encode(data).decode()

    async def test_image_is_not_cached(self, service, oci_client):
        await service.encode_image(b"x", "image/png")
        await service.encode_image(b"x", "image/png")
        assert oci_client.embed_text.call_count == 2

    async def test_lazy_loads_client(self, monkeypatch, test_settings, oci_client):
        monkeypatch.setattr(embedder_module, "build_genai_client", MagicMock(return_value=oci_client))
        svc = EmbeddingService(test_settings)
        await svc.encode_image(b"x", "image/jpeg")
        assert svc.is_loaded

    async def test_errors_propagate(self, service, oci_client):
        oci_client.embed_text.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.encode_image(b"x", "image/png")
