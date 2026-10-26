from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import oci
import pytest

from services import oci_genai
from services.oci_genai import build_genai_client, extract_embeddings, oci_error_fields


def _service_error() -> oci.exceptions.ServiceError:
    return oci.exceptions.ServiceError(429, "TooManyRequests", {"opc-request-id": "rid-1"}, "slow down")


class TestExtractEmbeddings:
    def test_reads_plain_embeddings(self):
        result = SimpleNamespace(embeddings=[(1.0, 2.0)], embeddings_by_type=None)
        assert extract_embeddings(result) == [[1.0, 2.0]]

    def test_reads_embeddings_by_type_dict(self):
        result = SimpleNamespace(embeddings=None, embeddings_by_type={"float": [[0.1, 0.2]]})
        assert extract_embeddings(result) == [[0.1, 0.2]]

    def test_reads_embeddings_by_type_object(self):
        result = SimpleNamespace(embeddings=[], embeddings_by_type=SimpleNamespace(float=[[3.0]]))
        assert extract_embeddings(result) == [[3.0]]

    @pytest.mark.parametrize("result", [
        SimpleNamespace(),
        SimpleNamespace(embeddings=[], embeddings_by_type=None),
        SimpleNamespace(embeddings=None, embeddings_by_type={"int8": [[1]]}),
    ])
    def test_raises_when_no_embeddings(self, result):
        with pytest.raises(ValueError, match="no embeddings"):
            extract_embeddings(result)


class TestOciErrorFields:
    def test_service_error_details(self):
        fields = oci_error_fields(_service_error())
        assert fields == {
            "oci_status": 429,
            "oci_code": "TooManyRequests",
            "oci_message": "slow down",
            "oci_request_id": "rid-1",
        }

    def test_other_exception_type(self):
        assert oci_error_fields(KeyError("x")) == {"error_type": "KeyError"}


class TestBuildGenaiClient:
    def test_builds_client_with_region_and_endpoint(self, monkeypatch, test_settings):
        from_file = MagicMock(return_value={"tenancy": "t"})
        client_cls = MagicMock(return_value="client")
        monkeypatch.setattr(oci_genai.oci.config, "from_file", from_file)
        monkeypatch.setattr(oci_genai.oci.generative_ai_inference, "GenerativeAiInferenceClient", client_cls)

        assert build_genai_client(test_settings) == "client"

        kwargs = from_file.call_args.kwargs
        assert kwargs["profile_name"] == "DEFAULT"
        assert "~" not in kwargs["file_location"]  # path is expanded
        client_kwargs = client_cls.call_args.kwargs
        assert client_kwargs["config"]["region"] == "us-chicago-1"
        assert client_kwargs["service_endpoint"] == "https://genai.test"

    @pytest.mark.parametrize("error", [
        oci.exceptions.ConfigFileNotFound("missing"),
        _service_error(),
        RuntimeError("boom"),
    ])
    def test_reraises_errors(self, monkeypatch, test_settings, error):
        monkeypatch.setattr(oci_genai.oci.config, "from_file", MagicMock(side_effect=error))
        with pytest.raises(type(error)):
            build_genai_client(test_settings)
