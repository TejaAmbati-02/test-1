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

    def test_plain_embeddings_win_over_by_type(self):
        result = SimpleNamespace(embeddings=[[1.0]], embeddings_by_type={"float": [[9.0]]})
        assert extract_embeddings(result) == [[1.0]]

    def test_multiple_vectors_are_lists(self):
        result = SimpleNamespace(embeddings=[(1.0, 2.0), (3.0, 4.0)])
        out = extract_embeddings(result)
        assert out == [[1.0, 2.0], [3.0, 4.0]]
        assert all(isinstance(v, list) for v in out)

    def test_unexpected_error_is_reraised(self):
        # A non-iterable vector raises TypeError, which is logged and re-raised unchanged.
        with pytest.raises(TypeError):
            extract_embeddings(SimpleNamespace(embeddings=[1.0]))

    @pytest.mark.parametrize("result", [
        SimpleNamespace(),
        SimpleNamespace(embeddings=None, embeddings_by_type={"float": []}),
        SimpleNamespace(embeddings=None, embeddings_by_type=SimpleNamespace(int8=[[1]])),
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

    def test_broken_service_error_falls_back_to_type(self):
        # A ServiceError missing its attributes must not mask the original error.
        class BrokenServiceError(oci.exceptions.ServiceError):
            def __init__(self):  # skip ServiceError.__init__, so no attributes are set
                Exception.__init__(self, "broken")

            @property
            def status(self):
                raise RuntimeError("no status")

        assert oci_error_fields(BrokenServiceError()) == {"error_type": "BrokenServiceError"}


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

    def test_client_creation_failure_is_reraised(self, monkeypatch, test_settings):
        monkeypatch.setattr(oci_genai.oci.config, "from_file", MagicMock(return_value={}))
        monkeypatch.setattr(
            oci_genai.oci.generative_ai_inference, "GenerativeAiInferenceClient",
            MagicMock(side_effect=_service_error()),
        )
        with pytest.raises(oci.exceptions.ServiceError):
            build_genai_client(test_settings)

    def test_region_overrides_config_file(self, monkeypatch, test_settings):
        monkeypatch.setattr(oci_genai.oci.config, "from_file", MagicMock(return_value={"region": "eu-frankfurt-1"}))
        client_cls = MagicMock(return_value="client")
        monkeypatch.setattr(oci_genai.oci.generative_ai_inference, "GenerativeAiInferenceClient", client_cls)
        build_genai_client(test_settings.model_copy(update={"oci_region": "me-jeddah-1"}))
        assert client_cls.call_args.kwargs["config"]["region"] == "me-jeddah-1"
