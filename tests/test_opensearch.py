from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy import NotFoundError, RequestError

from services import opensearch as os_module
from models.response import ApiError
from services.opensearch import OpenSearchClientService
from tests.conftest import os_search_result


@pytest.fixture
def raw_client():
    return AsyncMock()


@pytest.fixture
def service(test_settings, raw_client):
    svc = OpenSearchClientService(test_settings)
    svc.client = raw_client
    return svc


class TestClient:
    def test_client_is_built_from_settings(self, monkeypatch, test_settings):
        ctor = MagicMock(return_value="client")
        monkeypatch.setattr(os_module, "AsyncOpenSearch", ctor)
        assert OpenSearchClientService(test_settings).client == "client"
        ctor.assert_called_once()
        kwargs = ctor.call_args.kwargs
        assert kwargs["hosts"] == [{"host": "opensearch.test", "port": 9200}]
        assert kwargs["http_auth"] == ("admin", "secret")
        assert kwargs["use_ssl"] is True
        assert kwargs["timeout"] == 30
        assert kwargs["verify_certs"] is False
        assert kwargs["http_compress"] is True


class TestClose:
    async def test_close(self, service, raw_client):
        await service.close()
        raw_client.close.assert_awaited_once()

    async def test_close_error_is_logged_not_raised(self, service, raw_client):
        raw_client.close.side_effect = RuntimeError("boom")
        await service.close()


class TestPingAndInfo:
    @pytest.mark.parametrize("reachable", [True, False])
    async def test_ping(self, service, raw_client, reachable):
        raw_client.ping.return_value = reachable
        assert await service.ping() is reachable

    async def test_ping_exception_returns_false(self, service, raw_client):
        raw_client.ping.side_effect = ConnectionError("down")
        assert await service.ping() is False

    async def test_cluster_name(self, service, raw_client):
        raw_client.info.return_value = {"cluster_name": "asmo-cluster"}
        assert await service.cluster_name() == "asmo-cluster"

    async def test_cluster_name_error_returns_none(self, service, raw_client):
        raw_client.info.side_effect = ConnectionError("down")
        assert await service.cluster_name() is None


class TestIndices:
    async def test_list_indices_skips_system_indices(self, service, raw_client):
        raw_client.cat.indices.return_value = [
            {"index": "products", "docs.count": "12", "store.size": "2048", "status": "open"},
            {"index": ".kibana", "docs.count": "1", "store.size": "1", "status": "open"},
            {"index": "empty", "docs.count": None, "store.size": None},
        ]
        assert await service.list_indices() == [
            {"name": "products", "doc_count": 12, "size_bytes": 2048, "status": "open"},
            {"name": "empty", "doc_count": 0, "size_bytes": 0, "status": "unknown"},
        ]
        raw_client.cat.indices.assert_awaited_once_with(format="json", bytes="b")

    async def test_list_indices_error_propagates(self, service, raw_client):
        raw_client.cat.indices.side_effect = ConnectionError("down")
        with pytest.raises(ConnectionError):
            await service.list_indices()


class TestPipelines:
    async def test_pipeline_exists(self, service, raw_client):
        assert await service.pipeline_exists("p1") is True
        raw_client.transport.perform_request.assert_awaited_once_with("GET", "/_search/pipeline/p1")

    async def test_pipeline_missing(self, service, raw_client):
        raw_client.transport.perform_request.side_effect = NotFoundError(404, "not found", {})
        assert await service.pipeline_exists("p1") is False

    async def test_pipeline_exists_any_error_returns_false(self, service, raw_client):
        raw_client.transport.perform_request.side_effect = ConnectionError("down")
        assert await service.pipeline_exists("p1") is False

    async def test_create_rrf_pipeline(self, service, raw_client):
        raw_client.transport.perform_request.return_value = {"acknowledged": True}
        assert await service.create_search_pipeline("p1", rank_constant=42) == {"acknowledged": True}
        method, path = raw_client.transport.perform_request.call_args.args
        body = raw_client.transport.perform_request.call_args.kwargs["body"]
        assert (method, path) == ("PUT", "/_search/pipeline/p1")
        combination = body["phase_results_processors"][0]["score-ranker-processor"]["combination"]
        assert combination == {"technique": "rrf", "rank_constant": 42}
        assert body["description"] == "RRF hybrid search pipeline"

    async def test_create_weighted_rrf_pipeline(self, service, raw_client):
        await service.create_search_pipeline("p2", weights=[0.2, 0.2, 0.6])
        body = raw_client.transport.perform_request.call_args.kwargs["body"]
        combination = body["phase_results_processors"][0]["score-ranker-processor"]["combination"]
        assert combination == {
            "technique": "rrf", "rank_constant": 60, "parameters": {"weights": [0.2, 0.2, 0.6]},
        }
        assert body["description"] == "RRF hybrid search pipeline (weighted)"

    async def test_empty_weights_create_unweighted_pipeline(self, service, raw_client):
        await service.create_search_pipeline("p3", weights=[])
        body = raw_client.transport.perform_request.call_args.kwargs["body"]
        combination = body["phase_results_processors"][0]["score-ranker-processor"]["combination"]
        assert "parameters" not in combination
        assert body["description"] == "RRF hybrid search pipeline"

    async def test_create_pipeline_error_propagates(self, service, raw_client):
        raw_client.transport.perform_request.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await service.create_search_pipeline("p1")


class TestSearch:
    async def test_search_with_pipeline(self, service, raw_client):
        raw_client.search.return_value = os_search_result([{"_source": {"sku_id": "1"}}])
        result = await service.search("products", {"size": 1}, pipeline_id="p1")
        assert result["hits"]["total"]["value"] == 1
        raw_client.search.assert_awaited_once_with(index="products", body={"size": 1}, params={"search_pipeline": "p1"})

    async def test_search_without_pipeline(self, service, raw_client):
        raw_client.search.return_value = os_search_result()
        await service.search("products", {})
        assert raw_client.search.call_args.kwargs["params"] == {}

    async def test_missing_index(self, service, raw_client):
        raw_client.search.side_effect = NotFoundError(404, "index_not_found_exception", {})
        with pytest.raises(ApiError) as e:
            await service.search("products", {})
        assert (e.value.status_code, e.value.detail) == (404, {"code": "INDEX_NOT_FOUND", "message": "Index 'products' not found"})

    async def test_bad_query(self, service, raw_client):
        raw_client.search.side_effect = RequestError(400, "parsing_exception", {"error": "bad"})
        with pytest.raises(ApiError) as e:
            await service.search("products", {})
        assert (e.value.status_code, e.value.detail["code"]) == (400, "QUERY_ERROR")
        assert "OpenSearch query error" in e.value.detail["message"]

    async def test_unexpected_error_propagates(self, service, raw_client):
        raw_client.search.side_effect = ConnectionError("down")
        with pytest.raises(ConnectionError):
            await service.search("products", {})

    async def test_search_body_is_passed_unredacted(self, service, raw_client):
        raw_client.search.return_value = os_search_result()
        vector = [0.5] * 32
        body = {"query": {"knn": {"content_vector": {"vector": vector, "k": 10}}}}
        await service.search("products", body)
        assert raw_client.search.call_args.kwargs["body"]["query"]["knn"]["content_vector"]["vector"] == vector

class TestLoggedBody:
    """The search logs its body with embedding vectors shortened to "<vector dim=N>"."""

    async def _logged(self, service, raw_client, monkeypatch, body) -> dict:
        raw_client.search.return_value = os_search_result()
        info = MagicMock()
        monkeypatch.setattr(os_module.logger, "info", info)
        await service.search("products", body)
        return json.loads(info.call_args_list[0].kwargs["extra"]["input_data"]["body"])

    async def test_vectors_are_shortened(self, service, raw_client, monkeypatch):
        body = {"size": 10, "query": {"hybrid": {"queries": [
            {"multi_match": {"query": "valve", "fields": ["a", "b"]}},
            {"knn": {"content_vector": {"vector": [1.0] * 1024, "k": 100}}},
        ]}}}
        logged = await self._logged(service, raw_client, monkeypatch, body)
        assert logged["size"] == 10
        assert logged["query"]["hybrid"]["queries"][0] == {"multi_match": {"query": "valve", "fields": ["a", "b"]}}
        assert logged["query"]["hybrid"]["queries"][1]["knn"]["content_vector"] == {"vector": "<vector dim=1024>", "k": 100}
        # The body sent to OpenSearch is not modified.
        assert raw_client.search.call_args.kwargs["body"]["query"]["hybrid"]["queries"][1]["knn"]["content_vector"]["vector"] == [1.0] * 1024

    async def test_short_and_mixed_lists_are_kept(self, service, raw_client, monkeypatch):
        body = {"short": [0.1] * 16, "mixed": [0.1] * 20 + ["x"], "neg": [-0.5, 1e-3] * 9}
        logged = await self._logged(service, raw_client, monkeypatch, body)
        assert logged["short"] == [0.1] * 16
        assert logged["mixed"] == [0.1] * 20 + ["x"]
        assert logged["neg"] == "<vector dim=18>"
