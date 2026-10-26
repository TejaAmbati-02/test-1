from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from opensearchpy import NotFoundError, RequestError

from services import opensearch as os_module
from services.opensearch import IndexNotFoundError, OpenSearchClientService, OpenSearchQueryError
from tests.conftest import os_search_result


@pytest.fixture
def raw_client():
    return AsyncMock()


@pytest.fixture
def service(test_settings, raw_client):
    svc = OpenSearchClientService(test_settings)
    svc._client = raw_client
    return svc


class TestGetClient:
    async def test_creates_client_once(self, monkeypatch, test_settings):
        ctor = MagicMock(return_value="client")
        monkeypatch.setattr(os_module, "AsyncOpenSearch", ctor)
        svc = OpenSearchClientService(test_settings)
        assert await svc.get_client() == "client"
        assert await svc.get_client() == "client"
        ctor.assert_called_once()
        kwargs = ctor.call_args.kwargs
        assert kwargs["hosts"] == [{"host": "opensearch.test", "port": 9200}]
        assert kwargs["http_auth"] == ("admin", "secret")
        assert kwargs["use_ssl"] is True
        assert kwargs["timeout"] == 30

    async def test_creation_failure_propagates(self, monkeypatch, test_settings):
        monkeypatch.setattr(os_module, "AsyncOpenSearch", MagicMock(side_effect=ValueError("bad host")))
        with pytest.raises(ValueError):
            await OpenSearchClientService(test_settings).get_client()


class TestClose:
    async def test_close_without_client(self, test_settings):
        await OpenSearchClientService(test_settings).close()

    async def test_close_resets_client(self, service, raw_client):
        await service.close()
        raw_client.close.assert_awaited_once()
        assert service._client is None

    async def test_close_error_is_swallowed(self, service, raw_client):
        raw_client.close.side_effect = RuntimeError("boom")
        await service.close()
        assert service._client is raw_client


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

    @pytest.mark.parametrize("exists", [True, False])
    async def test_index_exists(self, service, raw_client, exists):
        raw_client.indices.exists.return_value = exists
        assert await service.index_exists("products") is exists

    async def test_index_exists_error_returns_false(self, service, raw_client):
        raw_client.indices.exists.side_effect = ConnectionError("down")
        assert await service.index_exists("products") is False


class TestPipelines:
    async def test_pipeline_exists(self, service, raw_client):
        assert await service.pipeline_exists("p1") is True
        raw_client.transport.perform_request.assert_awaited_once_with("GET", "/_search/pipeline/p1")

    async def test_pipeline_missing(self, service, raw_client):
        raw_client.transport.perform_request.side_effect = NotFoundError(404, "not found", {})
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
        with pytest.raises(IndexNotFoundError, match="Index 'products' not found"):
            await service.search("products", {})

    async def test_bad_query(self, service, raw_client):
        raw_client.search.side_effect = RequestError(400, "parsing_exception", {"error": "bad"})
        with pytest.raises(OpenSearchQueryError, match="OpenSearch query error"):
            await service.search("products", {})

    async def test_unexpected_error_propagates(self, service, raw_client):
        raw_client.search.side_effect = ConnectionError("down")
        with pytest.raises(ConnectionError):
            await service.search("products", {})
