"""Reranker Model API Tests based on PAI-RAG API documentation.
Reference: https://help.aliyun.com/zh/pai/use-cases/rag-api-interface-for-v0-4-x
"""
import os
from typing import Generator
from fastapi.testclient import TestClient
import pytest
from httpx import Client

os.environ["SQLITE_URL"] = "sqlite+aiosqlite:///./localdata/pytest.db"
os.environ["DB_TYPE"] = "sqlite"


@pytest.fixture()
def client() -> Generator[None, None, Client]:
    from app.main import app
    with TestClient(app) as client:
        yield client


class TestRerankerAPI:
    """Test cases for Reranker Model CRUD operations."""

    def test_list_rerankers(self, client: Client):
        """Test GET /v1/config/rerankers - List reranker models."""
        response = client.get("/v1/config/rerankers")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert "data" in resp_json

    def test_list_rerankers_with_pagination(self, client: Client):
        """Test GET /v1/config/rerankers with pagination parameters."""
        response = client.get("/v1/config/rerankers?page=1&size=10")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert "items" in resp_json["data"]
        assert "total" in resp_json["data"]

    def test_list_rerankers_filter_by_provider(self, client: Client):
        """Test GET /v1/config/rerankers with provider_name filter."""
        response = client.get("/v1/config/rerankers?provider_name=dashscope")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200

    def test_create_reranker(self, client: Client):
        """Test POST /v1/config/rerankers - Create a new reranker model."""
        create_payload = {
            "model_id": "test-reranker-model",
            "model_name": "gte-rerank",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "sk-test-key",
            "provider_name": "dashscope"
        }
        
        response = client.post("/v1/config/rerankers", json=create_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        
        reranker_data = resp_json["data"]
        reranker_id = reranker_data["id"]
        assert reranker_data["model_id"] == "test-reranker-model"
        assert "api_key" not in reranker_data, "API key should not be returned"
        
        # Cleanup
        client.delete(f"/v1/config/rerankers/{reranker_id}")

    def test_get_reranker_by_model_name(self, client: Client):
        """Test GET /v1/config/rerankers?model_name=xxx - Get reranker by model name."""
        # First create a reranker model
        create_payload = {
            "model_id": "test-reranker-get",
            "model_name": "test-reranker-name",
            "base_url": "http://localhost:8000",
            "api_key": "test-key"
        }
        create_response = client.post("/v1/config/rerankers", json=create_payload)
        reranker_id = create_response.json()["data"]["id"]
        
        # Get by model_name
        get_response = client.get("/v1/config/rerankers?model_name=test-reranker-name")
        assert get_response.status_code == 200
        resp_json = get_response.json()
        assert resp_json["code"] == 200
        
        # Cleanup
        client.delete(f"/v1/config/rerankers/{reranker_id}")

    def test_update_reranker(self, client: Client):
        """Test PUT /v1/config/rerankers/{reranker_id} - Update reranker model."""
        # First create a reranker model
        create_payload = {
            "model_id": "test-reranker-update",
            "model_name": "original-reranker",
            "base_url": "http://localhost:8000",
            "api_key": "test-key"
        }
        create_response = client.post("/v1/config/rerankers", json=create_payload)
        reranker_id = create_response.json()["data"]["id"]
        
        # Update
        update_payload = {
            "model_id": "test-reranker-update",
            "model_name": "updated-reranker",
            "base_url": "http://localhost:8001",
            "api_key": "new-test-key"
        }
        update_response = client.put(f"/v1/config/rerankers/{reranker_id}", json=update_payload)
        assert update_response.status_code == 200
        resp_json = update_response.json()
        assert resp_json["code"] == 200
        assert resp_json["data"]["model_name"] == "updated-reranker"
        
        # Cleanup
        client.delete(f"/v1/config/rerankers/{reranker_id}")

    def test_delete_reranker(self, client: Client):
        """Test DELETE /v1/config/rerankers/{reranker_id} - Delete reranker model."""
        # First create a reranker model
        create_payload = {
            "model_id": "test-reranker-delete",
            "model_name": "to-delete-reranker",
            "base_url": "http://localhost:8000",
            "api_key": "test-key"
        }
        create_response = client.post("/v1/config/rerankers", json=create_payload)
        reranker_id = create_response.json()["data"]["id"]
        
        # Delete
        delete_response = client.delete(f"/v1/config/rerankers/{reranker_id}")
        assert delete_response.status_code == 200
        resp_json = delete_response.json()
        assert resp_json["code"] == 200

    def test_get_reranker_providers(self, client: Client):
        """Test GET /v1/config/rerankers/providers - Get distinct provider names."""
        response = client.get("/v1/config/rerankers/providers")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert isinstance(resp_json["data"], list)

