"""Embedding Model API Tests based on PAI-RAG API documentation.
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


class TestEmbeddingAPI:
    """Test cases for Embedding Model CRUD operations."""

    def test_list_embeddings(self, client: Client):
        """Test GET /v1/config/embeddings - List embedding models."""
        response = client.get("/v1/config/embeddings")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert "data" in resp_json

    def test_list_embeddings_with_pagination(self, client: Client):
        """Test GET /v1/config/embeddings with pagination parameters."""
        response = client.get("/v1/config/embeddings?page=1&size=10")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert "items" in resp_json["data"]
        assert "total" in resp_json["data"]
        assert "page" in resp_json["data"]
        assert "size" in resp_json["data"]

    def test_list_embeddings_filter_by_provider(self, client: Client):
        """Test GET /v1/config/embeddings with provider_name filter."""
        response = client.get("/v1/config/embeddings?provider_name=dashscope")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200

    def test_create_embedding(self, client: Client):
        """Test POST /v1/config/embeddings - Create a new embedding model."""
        create_payload = {
            "model_id": "test-embedding-model",
            "model_name": "text-embedding-v2",
            "type": "openai_like",  # EmbeddingType enum value
            "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "sk-test-key",
            "provider_name": "dashscope"
        }
        
        response = client.post("/v1/config/embeddings", json=create_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        
        emb_data = resp_json["data"]
        emb_id = emb_data["id"]
        assert emb_data["model_id"] == "test-embedding-model"
        assert emb_data["model_name"] == "text-embedding-v2"
        assert "api_key" not in emb_data, "API key should not be returned"
        
        # Cleanup
        client.delete(f"/v1/config/embeddings/{emb_id}")

    def test_get_embedding_by_model_name(self, client: Client):
        """Test GET /v1/config/embeddings?model_name=xxx - Get embedding by model name."""
        # First create an embedding model
        create_payload = {
            "model_id": "test-emb-get",
            "model_name": "test-model-name",
            "type": "openai_like",
            "endpoint": "http://localhost:8000",
            "api_key": "test-key"
        }
        create_response = client.post("/v1/config/embeddings", json=create_payload)
        assert create_response.status_code == 200, f"Create failed: {create_response.json()}"
        emb_id = create_response.json()["data"]["id"]
        
        # Get by model_name
        get_response = client.get("/v1/config/embeddings?model_name=test-model-name")
        assert get_response.status_code == 200
        resp_json = get_response.json()
        assert resp_json["code"] == 200
        
        # Cleanup
        client.delete(f"/v1/config/embeddings/{emb_id}")

    def test_update_embedding(self, client: Client):
        """Test PUT /v1/config/embeddings/{emb_id} - Update embedding model."""
        # First create an embedding model
        create_payload = {
            "model_id": "test-emb-update",
            "model_name": "original-name",
            "type": "openai_like",
            "endpoint": "http://localhost:8000",
            "api_key": "test-key"
        }
        create_response = client.post("/v1/config/embeddings", json=create_payload)
        assert create_response.status_code == 200, f"Create failed: {create_response.json()}"
        emb_id = create_response.json()["data"]["id"]
        
        # Update
        update_payload = {
            "model_id": "test-emb-update",
            "model_name": "updated-name",
            "type": "openai_like",
            "endpoint": "http://localhost:8001",
            "api_key": "new-test-key"
        }
        update_response = client.put(f"/v1/config/embeddings/{emb_id}", json=update_payload)
        assert update_response.status_code == 200
        resp_json = update_response.json()
        assert resp_json["code"] == 200
        assert resp_json["data"]["model_name"] == "updated-name"
        
        # Cleanup
        client.delete(f"/v1/config/embeddings/{emb_id}")

    def test_delete_embedding(self, client: Client):
        """Test DELETE /v1/config/embeddings/{emb_id} - Delete embedding model."""
        # First create an embedding model
        create_payload = {
            "model_id": "test-emb-delete",
            "model_name": "to-delete",
            "type": "openai_like",
            "endpoint": "http://localhost:8000",
            "api_key": "test-key"
        }
        create_response = client.post("/v1/config/embeddings", json=create_payload)
        assert create_response.status_code == 200, f"Create failed: {create_response.json()}"
        emb_id = create_response.json()["data"]["id"]
        
        # Delete
        delete_response = client.delete(f"/v1/config/embeddings/{emb_id}")
        assert delete_response.status_code == 200
        resp_json = delete_response.json()
        assert resp_json["code"] == 200

    def test_get_embedding_providers(self, client: Client):
        """Test GET /v1/config/embeddings/providers - Get distinct provider names."""
        response = client.get("/v1/config/embeddings/providers")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert isinstance(resp_json["data"], list)

