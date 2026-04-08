"""Integration tests for LLM and embedding model configuration CRUD."""

import os
import pytest
from tests.integration.conftest import pytestmark, DASHSCOPE_ENDPOINT  # noqa: F401


class TestEmbeddingModelConfig:
    """Tests for embedding model CRUD via /v1/config/embeddings."""

    def test_create_embedding_model(self, client):
        """Create an embedding model and verify response."""
        payload = {
            "model_id": "emb-crud-test",
            "model_name": "text-embedding-v4",
            "type": "openai_like",
            "endpoint": DASHSCOPE_ENDPOINT,
            "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "provider_name": "openai_like",
        }
        response = client.post("/v1/config/embeddings", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        emb = data["data"]
        assert emb["model_id"] == "emb-crud-test"
        assert "id" in emb

        # Cleanup
        client.delete(f"/v1/config/embeddings/{emb['id']}")

    def test_list_embedding_models(self, client, test_embedding_model):
        """List embedding models includes the fixture model."""
        response = client.get("/v1/config/embeddings")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        items = data["data"]["items"]
        ids = [m["model_id"] for m in items]
        assert test_embedding_model["model_id"] in ids

    def test_get_embedding_model_by_id(self, client, test_embedding_model):
        """Get an embedding model by filtering on model_name."""
        response = client.get(
            f"/v1/config/embeddings?model_name={test_embedding_model['model_name']}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_delete_embedding_model(self, client):
        """Create then delete an embedding model."""
        payload = {
            "model_id": "emb-delete-test",
            "model_name": "text-embedding-v4",
            "type": "openai_like",
            "endpoint": DASHSCOPE_ENDPOINT,
            "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "provider_name": "openai_like",
        }
        resp = client.post("/v1/config/embeddings", json=payload)
        emb_id = resp.json()["data"]["id"]

        delete_resp = client.delete(f"/v1/config/embeddings/{emb_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["code"] == 200


class TestLlmModelConfig:
    """Tests for LLM model CRUD via /v1/config/llms."""

    def test_create_llm_model(self, client):
        """Create an LLM model and verify response."""
        payload = {
            "model_id": "llm-crud-test",
            "model_name": "qwen3.5-plus",
            "model": "qwen3.5-plus",
            "base_url": DASHSCOPE_ENDPOINT,
            "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "temperature": 0.1,
            "context_window": 8192,
            "provider_name": "dashscope",
        }
        response = client.post("/v1/config/llms", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        llm = data["data"]
        assert llm["model_id"] == "llm-crud-test"
        assert "id" in llm

        # Cleanup
        client.delete(f"/v1/config/llms/{llm['id']}")

    def test_list_llm_models(self, client, test_llm_model):
        """List LLM models includes the fixture model."""
        response = client.get("/v1/config/llms")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        items = data["data"]["items"]
        ids = [m["model_id"] for m in items]
        assert test_llm_model["model_id"] in ids

    def test_get_llm_model_by_id(self, client, test_llm_model):
        """Get a specific LLM model by id."""
        llm_id = test_llm_model["id"]
        response = client.get(f"/v1/config/llms/{llm_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["id"] == llm_id

    def test_update_llm_model(self, client, test_llm_model):
        """Update an LLM model's temperature."""
        llm_id = test_llm_model["id"]
        update_payload = {
            "model_id": test_llm_model["model_id"],
            "model_name": test_llm_model.get("model_name", "qwen3.5-plus"),
            "model": test_llm_model.get("model", "qwen3.5-plus"),
            "base_url": DASHSCOPE_ENDPOINT,
            "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "temperature": 0.5,
            "context_window": 8192,
            "provider_name": "dashscope",
        }
        response = client.put(f"/v1/config/llms/{llm_id}", json=update_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["temperature"] == 0.5

    def test_delete_llm_model(self, client):
        """Create then delete an LLM model."""
        payload = {
            "model_id": "llm-delete-test",
            "model_name": "qwen3.5-plus",
            "model": "qwen3.5-plus",
            "base_url": DASHSCOPE_ENDPOINT,
            "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "provider_name": "dashscope",
        }
        resp = client.post("/v1/config/llms", json=payload)
        assert resp.status_code == 200, f"Failed to create LLM: {resp.json()}"
        llm_id = resp.json()["data"]["id"]

        delete_resp = client.delete(f"/v1/config/llms/{llm_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["code"] == 200
