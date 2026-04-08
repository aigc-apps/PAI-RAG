"""Knowledgebase API Tests based on PAI-RAG API documentation.
Reference: https://help.aliyun.com/zh/pai/use-cases/rag-api-interface-for-v0-4-x
"""
import os
from typing import Generator, Any
from fastapi.testclient import TestClient
import pytest
from httpx import Client
from loguru import logger


class TestKnowledgeBaseAPI:
    """Test cases for Knowledgebase CRUD operations."""

    def test_list_knowledgebases(self, client: Client):
        """Test GET /v1/config/knowledgebases - List Knowledgebases with pagination."""
        response = client.get("/v1/config/knowledgebases?page=1&size=10")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert "data" in resp_json
        assert "items" in resp_json["data"]
        assert "total" in resp_json["data"]

    def test_create_knowledgebase(self, client: Client, test_embedding_model: Any):
        """Test POST /v1/config/knowledgebases - Create a new Knowledgebase."""
        create_payload = {
            "name": "test_kb_api",
            "description": "这是一个API测试知识库",
            "chunk_config": {
                "parser_type": "structure",
                "separator": "\n\n",
                "chunk_size": 1000,
                "chunk_overlap": 50
            },
            "embedding_model": "text-embedding-v3",
            "embedding_provider_name": "openai_like",
            "retrieval_config": {
                "retrieval_mode": "hybrid",
                "top_k": 5,
                "similarity_threshold": 0.2,
                "enable_rerank": False,
                "vector_weight": 0.7
            }
        }
        
        response = client.post("/v1/config/knowledgebases", json=create_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert "data" in resp_json
        
        kb_data = resp_json["data"]
        kb_id = kb_data["id"]
        assert kb_data["name"] == "test_kb_api"
        assert kb_data["description"] == "这是一个API测试知识库"
        assert kb_data["embedding_model"] == "text-embedding-v3"
        
        # Cleanup: Delete the created Knowledgebase
        delete_response = client.delete(f"/v1/config/knowledgebases/{kb_id}")
        assert delete_response.status_code == 200
        logger.info(f"Deleted knowledgebase test_kb_api successfully.")

    def test_get_knowledgebase_by_id(self, client: Client):
        """Test GET /v1/config/knowledgebases/{kb_id} - Get Knowledgebase details."""
        # First create a Knowledgebase
        create_payload = {
            "name": "test_kb_get",
            "description": "测试获取知识库详情",
            "embedding_model": "text-embedding-v3",
            "embedding_provider_name": "openai_like",
        }
        create_response = client.post("/v1/config/knowledgebases", json=create_payload)
        assert create_response.status_code == 200, f"Create failed: {create_response.json()}"
        create_json = create_response.json()
        assert create_json.get("data") is not None, f"No data in response: {create_json}"
        kb_id = create_json["data"]["id"]
        
        try:
            # Get the Knowledgebase
            get_response = client.get(f"/v1/config/knowledgebases/{kb_id}")
            assert get_response.status_code == 200
            resp_json = get_response.json()
            assert resp_json["code"] == 200
            assert resp_json["data"]["id"] == kb_id
            assert resp_json["data"]["name"] == "test_kb_get"
        finally:
            # Cleanup
            client.delete(f"/v1/config/knowledgebases/{kb_id}")
            logger.info(f"Deleted knowledgebase test_kb_get successfully.")
    
    def test_update_knowledgebase(self, client: Client):
        """Test PUT /v1/config/knowledgebases/{kb_id} - Update Knowledgebase."""
        # First create a Knowledgebase
        create_payload = {
            "name": "test_kb_update",
            "description": "原始描述",
            "embedding_model": "text-embedding-v3",
            "embedding_provider_name": "openai_like",
        }
        create_response = client.post("/v1/config/knowledgebases", json=create_payload)
        assert create_response.status_code == 200, f"Create failed: {create_response.json()}"
        create_json = create_response.json()
        assert create_json.get("data") is not None, f"No data in response: {create_json}"
        kb_id = create_json["data"]["id"]
        
        try:
            # Update the Knowledgebase
            update_payload = {
                "name": "test_kb_update",
                "description": "更新后的描述",
                "embedding_model": "text-embedding-v3",
                "embedding_provider_name": "openai_like",
                "retrieval_config": {
                    "retrieval_mode": "vector",
                    "top_k": 10,
                    "similarity_threshold": 0.3
                }
            }
            update_response = client.put(f"/v1/config/knowledgebases/{kb_id}", json=update_payload)
            assert update_response.status_code == 200
            resp_json = update_response.json()
            assert resp_json["code"] == 200
            assert resp_json["data"]["description"] == "更新后的描述"
        finally:
            # Cleanup
            client.delete(f"/v1/config/knowledgebases/{kb_id}")

    def test_delete_knowledgebase(self, client: Client):
        """Test DELETE /v1/config/knowledgebases/{kb_id} - Delete Knowledgebase."""
        # First create a Knowledgebase
        create_payload = {
            "name": "test_kb_delete",
            "description": "测试删除知识库",
            "embedding_model": "text-embedding-v3",
            "embedding_provider_name": "openai_like",
        }
        create_response = client.post("/v1/config/knowledgebases", json=create_payload)
        assert create_response.status_code == 200, f"Create failed: {create_response.json()}"
        create_json = create_response.json()
        assert create_json.get("data") is not None, f"No data in response: {create_json}"
        kb_id = create_json["data"]["id"]
        
        # Delete the Knowledgebase
        delete_response = client.delete(f"/v1/config/knowledgebases/{kb_id}")
        assert delete_response.status_code == 200
        resp_json = delete_response.json()
        assert resp_json["code"] == 200
        
        # Verify it's deleted - API may return 400 or 404 for non-existent KB
        get_response = client.get(f"/v1/config/knowledgebases/{kb_id}")
        assert get_response.status_code in [400, 404]

    def test_create_knowledgebase_duplicate_name(self, client: Client):
        """Test creating Knowledgebase with duplicate name should fail."""
        create_payload = {
            "name": "test_kb_duplicate",
            "description": "测试重复名称知识库",
            "embedding_model": "text-embedding-v3",
            "embedding_provider_name": "openai_like",
        }
        
        # Create first Knowledgebase
        response1 = client.post("/v1/config/knowledgebases", json=create_payload)
        assert response1.status_code == 200, f"First create failed: {response1.json()}"
        resp1_json = response1.json()
        assert resp1_json.get("data") is not None, f"No data in response: {resp1_json}"
        kb_id = resp1_json["data"]["id"]
        
        try:
            # Try to create with same name - should fail
            response2 = client.post("/v1/config/knowledgebases", json=create_payload)
            assert response2.status_code in [400, 409, 500], f"Duplicate should fail: {response2.json()}"
        finally:
            # Cleanup
            client.delete(f"/v1/config/knowledgebases/{kb_id}")

    def test_get_nonexistent_knowledgebase(self, client: Client):
        """Test getting a non-existent Knowledgebase should return error."""
        response = client.get("/v1/config/knowledgebases/nonexistent_kb_id")
        # API may return 400 or 404 for non-existent resources
        assert response.status_code in [400, 404]

