"""Retrieval API Tests based on PAI-RAG API documentation.
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


@pytest.fixture()
def test_knowledgebase(client: Client):
    """Create a test knowledge base for retrieval tests."""
    create_payload = {
        "name": "test_kb_retrieval",
        "description": "用于检索测试的知识库",
        "embedding_model": "BAAI/bge-m3",
        "retrieval_config": {
            "retrieval_mode": "vector",
            "top_k": 5,
            "similarity_threshold": 0.2
        }
    }
    response = client.post("/v1/config/knowledgebases", json=create_payload)
    kb_data = response.json()["data"]
    yield kb_data
    # Cleanup
    client.delete(f"/v1/config/knowledgebases/{kb_data['id']}")


class TestRetrievalAPI:
    """Test cases for Retrieval API."""

    def test_retrieval_basic(self, client: Client, test_knowledgebase):
        """Test POST /v1/retrieval - Basic retrieval request."""
        kb_id = test_knowledgebase["id"]
        
        retrieval_payload = {
            "query": "测试查询",
            "knowledge_id": kb_id
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert "records" in resp_json

    def test_retrieval_with_retrieval_setting(self, client: Client, test_knowledgebase):
        """Test retrieval with custom retrieval settings that override KB defaults."""
        kb_id = test_knowledgebase["id"]
        
        retrieval_payload = {
            "query": "测试查询",
            "knowledge_id": kb_id,
            "retrieval_setting": {
                "top_k": 3,
                "score_threshold": 0.5
            }
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert "records" in resp_json

    def test_retrieval_with_metadata_condition(self, client: Client, test_knowledgebase):
        """Test retrieval with metadata filtering conditions."""
        kb_id = test_knowledgebase["id"]
        
        retrieval_payload = {
            "query": "测试查询",
            "knowledge_id": kb_id,
            "metadata_condition": {
                "conditions": [
                    {
                        "name": "department",
                        "value": "it",
                        "comparison_operator": "="
                    }
                ],
                "logical_operator": "and"
            }
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert "records" in resp_json

    def test_retrieval_with_user_id(self, client: Client, test_knowledgebase):
        """Test retrieval with user_id for personalization/tracking."""
        kb_id = test_knowledgebase["id"]
        
        retrieval_payload = {
            "query": "测试查询",
            "knowledge_id": kb_id,
            "user_id": "test_user_123"
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        assert response.status_code == 200
        resp_json = response.json()
        assert "records" in resp_json

    def test_retrieval_invalid_knowledge_id(self, client: Client):
        """Test retrieval with non-existent knowledge_id should fail."""
        retrieval_payload = {
            "query": "测试查询",
            "knowledge_id": "non_existent_kb_id"
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        # Should return error for non-existent KB
        assert response.status_code in [400, 404, 500]

    def test_retrieval_empty_query(self, client: Client, test_knowledgebase):
        """Test retrieval with empty query."""
        kb_id = test_knowledgebase["id"]
        
        retrieval_payload = {
            "query": "",
            "knowledge_id": kb_id
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        # Empty query might still work or return validation error
        assert response.status_code in [200, 400, 422]

    def test_retrieval_metadata_operators(self, client: Client, test_knowledgebase):
        """Test retrieval with various metadata comparison operators."""
        kb_id = test_knowledgebase["id"]
        
        # Test 'contains' operator
        retrieval_payload = {
            "query": "测试查询",
            "knowledge_id": kb_id,
            "metadata_condition": {
                "conditions": [
                    {
                        "name": "file_name",
                        "value": "test",
                        "comparison_operator": "contains"
                    }
                ],
                "logical_operator": "and"
            }
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        assert response.status_code == 200

    def test_retrieval_multiple_conditions(self, client: Client, test_knowledgebase):
        """Test retrieval with multiple metadata conditions."""
        kb_id = test_knowledgebase["id"]
        
        retrieval_payload = {
            "query": "测试查询",
            "knowledge_id": kb_id,
            "metadata_condition": {
                "conditions": [
                    {
                        "name": "department",
                        "value": "it",
                        "comparison_operator": "="
                    },
                    {
                        "name": "file_name",
                        "value": ".txt",
                        "comparison_operator": "end with"
                    }
                ],
                "logical_operator": "and"
            }
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        assert response.status_code == 200

