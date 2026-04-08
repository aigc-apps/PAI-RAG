"""Retrieval API Tests based on PAI-RAG API documentation.
Reference: https://help.aliyun.com/zh/pai/use-cases/rag-api-interface-for-v0-4-x
"""
import os
from typing import Generator, Any
from fastapi.testclient import TestClient
import pytest
from httpx import Client

class TestRetrievalAPI:
    """Test cases for Retrieval API."""

    @pytest.mark.skip(reason="Requires ChromaDB vector store infrastructure")
    def test_retrieval_basic(self, client: Client, test_knowledgebase: Any):
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

    @pytest.mark.skip(reason="Requires ChromaDB vector store infrastructure")
    def test_retrieval_with_retrieval_setting(self, client: Client, test_knowledgebase:Any):
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

    @pytest.mark.skip(reason="Requires ChromaDB vector store infrastructure")
    def test_retrieval_with_metadata_condition(self, client: Client, test_knowledgebase:Any):
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

    @pytest.mark.skip(reason="Requires ChromaDB vector store infrastructure")
    def test_retrieval_with_user_id(self, client: Client, test_knowledgebase:Any):
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

    @pytest.mark.skip(reason="Requires ChromaDB vector store infrastructure")
    def test_retrieval_invalid_knowledge_id(self, client: Client, test_knowledgebase:Any):
        """Test retrieval with non-existent knowledge_id should fail."""
        retrieval_payload = {
            "query": "测试查询",
            "knowledge_id": "non_existent_kb_id"
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        # Should return error for non-existent KB
        assert response.status_code in [400, 404, 500]

    @pytest.mark.skip(reason="Requires ChromaDB vector store infrastructure")
    def test_retrieval_empty_query(self, client: Client, test_knowledgebase:Any):
        """Test retrieval with empty query."""
        kb_id = test_knowledgebase["id"]
        
        retrieval_payload = {
            "query": "",
            "knowledge_id": kb_id
        }
        
        response = client.post("/v1/retrieval", json=retrieval_payload)
        # Empty query might still work or return validation error
        assert response.status_code == 200

    @pytest.mark.skip(reason="Requires ChromaDB vector store infrastructure")
    def test_retrieval_metadata_operators(self, client: Client, test_knowledgebase:Any):
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

    @pytest.mark.skip(reason="Requires ChromaDB vector store infrastructure")
    def test_retrieval_multiple_conditions(self, client: Client, test_knowledgebase:Any):
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

