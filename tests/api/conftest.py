"""Shared fixtures for API tests.

Reference: https://help.aliyun.com/zh/pai/use-cases/rag-api-interface-for-v0-4-x
"""
import os
from typing import Generator
from fastapi.testclient import TestClient
import pytest
from httpx import Client

# Set up test database before importing app
os.environ["SQLITE_URL"] = "sqlite+aiosqlite:///./localdata/pytest.db"
os.environ["DB_TYPE"] = "sqlite"


@pytest.fixture(scope="session")
def app():
    """Create FastAPI application instance."""
    from app.main import app
    return app


@pytest.fixture()
def client(app) -> Generator[None, None, Client]:
    """Create test client for API testing."""
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def test_llm_model(client: Client):
    """Create a test LLM model and cleanup after test."""
    create_payload = {
        "model_id": "test-llm-fixture",
        "base_url": "http://localhost:8000",
        "model": "Qwen3-8B",
        "api_key": "sk-test-fixture",
        "temperature": 0.7,
        "context_window": 8192,
    }
    response = client.post("/v1/config/llms", json=create_payload)
    llm_data = response.json()["data"]
    yield llm_data
    # Cleanup
    client.delete(f"/v1/config/llms/{llm_data['id']}")


@pytest.fixture()
def test_embedding_model(client: Client):
    """Create a test embedding model and cleanup after test."""
    create_payload = {
        "model_id": "test-embedding-fixture",
        "model_name": "test-embedding",
        "type": "openai_like",
        "endpoint": "http://localhost:8000",
        "api_key": "test-key"
    }
    response = client.post("/v1/config/embeddings", json=create_payload)
    emb_data = response.json()["data"]
    yield emb_data
    # Cleanup
    client.delete(f"/v1/config/embeddings/{emb_data['id']}")


@pytest.fixture()
def test_reranker_model(client: Client):
    """Create a test reranker model and cleanup after test."""
    create_payload = {
        "model_id": "test-reranker-fixture",
        "model_name": "test-reranker",
        "base_url": "http://localhost:8000",
        "api_key": "test-key"
    }
    response = client.post("/v1/config/rerankers", json=create_payload)
    reranker_data = response.json()["data"]
    yield reranker_data
    # Cleanup
    client.delete(f"/v1/config/rerankers/{reranker_data['id']}")


@pytest.fixture()
def test_knowledgebase(client: Client):
    """Create a test knowledge base and cleanup after test."""
    create_payload = {
        "name": "test_kb_fixture",
        "description": "Fixture knowledge base for testing",
        "embedding_model": "BAAI/bge-m3",
        "chunk_config": {
            "parser_type": "structure",
            "chunk_size": 1000,
            "chunk_overlap": 50
        },
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

