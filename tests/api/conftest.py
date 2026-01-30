"""Shared fixtures for API tests.

Reference: https://help.aliyun.com/zh/pai/use-cases/rag-api-interface-for-v0-4-x
"""
import os
import asyncio
from typing import Generator
from fastapi.testclient import TestClient
import pytest
from httpx import Client
from typing import Any
from loguru import logger

if os.path.exists("./localdata/pytest.db"):
    logger.info("Removing existing test database...")
    os.remove("./localdata/pytest.db")

# Set up test database before importing app
os.environ["SQLITE_URL"] = "sqlite+aiosqlite:///./localdata/pytest.db"
os.environ["DB_TYPE"] = "sqlite"

@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def app():
    """Create FastAPI application instance."""
    from app.main import app
    return app


@pytest.fixture(scope="session")
def client(app) -> Generator[None, None, Client]:
    """Create test client for API testing."""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def test_llm_model(client: Client):
    """Create a test LLM model and cleanup after test."""
    create_payload = {
        "model_id": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "temperature": 0.1,
        "context_window": 8192,
        "provider_name": "dashscope",
        "type": "dashscope",
    }
    response = client.post("/v1/config/llms", json=create_payload)
    llm_data = response.json()["data"]
    yield llm_data
    # Cleanup
    client.delete(f"/v1/config/llms/{llm_data['id']}")


@pytest.fixture(scope="session")
def test_embedding_model(client: Client):
    """Create a test embedding model and cleanup after test."""
    create_payload = {
        "model_id": "text-embedding-v3",
        "model_name": "text-embedding-v3",
        "type": "openai_like",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "provider_name": "openai_like"
    }
    response = client.post("/v1/config/embeddings", json=create_payload)
    emb_data = response.json()["data"]
    logger.debug(f"test_embedding_model: {emb_data}")
    yield emb_data
    # Cleanup
    embed_id = emb_data.get("id") if emb_data else None
    if embed_id:
        client.delete(f"/v1/config/embeddings/{embed_id}")


@pytest.fixture(scope="session")
def test_reranker_model(client: Client):
    """Create a test reranker model and cleanup after test."""
    create_payload = {
        "model_id": "qwen3-rerank",
        "model_name": "qwen3-rerank",
        "base_url": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "provider_name": "dashscope",
        "type": "dashscope",
    }
    response = client.post("/v1/config/rerankers", json=create_payload)
    reranker_data = response.json()["data"]
    yield reranker_data
    # Cleanup
    client.delete(f"/v1/config/rerankers/{reranker_data['id']}")


@pytest.fixture(scope="session")
def test_knowledgebase(client: Client, test_embedding_model: Any, test_reranker_model: Any):
    """Create a test knowledgebase and cleanup after test."""
    create_payload = {
        "name": "test_kb_for_retrieval",
        "description": "Fixture knowledgebase for testing retrieval",
        "embedding_model": test_embedding_model["model_id"],
        "embedding_provider_name": test_embedding_model["provider_name"],
        "chunk_config": {
            "parser_type": "structure",
            "chunk_size": 1000,
            "chunk_overlap": 50
        },
        "retrieval_config": {
            "retrieval_mode": "vector",
            "top_k": 5,
            "similarity_threshold": 0.2,
            "enable_rerank": True,
            "rerank_model": test_reranker_model["model_id"],
            "rerank_provider_name": test_reranker_model["provider_name"],
            "rerank_top_k": 5,
        }
    }
    response = client.post("/v1/config/knowledgebases", json=create_payload)
    kb_data = response.json()["data"]
    yield kb_data
    # Cleanup
    client.delete(f"/v1/config/knowledgebases/{kb_data['id']}")


@pytest.fixture(scope="session")
def test_knowledgebase_for_file(client: Client, test_embedding_model: Any, test_reranker_model: Any):
    """Create a test knowledgebase and cleanup after test."""
    create_payload = {
        "name": "test_kb_for_file",
        "description": "Fixture knowledgebase for testing file",
        "embedding_model": test_embedding_model["model_id"],
        "embedding_provider_name": test_embedding_model["provider_name"],
        "chunk_config": {
            "parser_type": "structure",
            "chunk_size": 1000,
            "chunk_overlap": 50
        },
        "retrieval_config": {
            "retrieval_mode": "vector",
            "top_k": 5,
            "similarity_threshold": 0.2,
            "enable_rerank": True,
            "rerank_model": test_reranker_model["model_id"],
            "rerank_provider_name": test_reranker_model["provider_name"],
            "rerank_top_k": 5,
        }
    }
    response = client.post("/v1/config/knowledgebases", json=create_payload)
    kb_data = response.json()["data"]
    yield kb_data
    # Cleanup
    client.delete(f"/v1/config/knowledgebases/{kb_data['id']}")

