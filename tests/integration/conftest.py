"""Shared fixtures for integration tests.

Requires:
- DASHSCOPE_API_KEY env var (for text-embedding-v4 and qwen3.5-plus)
- redis-server running locally on port 6379
- ChromaDB auto-starts via LocalChromaService in app lifespan (port 8684)
- SQLite DB created automatically
"""

import os
import sys
import asyncio
from typing import Any, Generator

import pytest
from httpx import Client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

# Use a dedicated SQLite DB for integration tests
if os.path.exists("./localdata/pytest_integration.db"):
    os.remove("./localdata/pytest_integration.db")

os.environ["SQLITE_URL"] = "sqlite+aiosqlite:///./localdata/pytest_integration.db"
os.environ["DB_TYPE"] = "sqlite"
# Do NOT set DISABLE_REDIS_CACHE_IN_TESTS — use real Redis

# Skip entire directory if DASHSCOPE_API_KEY is missing
pytestmark = pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="DASHSCOPE_API_KEY required for integration tests",
)

DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def test_embedding_model(client: Client) -> dict:
    """Create a test embedding model (text-embedding-v4 via DashScope)."""
    create_payload = {
        "model_id": "text-embedding-v4",
        "model_name": "text-embedding-v4",
        "type": "openai_like",
        "endpoint": DASHSCOPE_ENDPOINT,
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "provider_name": "openai_like",
    }
    response = client.post("/v1/config/embeddings", json=create_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200, f"Failed to create embedding model: {data}"
    emb_data = data["data"]
    yield emb_data
    client.delete(f"/v1/config/embeddings/{emb_data['id']}")


@pytest.fixture(scope="session")
def test_llm_model(client: Client) -> dict:
    """Create a test LLM model (qwen3.5-plus via DashScope)."""
    create_payload = {
        "model_id": "qwen3.5-plus",
        "model_name": "qwen3.5-plus",
        "model": "qwen3.5-plus",
        "base_url": DASHSCOPE_ENDPOINT,
        "api_key": os.environ.get("DASHSCOPE_API_KEY"),
        "temperature": 0.1,
        "context_window": 8192,
        "provider_name": "dashscope",
    }
    response = client.post("/v1/config/llms", json=create_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200, f"Failed to create LLM model: {data}"
    llm_data = data["data"]
    yield llm_data
    client.delete(f"/v1/config/llms/{llm_data['id']}")


@pytest.fixture(scope="session")
def test_knowledgebase(client: Client, test_embedding_model: Any) -> dict:
    """Create a test knowledgebase for integration tests."""
    create_payload = {
        "name": "integ_test_kb",
        "description": "Integration test knowledgebase",
        "embedding_model": test_embedding_model["model_id"],
        "embedding_provider_name": test_embedding_model.get(
            "provider_name", "openai_like"
        ),
        "chunk_config": {
            "parser_type": "structure",
            "chunk_size": 1000,
            "chunk_overlap": 50,
        },
        "retrieval_config": {
            "retrieval_mode": "vector",
            "top_k": 5,
            "similarity_threshold": 0.0,
        },
    }
    response = client.post("/v1/config/knowledgebases", json=create_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 200, f"Failed to create KB: {data}"
    kb_data = data["data"]
    yield kb_data
    client.delete(f"/v1/config/knowledgebases/{kb_data['id']}")
