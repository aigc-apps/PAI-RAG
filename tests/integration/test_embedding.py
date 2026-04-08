"""Integration tests for the /v1/embeddings endpoint with real DashScope model."""

import pytest
from tests.integration.conftest import pytestmark  # noqa: F401


class TestEmbeddingEndpoint:
    """Tests for POST /v1/embeddings with real text-embedding-v4."""

    def test_embed_single_string(self, client, test_embedding_model):
        """Embed a single string and verify vector returned."""
        payload = {
            "input": "Hello, world!",
            "model": test_embedding_model["model_id"],
        }
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        embedding = data["data"][0]
        assert embedding["object"] == "embedding"
        assert embedding["index"] == 0
        assert isinstance(embedding["embedding"], list)
        assert len(embedding["embedding"]) > 0
        # Verify it's a list of floats
        assert all(isinstance(v, (int, float)) for v in embedding["embedding"][:10])

    def test_embed_batch_strings(self, client, test_embedding_model):
        """Embed a list of strings and verify batch results."""
        texts = ["First document", "Second document", "Third document"]
        payload = {
            "input": texts,
            "model": test_embedding_model["model_id"],
        }
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == len(texts)
        for i, emb in enumerate(data["data"]):
            assert emb["index"] == i
            assert isinstance(emb["embedding"], list)
            assert len(emb["embedding"]) > 0

    def test_embed_empty_input(self, client, test_embedding_model):
        """Empty string input should either return error or an embedding."""
        payload = {"input": "", "model": test_embedding_model["model_id"]}
        response = client.post("/v1/embeddings", json=payload)
        # Some embedding models accept empty strings, some reject them
        assert response.status_code in (200, 400, 422)

    def test_embed_nonexistent_model_returns_error(self, client):
        """Using a nonexistent model should return an error."""
        payload = {
            "input": "test text",
            "model": "nonexistent-model-xyz",
        }
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 400 or response.status_code == 500
