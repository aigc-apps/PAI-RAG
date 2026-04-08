"""Integration tests for vector retrieval with real embeddings."""

import pytest
from tests.integration.conftest import pytestmark  # noqa: F401


class TestRetrieval:
    """Tests for /v1/retrieval endpoint with real DashScope embeddings."""

    @pytest.fixture(scope="class")
    def kb_with_chunks(self, client, test_knowledgebase):
        """Create a file record and add chunks to the test KB for retrieval."""
        kb_id = test_knowledgebase["id"]

        # Upload a placeholder file (auto_parse=false)
        import io

        file_content = b"Placeholder file for chunk tests."
        files = {"files": ("retrieval_test.txt", io.BytesIO(file_content), "text/plain")}
        data = {"auto_parse": "false"}
        upload_resp = client.post(
            f"/v1/config/knowledgebases/{kb_id}/files",
            files=files,
            data=data,
        )
        assert upload_resp.status_code == 200
        file_list = upload_resp.json()["data"]
        file_id = file_list[0]["id"] if isinstance(file_list, list) else file_list["id"]

        # Add chunks with known text content
        chunks_text = [
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.",
            "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
            "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "The Great Wall of China is a series of fortifications built along the northern borders of China.",
        ]
        chunk_ids = []
        for text in chunks_text:
            resp = client.post(
                f"/v1/config/knowledgebases/{kb_id}/files/{file_id}/chunks",
                json={"text": text, "chunk_metadata": {}},
            )
            assert resp.status_code == 200, f"Failed to add chunk: {resp.json()}"
            chunk_data = resp.json().get("data")
            if chunk_data and "id" in chunk_data:
                chunk_ids.append(chunk_data["id"])

        return {
            "kb_id": kb_id,
            "file_id": file_id,
            "chunk_ids": chunk_ids,
            "chunks_text": chunks_text,
        }

    def test_retrieve_matching_query(self, client, kb_with_chunks):
        """Retrieve chunks with a query matching known content."""
        payload = {
            "query": "What is the Eiffel Tower?",
            "knowledge_id": kb_with_chunks["kb_id"],
        }
        response = client.post("/v1/retrieval", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Response is {"records": [...]} directly (not wrapped in ResponseModel)
        records = data["records"]
        assert len(records) > 0
        # The top result should be related to the Eiffel Tower
        top_content = records[0]["content"].lower()
        assert "eiffel" in top_content or "tower" in top_content or "paris" in top_content

    def test_retrieve_with_high_threshold_returns_fewer(self, client, kb_with_chunks):
        """Retrieve with a very high similarity threshold should return fewer or no results."""
        payload = {
            "query": "completely unrelated quantum physics topic",
            "knowledge_id": kb_with_chunks["kb_id"],
            "retrieval_setting": {
                "similarity_threshold": 0.99,
                "top_k": 5,
            },
        }
        response = client.post("/v1/retrieval", json=payload)
        assert response.status_code == 200
        data = response.json()
        records = data["records"]
        # With very high threshold and unrelated query, should get few/no results
        assert len(records) <= 1

    def test_retrieve_with_top_k(self, client, kb_with_chunks):
        """Retrieve with top_k=1 returns at most 1 result."""
        payload = {
            "query": "machine learning artificial intelligence",
            "knowledge_id": kb_with_chunks["kb_id"],
            "retrieval_setting": {
                "top_k": 1,
            },
        }
        response = client.post("/v1/retrieval", json=payload)
        assert response.status_code == 200
        data = response.json()
        records = data["records"]
        assert len(records) <= 1

    def test_retrieve_response_structure(self, client, kb_with_chunks):
        """Verify retrieval response has expected fields."""
        payload = {
            "query": "Great Wall of China",
            "knowledge_id": kb_with_chunks["kb_id"],
        }
        response = client.post("/v1/retrieval", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        if data["records"]:
            record = data["records"][0]
            assert "content" in record
            assert "score" in record

    def test_retrieve_nonexistent_kb(self, client):
        """Retrieve from a non-existent KB should return an error."""
        payload = {
            "query": "test query",
            "knowledge_id": "nonexistent-kb-id-99999",
        }
        response = client.post("/v1/retrieval", json=payload)
        # Should be an error response (400 or 500)
        assert response.status_code in (400, 500)
