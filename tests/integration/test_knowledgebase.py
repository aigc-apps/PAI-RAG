"""Integration tests for knowledgebase CRUD, file upload, and chunk management."""

import io
import os
import pytest
from tests.integration.conftest import pytestmark  # noqa: F401


class TestKnowledgebaseCrud:
    """Tests for KB create, list, get, update, delete."""

    def test_create_knowledgebase(self, client, test_embedding_model):
        """Create a knowledgebase and verify response."""
        payload = {
            "name": "kb_crud_test",
            "description": "Test KB for CRUD",
            "embedding_model": test_embedding_model["model_id"],
            "embedding_provider_name": test_embedding_model.get(
                "provider_name", "openai_like"
            ),
        }
        response = client.post("/v1/config/knowledgebases", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        kb = data["data"]
        assert kb["name"] == "kb_crud_test"
        assert "id" in kb

        # Cleanup
        client.delete(f"/v1/config/knowledgebases/{kb['id']}")

    def test_list_knowledgebases(self, client, test_knowledgebase):
        """List knowledgebases includes the fixture KB."""
        response = client.get("/v1/config/knowledgebases")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        items = data["data"]["items"]
        names = [kb["name"] for kb in items]
        assert test_knowledgebase["name"] in names

    def test_get_knowledgebase_by_id(self, client, test_knowledgebase):
        """Get a specific knowledgebase by id."""
        kb_id = test_knowledgebase["id"]
        response = client.get(f"/v1/config/knowledgebases/{kb_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["id"] == kb_id

    def test_update_knowledgebase(self, client, test_knowledgebase, test_embedding_model):
        """Update a knowledgebase description."""
        kb_id = test_knowledgebase["id"]
        update_payload = {
            "name": test_knowledgebase["name"],
            "description": "Updated description",
            "embedding_model": test_embedding_model["model_id"],
            "embedding_provider_name": test_embedding_model.get(
                "provider_name", "openai_like"
            ),
        }
        response = client.put(
            f"/v1/config/knowledgebases/{kb_id}", json=update_payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["description"] == "Updated description"

    def test_create_duplicate_name_fails(self, client, test_knowledgebase, test_embedding_model):
        """Creating a KB with a duplicate name should fail."""
        payload = {
            "name": test_knowledgebase["name"],
            "description": "Duplicate",
            "embedding_model": test_embedding_model["model_id"],
            "embedding_provider_name": test_embedding_model.get(
                "provider_name", "openai_like"
            ),
        }
        response = client.post("/v1/config/knowledgebases", json=payload)
        # Should return an error (either 400/409 or code != 200)
        data = response.json()
        assert data["code"] != 200 or response.status_code != 200

    def test_get_nonexistent_knowledgebase(self, client):
        """Getting a non-existent KB should return an error."""
        response = client.get(
            "/v1/config/knowledgebases/nonexistent-id-12345"
        )
        # Either 404 or error code in response
        data = response.json()
        assert response.status_code == 404 or data.get("code") != 200

    def test_delete_knowledgebase(self, client, test_embedding_model):
        """Create then delete a knowledgebase."""
        payload = {
            "name": "kb_delete_test",
            "description": "Will be deleted",
            "embedding_model": test_embedding_model["model_id"],
            "embedding_provider_name": test_embedding_model.get(
                "provider_name", "openai_like"
            ),
        }
        resp = client.post("/v1/config/knowledgebases", json=payload)
        kb_id = resp.json()["data"]["id"]

        delete_resp = client.delete(f"/v1/config/knowledgebases/{kb_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["code"] == 200


class TestFileUpload:
    """Tests for file upload (without auto-parse to avoid Celery dependency)."""

    def test_upload_txt_file(self, client, test_knowledgebase):
        """Upload a small .txt file with auto_parse=false."""
        kb_id = test_knowledgebase["id"]
        file_content = b"This is a test document for integration testing."
        files = {"files": ("test_doc.txt", io.BytesIO(file_content), "text/plain")}
        data = {"auto_parse": "false"}

        response = client.post(
            f"/v1/config/knowledgebases/{kb_id}/files",
            files=files,
            data=data,
        )
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["code"] == 200

    def test_list_files(self, client, test_knowledgebase):
        """List files in a knowledgebase."""
        kb_id = test_knowledgebase["id"]
        response = client.get(f"/v1/config/knowledgebases/{kb_id}/files")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_list_files_with_pagination(self, client, test_knowledgebase):
        """List files with pagination parameters."""
        kb_id = test_knowledgebase["id"]
        response = client.get(
            f"/v1/config/knowledgebases/{kb_id}/files?page=1&size=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
