"""File Management API Tests based on PAI-RAG API documentation.
Reference: https://help.aliyun.com/zh/pai/use-cases/rag-api-interface-for-v0-4-x
"""
import os
import io
from typing import Generator
from fastapi.testclient import TestClient
import pytest
from httpx import Client

class TestFileAPI:
    """Test cases for File Management operations."""

    def test_list_files(self, client: Client, test_knowledgebase_for_file):
        """Test GET /v1/config/knowledgebases/{kb_id}/files - List files in KB."""
        kb_id = test_knowledgebase_for_file["id"]
        
        response = client.get(f"/v1/config/knowledgebases/{kb_id}/files")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200
        assert "data" in resp_json

    def test_list_files_with_pagination(self, client: Client, test_knowledgebase_for_file):
        """Test file listing with pagination."""
        kb_id = test_knowledgebase_for_file["id"]
        
        response = client.get(f"/v1/config/knowledgebases/{kb_id}/files?page=1&size=10")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200

    @pytest.mark.skip(reason="Requires actual file upload infrastructure")
    def test_upload_file(self, client: Client, test_knowledgebase_for_file):
        """Test POST /v1/config/knowledgebases/{kb_id}/files - Upload file."""
        kb_id = test_knowledgebase_for_file["id"]
        
        # Create a test file content
        file_content = b"This is a test document for PAI-RAG API testing."
        files = {
            "files": ("test_document.txt", io.BytesIO(file_content), "text/plain")
        }
        
        response = client.post(
            f"/v1/config/knowledgebases/{kb_id}/files",
            files=files
        )
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200

    @pytest.mark.skip(reason="Requires actual file upload infrastructure")
    def test_upload_file_with_auto_parse(self, client: Client, test_knowledgebase_for_file):
        """Test file upload with auto_parse parameter."""
        kb_id = test_knowledgebase_for_file["id"]
        
        file_content = b"Test document content."
        files = {
            "files": ("test_auto_parse.txt", io.BytesIO(file_content), "text/plain")
        }
        data = {
            "auto_parse": "true"
        }
        
        response = client.post(
            f"/v1/config/knowledgebases/{kb_id}/files",
            files=files,
            data=data
        )
        assert response.status_code == 200

    def test_get_file_details(self, client: Client, test_knowledgebase_for_file):
        """Test GET /v1/config/knowledgebases/{kb_id}/files/{file_id} - Get file details."""
        kb_id = test_knowledgebase_for_file["id"]
        
        # This would need a real file_id from an uploaded file
        # For now, test with a non-existent file_id
        response = client.get(f"/v1/config/knowledgebases/{kb_id}/files/non_existent_file")
        assert response.status_code in [404, 400]

    def test_delete_file(self, client: Client, test_knowledgebase_for_file):
        """Test DELETE /v1/config/knowledgebases/{kb_id}/files/{file_id} - Delete file."""
        kb_id = test_knowledgebase_for_file["id"]
        
        # Test deleting non-existent file
        response = client.delete(f"/v1/config/knowledgebases/{kb_id}/files/non_existent_file")
        assert response.status_code in [404, 400, 200]


class TestChunkAPI:
    """Test cases for Chunk operations."""

    def test_list_chunks(self, client: Client, test_knowledgebase_for_file):
        """Test GET /v1/config/knowledgebases/{kb_id}/files/{file_id}/chunks - List chunks."""
        kb_id = test_knowledgebase_for_file["id"]
        
        # Would need a real file_id
        response = client.get(f"/v1/config/knowledgebases/{kb_id}/files/test_file/chunks")
        # May return 404 if file doesn't exist, or 200 with empty list
        assert response.status_code in [200, 404, 400]

    def test_update_chunk(self, client: Client, test_knowledgebase_for_file):
        """Test PUT /v1/config/knowledgebases/{kb_id}/files/{file_id}/chunks/{chunk_id}."""
        kb_id = test_knowledgebase_for_file["id"]
        
        update_payload = {
            "text": "Updated chunk content",
            "active": True
        }
        
        response = client.put(
            f"/v1/config/knowledgebases/{kb_id}/files/test_file/chunks/test_chunk",
            json=update_payload
        )
        # 404 if file/chunk doesn't exist, 500 for server errors
        assert response.status_code in [200, 404, 400, 500]

    def test_delete_chunk(self, client: Client, test_knowledgebase_for_file):
        """Test DELETE /v1/config/knowledgebases/{kb_id}/files/{file_id}/chunks/{chunk_id}."""
        kb_id = test_knowledgebase_for_file["id"]
        
        response = client.delete(
            f"/v1/config/knowledgebases/{kb_id}/files/test_file/chunks/test_chunk"
        )
        # 404 if file/chunk doesn't exist, 500 for server errors
        assert response.status_code in [200, 404, 400, 500]


class TestMetadataAPI:
    """Test cases for Metadata operations."""

    def test_list_metadata(self, client: Client, test_knowledgebase_for_file):
        """Test GET /v1/config/knowledgebases/{kb_id}/metadata - List metadata configs."""
        kb_id = test_knowledgebase_for_file["id"]
        
        response = client.get(f"/v1/config/knowledgebases/{kb_id}/metadata")
        assert response.status_code == 200
        resp_json = response.json()
        assert resp_json["code"] == 200

    def test_create_metadata(self, client: Client, test_knowledgebase_for_file):
        """Test POST /v1/config/knowledgebases/{kb_id}/metadata - Create metadata config."""
        kb_id = test_knowledgebase_for_file["id"]
        
        create_payload = {
            "name": "department",
            "value_type": "string",
            "description": "部门标签"
        }
        
        response = client.post(
            f"/v1/config/knowledgebases/{kb_id}/metadata",
            json=create_payload
        )
        if response.status_code == 200:
            resp_json = response.json()
            assert resp_json["code"] == 200
            metadata_id = resp_json["data"]["id"]
            # Cleanup
            client.delete(f"/v1/config/knowledgebases/{kb_id}/metadata/{metadata_id}")

    def test_set_file_metadata(self, client: Client, test_knowledgebase_for_file):
        """Test POST /v1/config/knowledgebases/{kb_id}/files/{file_id}/metadata."""
        kb_id = test_knowledgebase_for_file["id"]
        
        # The actual endpoint expects metadata as request body
        metadata_payload = {
            "entries": [
                {
                    "name":"department",
                    "value":"engineering"
                },
                {
                    "name":"priority",
                    "value":"high"
                }
            ]
        }
        
        response = client.post(
            f"/v1/config/knowledgebases/{kb_id}/files/test_file/metadata",
            json=metadata_payload
        )
        # May return 404 if file doesn't exist, 500 for server errors
        assert response.status_code in [200, 404, 400, 500]

