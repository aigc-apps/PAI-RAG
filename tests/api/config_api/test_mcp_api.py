import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.injection import get_mcpserver_service
from app.main import app


class TestMcpAPI:
    @pytest.fixture(autouse=True)
    def setup_mcp_service(self, api_client):
        self.mock_service = AsyncMock()
        app.dependency_overrides[get_mcpserver_service] = lambda: self.mock_service
        self.client = api_client

    def _make_mcp_entity(self, mcp_id="mcp1", name="test-mcp"):
        entity = MagicMock()
        entity.model_dump = MagicMock(return_value={
            "id": mcp_id,
            "name": name,
            "url": "http://mcp.example.com",
            "type": "sse",
            "enabled": True,
            "need_token": False,
            "tenant_id": "test-tenant",
        })
        entity.__class__ = type("McpServerEntity", (), {})
        return entity

    def test_create_mcp_success(self):
        self.mock_service.create_mcpserver.return_value = self._make_mcp_entity()

        response = self.client.post("/v1/config/mcps", json={
            "name": "test-mcp",
            "url": "http://mcp.example.com",
            "type": "sse",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_create_mcp_value_error(self):
        self.mock_service.create_mcpserver.side_effect = ValueError("duplicate name")
        response = self.client.post("/v1/config/mcps", json={
            "name": "test-mcp",
            "url": "http://mcp.example.com",
        })
        assert response.status_code == 400

    def test_create_mcp_error(self):
        self.mock_service.create_mcpserver.side_effect = Exception("db error")
        response = self.client.post("/v1/config/mcps", json={
            "name": "test-mcp",
            "url": "http://mcp.example.com",
        })
        assert response.status_code == 500

    def test_list_mcps_success(self):
        mock_result = MagicMock()
        mock_result.model_dump = MagicMock(return_value={
            "items": [{"id": "mcp1", "name": "test"}],
            "total": 1,
            "pages": 1,
            "page": 1,
            "size": 10,
        })
        mock_result.__class__ = type("PagedResult", (), {})
        self.mock_service.list_mcpservers.return_value = mock_result

        response = self.client.get("/v1/config/mcps")
        assert response.status_code == 200

    def test_list_mcps_by_name(self):
        self.mock_service.get_mcpserver_by_name.return_value = self._make_mcp_entity()

        response = self.client.get("/v1/config/mcps?name=test-mcp")
        assert response.status_code == 200

    def test_list_mcps_by_name_not_found(self):
        self.mock_service.get_mcpserver_by_name.return_value = None

        response = self.client.get("/v1/config/mcps?name=nonexistent")
        # ``ApiException.not_found`` raises HTTP 404; the unified exception
        # decorator re-raises HTTPException as-is so the original status code
        # is preserved instead of being collapsed to 500.
        assert response.status_code == 404

    def test_read_mcp_success(self):
        self.mock_service.get_mcpserver.return_value = self._make_mcp_entity()

        response = self.client.get("/v1/config/mcps/mcp1")
        assert response.status_code == 200

    def test_read_mcp_not_found(self):
        self.mock_service.get_mcpserver.return_value = None

        response = self.client.get("/v1/config/mcps/nonexistent")
        # ``ApiException.not_found`` raises HTTP 404; the unified exception
        # decorator re-raises HTTPException as-is so the original status code
        # is preserved instead of being collapsed to 500.
        assert response.status_code == 404

    def test_update_mcp_success(self):
        self.mock_service.update_mcpserver.return_value = self._make_mcp_entity()

        response = self.client.put("/v1/config/mcps/mcp1", json={
            "name": "updated-mcp",
            "url": "http://new.example.com",
        })
        assert response.status_code == 200

    def test_update_mcp_value_error(self):
        self.mock_service.update_mcpserver.side_effect = ValueError("not found")
        response = self.client.put("/v1/config/mcps/mcp1", json={
            "name": "updated-mcp",
            "url": "http://new.example.com",
        })
        assert response.status_code == 400

    def test_delete_mcp_success(self):
        self.mock_service.delete_mcpserver.return_value = None

        response = self.client.delete("/v1/config/mcps/mcp1")
        assert response.status_code == 200

    def test_delete_mcp_value_error(self):
        self.mock_service.delete_mcpserver.side_effect = ValueError("not found")
        response = self.client.delete("/v1/config/mcps/mcp1")
        assert response.status_code == 400

    def test_delete_mcp_error(self):
        self.mock_service.delete_mcpserver.side_effect = Exception("db error")
        response = self.client.delete("/v1/config/mcps/mcp1")
        assert response.status_code == 500
