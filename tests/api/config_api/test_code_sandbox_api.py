import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.injection import get_codesandbox_service
from app.main import app


class TestCodeSandboxAPI:
    @pytest.fixture(autouse=True)
    def setup_codesandbox_service(self, api_client):
        self.mock_service = AsyncMock()
        app.dependency_overrides[get_codesandbox_service] = lambda: self.mock_service
        self.client = api_client

    def test_add_code_sandbox_config_success(self, mock_session):
        mock_entity = MagicMock()
        mock_entity.model_dump = MagicMock(return_value={
            "id": "cs1",
            "type": "aliyun-fc",
            "enabled": True,
            "timeout_default": 50,
        })
        mock_entity.__class__ = type("CodeSandboxConfigEntity", (), {})
        self.mock_service.create_or_update_codesandbox_config.return_value = mock_entity
        mock_session.refresh = AsyncMock()

        response = self.client.post("/v1/config/code_sandbox", json={
            "type": "aliyun-fc",
            "enabled": True,
            "api_key": "test-key",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_add_code_sandbox_config_unsupported_type(self):
        response = self.client.post("/v1/config/code_sandbox", json={
            "type": "docker",
            "enabled": True,
        })
        assert response.status_code == 400

    def test_add_code_sandbox_config_error(self, mock_session):
        self.mock_service.create_or_update_codesandbox_config.side_effect = Exception("db error")
        response = self.client.post("/v1/config/code_sandbox", json={
            "type": "aliyun-fc",
            "enabled": True,
        })
        assert response.status_code == 400

    def test_list_code_sandbox_config_success(self):
        mock_entity = MagicMock()
        mock_entity.type = "aliyun-fc"
        mock_entity.aliyun_id = "aid"
        mock_entity.interpreter_id = "iid"
        mock_entity.interpreter_name = "iname"
        mock_entity.enabled = True
        mock_entity.timeout_default = 50
        mock_entity.encrypted_api_key = None
        mock_entity.id = "cs1"
        self.mock_service.get_codesandbox_config_or_create.return_value = mock_entity

        response = self.client.get("/v1/config/code_sandbox")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_list_code_sandbox_config_not_found(self):
        self.mock_service.get_codesandbox_config_or_create.return_value = None

        response = self.client.get("/v1/config/code_sandbox")
        assert response.status_code == 200

    def test_list_code_sandbox_config_error(self):
        self.mock_service.get_codesandbox_config_or_create.side_effect = Exception("db error")
        response = self.client.get("/v1/config/code_sandbox")
        assert response.status_code == 400
