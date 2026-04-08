import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.injection import get_guardrail_service
from app.main import app


class TestGuardrailAPI:
    @pytest.fixture(autouse=True)
    def setup_guardrail_service(self, api_client):
        self.mock_service = AsyncMock()
        app.dependency_overrides[get_guardrail_service] = lambda: self.mock_service
        self.client = api_client

    def test_add_guardrail_config_success(self):
        mock_entity = MagicMock()
        mock_entity.model_dump = MagicMock(return_value={
            "id": "g1",
            "tenant_id": "test-tenant",
            "region_name": "cn-hangzhou",
            "region_id": "cn-hangzhou",
            "endpoint": "https://green.cn-hangzhou.aliyuncs.com",
        })
        mock_entity.__class__ = type("GuardrailConfigEntity", (), {})
        self.mock_service.create_or_update_guardrail_config.return_value = mock_entity

        response = self.client.post("/v1/config/guardrail", json={
            "region_name": "cn-hangzhou",
            "region_id": "cn-hangzhou",
            "endpoint": "https://green.cn-hangzhou.aliyuncs.com",
            "access_key_id": "ak",
            "access_key_secret": "sk",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "success" in data["message"].lower()

    def test_add_guardrail_config_error(self):
        self.mock_service.create_or_update_guardrail_config.side_effect = Exception("db error")
        response = self.client.post("/v1/config/guardrail", json={
            "region_name": "cn-hangzhou",
            "access_key_id": "ak",
            "access_key_secret": "sk",
        })
        assert response.status_code == 500

    def test_list_guardrail_configs_success(self):
        mock_entity = MagicMock()
        mock_entity.model_dump = MagicMock(return_value={
            "id": "g1",
            "region_name": "cn-hangzhou",
            "region_id": "cn-hangzhou",
            "endpoint": "https://green.cn-hangzhou.aliyuncs.com",
        })
        mock_entity.__class__ = type("GuardrailConfigEntity", (), {})
        self.mock_service.get_all_guardrail_configs.return_value = [mock_entity]

        response = self.client.get("/v1/config/guardrail")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_list_guardrail_configs_empty(self):
        self.mock_service.get_all_guardrail_configs.return_value = []
        response = self.client.get("/v1/config/guardrail")
        assert response.status_code == 200

    def test_list_guardrail_configs_error(self):
        self.mock_service.get_all_guardrail_configs.side_effect = Exception("db error")
        response = self.client.get("/v1/config/guardrail")
        assert response.status_code == 500
