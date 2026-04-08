import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from db.models.trace import TraceModel, TraceModelEntity
from service.injection import get_trace_service
from app.main import app


class TestTraceAPI:
    @pytest.fixture(autouse=True)
    def setup_trace_service(self, api_client):
        self.mock_service = AsyncMock()
        app.dependency_overrides[get_trace_service] = lambda: self.mock_service
        self.client = api_client

    def test_set_trace_config_success(self):
        mock_entity = MagicMock()
        mock_entity.model_dump = MagicMock(return_value={
            "id": "tr1",
            "tenant_id": "test-tenant",
            "endpoint": "http://trace.example.com",
            "token": "tok",
            "service_name": "svc",
            "enabled": True,
        })
        # Make it JSON serializable
        mock_entity.__class__ = type("TraceModelEntity", (), {})
        self.mock_service.create_or_update_trace_config.return_value = mock_entity

        response = self.client.post("/v1/config/trace", json={
            "endpoint": "http://trace.example.com",
            "token": "tok",
            "service_name": "svc",
            "enabled": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "success" in data["message"].lower()

    def test_set_trace_config_error(self):
        self.mock_service.create_or_update_trace_config.side_effect = Exception("db error")
        response = self.client.post("/v1/config/trace", json={
            "endpoint": "http://trace.example.com",
            "token": "tok",
            "service_name": "svc",
            "enabled": True,
        })
        assert response.status_code == 500

    def test_get_trace_config_success(self):
        mock_entity = MagicMock()
        mock_entity.model_dump = MagicMock(return_value={
            "id": "tr1",
            "endpoint": "http://trace.example.com",
            "token": "tok",
            "service_name": "svc",
            "enabled": True,
        })
        mock_entity.__class__ = type("TraceModelEntity", (), {})
        self.mock_service.get_trace_config.return_value = mock_entity

        response = self.client.get("/v1/config/trace")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_get_trace_config_not_found_returns_default(self):
        self.mock_service.get_trace_config.return_value = None

        response = self.client.get("/v1/config/trace")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_get_trace_config_error(self):
        self.mock_service.get_trace_config.side_effect = Exception("db error")
        response = self.client.get("/v1/config/trace")
        assert response.status_code == 500
