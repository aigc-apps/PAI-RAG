import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.tool.trace_service import TraceService, _load_trace_config_from_env
from db.models.trace import TraceModel, TraceModelEntity
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestLoadTraceConfigFromEnv:
    @patch.dict(os.environ, {"TRACE_ENDPOINT": "http://trace.example.com"}, clear=False)
    def test_loads_from_env(self):
        config = _load_trace_config_from_env()
        assert config is not None
        assert config.endpoint == "http://trace.example.com"

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_when_no_env(self):
        config = _load_trace_config_from_env()
        assert config is None


class TestTraceService:
    @pytest.fixture
    def service(self, mock_session):
        return TraceService(session=mock_session)

    async def test_get_trace_config_found(self, service, mock_session):
        config = TraceModelEntity(id="tr1", tenant_id=TENANT, endpoint="http://x")
        mock_session.exec.return_value = make_mock_result(first_value=config)
        result = await service.get_trace_config(TENANT)
        assert result == config

    async def test_get_trace_config_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_trace_config(TENANT)
        assert result is None

    @patch("service.tool.trace_service.init_instrument")
    async def test_create_trace_config(self, mock_init, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        mock_session.refresh = AsyncMock()
        data = TraceModel(endpoint="http://trace.example.com", service_name="test-svc", enabled=True, token="test-token")
        result = await service.create_or_update_trace_config(data, TENANT)
        assert result is not None
        mock_session.add.assert_called()

    @patch("service.tool.trace_service.init_instrument")
    async def test_update_trace_config(self, mock_init, service, mock_session):
        existing = TraceModelEntity(id="tr1", tenant_id=TENANT, endpoint="old", service_name="old-svc")
        mock_session.exec.return_value = make_mock_result(first_value=existing)
        mock_session.refresh = AsyncMock()
        data = TraceModel(endpoint="new_endpoint", service_name="new-svc")
        result = await service.create_or_update_trace_config(data, TENANT)
        assert result.endpoint == "new_endpoint"
