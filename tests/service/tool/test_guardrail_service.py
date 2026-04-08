import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.tool.guardrail_service import GuardrailService
from db.models.guardrail import GuardrailConfigEntity, GuardrailConfigCreate
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestGuardrailService:
    @pytest.fixture
    def service(self, mock_session):
        return GuardrailService(session=mock_session)

    async def test_get_guardrail_config_found(self, service, mock_session):
        config = GuardrailConfigEntity(id="g1", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=config)
        result = await service.get_guardrail_config("g1", TENANT)
        assert result == config

    async def test_get_guardrail_config_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_guardrail_config("bad", TENANT)
        assert result is None

    async def test_get_all_guardrail_configs(self, service, mock_session):
        configs = [GuardrailConfigEntity(id="g1", tenant_id=TENANT)]
        mock_session.exec.return_value = make_mock_result(all_values=configs)
        result = await service.get_all_guardrail_configs(TENANT)
        assert len(result) == 1

    @patch("service.tool.guardrail_service.encrypt_key", return_value="encrypted")
    async def test_create_guardrail_config(self, mock_encrypt, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        mock_session.refresh = AsyncMock()
        data = GuardrailConfigCreate(
            access_key_id="ak",
            access_key_secret="sk",
            endpoint="https://green.cn-hangzhou.aliyuncs.com",
            region_name="cn-hangzhou",
        )
        result = await service.create_or_update_guardrail_config(data, TENANT)
        assert result is not None
        mock_session.add.assert_called()

    @patch("service.tool.guardrail_service.encrypt_key", return_value="encrypted")
    async def test_update_guardrail_config(self, mock_encrypt, service, mock_session):
        existing = GuardrailConfigEntity(id="g1", tenant_id=TENANT, endpoint="old")
        mock_session.exec.return_value = make_mock_result(first_value=existing)
        mock_session.refresh = AsyncMock()
        data = GuardrailConfigCreate(endpoint="new_endpoint")
        result = await service.create_or_update_guardrail_config(data, TENANT)
        assert result.endpoint == "new_endpoint"
