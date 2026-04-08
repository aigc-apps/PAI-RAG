import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, patch
from service.tool.codesandbox_service import CodesandboxService
from db.models.code_sandbox import CodeSandboxConfigEntity, CodeSandboxConfigCreate
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestCodesandboxService:
    @pytest.fixture
    def service(self, mock_session):
        return CodesandboxService(session=mock_session)

    async def test_get_config_found(self, service, mock_session):
        config = CodeSandboxConfigEntity(id="cs1", tenant_id=TENANT, type="aliyun-fc")
        mock_session.exec.return_value = make_mock_result(first_value=config)
        result = await service.get_codesandbox_config("cs1", TENANT)
        assert result == config

    async def test_get_config_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_codesandbox_config("bad", TENANT)
        assert result is None

    async def test_get_all_configs(self, service, mock_session):
        configs = [CodeSandboxConfigEntity(id="cs1", tenant_id=TENANT, type="aliyun-fc")]
        mock_session.exec.return_value = make_mock_result(all_values=configs)
        result = await service.get_all_codesandbox_configs(TENANT)
        assert len(result) == 1

    @patch("service.tool.codesandbox_service.encrypt_key", return_value="encrypted")
    async def test_create_config(self, mock_encrypt, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        mock_session.refresh = AsyncMock()
        data = CodeSandboxConfigCreate(type="aliyun-fc", aliyun_id="aid", api_key="key")
        result = await service.create_or_update_codesandbox_config(data, TENANT)
        assert result is not None

    async def test_create_config_unsupported_type(self, service, mock_session):
        data = CodeSandboxConfigCreate(type="docker")
        with pytest.raises(ValueError, match="不支持"):
            await service.create_or_update_codesandbox_config(data, TENANT)

    @patch("service.tool.codesandbox_service.encrypt_key", return_value="encrypted")
    async def test_update_config(self, mock_encrypt, service, mock_session):
        existing = CodeSandboxConfigEntity(id="cs1", tenant_id=TENANT, type="aliyun-fc")
        mock_session.exec.return_value = make_mock_result(first_value=existing)
        mock_session.refresh = AsyncMock()
        data = CodeSandboxConfigCreate(type="aliyun-fc", enabled=True)
        result = await service.create_or_update_codesandbox_config(data, TENANT)
        assert result.enabled is True
