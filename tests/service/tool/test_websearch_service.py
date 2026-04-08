import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, patch
from service.tool.websearch_service import WebsearchService
from db.models.websearch import WebSearchConfigEntity, WebSearchConfigCreate
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestWebsearchService:
    @pytest.fixture
    def service(self, mock_session):
        return WebsearchService(session=mock_session)

    async def test_get_config_found(self, service, mock_session):
        config = WebSearchConfigEntity(id="w1", tenant_id=TENANT, type="tavily")
        mock_session.exec.return_value = make_mock_result(first_value=config)
        result = await service.get_websearch_config("w1", TENANT)
        assert result == config

    async def test_get_config_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_websearch_config("bad", TENANT)
        assert result is None

    async def test_get_all_configs(self, service, mock_session):
        configs = [WebSearchConfigEntity(id="w1", tenant_id=TENANT, type="tavily")]
        mock_session.exec.return_value = make_mock_result(all_values=configs)
        result = await service.get_all_websearch_configs(TENANT)
        assert len(result) == 1

    @patch("service.tool.websearch_service.encrypt_key", return_value="encrypted")
    async def test_create_config_tavily(self, mock_encrypt, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        mock_session.refresh = AsyncMock()
        data = WebSearchConfigCreate(type="tavily", tavily_api_key="key")
        result = await service.create_or_update_websearch_config(data, TENANT)
        assert result is not None

    async def test_create_config_unsupported_type(self, service, mock_session):
        data = WebSearchConfigCreate(type="google")
        with pytest.raises(ValueError, match="不支持"):
            await service.create_or_update_websearch_config(data, TENANT)

    @patch("service.tool.websearch_service.encrypt_key", return_value="encrypted")
    async def test_update_config(self, mock_encrypt, service, mock_session):
        existing = WebSearchConfigEntity(id="w1", tenant_id=TENANT, type="tavily")
        mock_session.exec.return_value = make_mock_result(first_value=existing)
        mock_session.refresh = AsyncMock()
        data = WebSearchConfigCreate(type="aliyun", access_key_id="ak", access_key_secret="sk")
        result = await service.create_or_update_websearch_config(data, TENANT)
        assert result.type == "aliyun"

    async def test_delete_config_success(self, service, mock_session):
        config = WebSearchConfigEntity(id="w1", tenant_id=TENANT, type="tavily")
        mock_session.exec.return_value = make_mock_result(first_value=config)
        await service.delete_websearch_config("w1", TENANT)
        mock_session.delete.assert_called_once()

    async def test_delete_config_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        with pytest.raises(ValueError, match="does not exist"):
            await service.delete_websearch_config("bad", TENANT)
