import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, patch
from service.tool.mcpserver_service import McpserverService
from db.models.mcp import McpServerEntity, McpServerCreate
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestMcpserverService:
    @pytest.fixture
    def service(self, mock_session):
        return McpserverService(session=mock_session)

    async def test_get_mcpserver_found(self, service, mock_session):
        mcp = McpServerEntity(id="mcp1", name="test-mcp", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=mcp)
        result = await service.get_mcpserver("mcp1", TENANT)
        assert result == mcp

    async def test_get_mcpserver_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_mcpserver("bad", TENANT)
        assert result is None

    async def test_get_mcpserver_by_name(self, service, mock_session):
        mcp = McpServerEntity(id="mcp1", name="test-mcp", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=mcp)
        result = await service.get_mcpserver_by_name("test-mcp", TENANT)
        assert result.name == "test-mcp"

    async def test_get_mcpserver_by_ids(self, service, mock_session):
        mcps = [McpServerEntity(id=f"mcp{i}", name=f"mcp-{i}", tenant_id=TENANT) for i in range(2)]
        mock_session.exec.return_value = make_mock_result(all_values=mcps)
        result = await service.get_mcpserver_by_ids(["mcp0", "mcp1"], TENANT)
        assert len(result) == 2

    async def test_list_mcpservers(self, service, mock_session):
        mcps = [McpServerEntity(id=f"mcp{i}", name=f"n{i}", tenant_id=TENANT) for i in range(3)]
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=3)
            else:
                return make_mock_result(all_values=mcps)
        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_mcpservers(TENANT)
        assert result.total == 3
        assert len(result.items) == 3

    @patch("service.tool.mcpserver_service.encrypt_key", return_value="encrypted")
    async def test_create_mcpserver(self, mock_encrypt, service, mock_session):
        mock_session.refresh = AsyncMock()
        data = McpServerCreate(name="test-mcp", url="http://mcp.example.com", type="sse")
        result = await service.create_mcpserver(data, TENANT)
        assert result is not None
        mock_session.add.assert_called_once()

    @patch("service.tool.mcpserver_service.encrypt_key", return_value="encrypted")
    async def test_update_mcpserver(self, mock_encrypt, service, mock_session):
        mcp = McpServerEntity(id="mcp1", name="old", url="http://old.com", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=mcp)
        mock_session.refresh = AsyncMock()
        data = McpServerCreate(name="new", url="http://new.com")
        result = await service.update_mcpserver("mcp1", data, TENANT)
        assert result.name == "new"

    async def test_update_mcpserver_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        data = McpServerCreate(name="x", url="http://x.com")
        with pytest.raises(ValueError, match="does not exist"):
            await service.update_mcpserver("bad", data, TENANT)

    async def test_delete_mcpserver(self, service, mock_session):
        mcp = McpServerEntity(id="mcp1", name="test", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=mcp)
        await service.delete_mcpserver("mcp1", TENANT)
        mock_session.delete.assert_called_once_with(mcp)

    async def test_delete_mcpserver_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        with pytest.raises(ValueError, match="does not exist"):
            await service.delete_mcpserver("bad", TENANT)

    async def test_get_all_mcpservers(self, service, mock_session):
        mcps = [McpServerEntity(id=f"mcp{i}", name=f"n{i}", tenant_id=TENANT) for i in range(2)]
        mock_session.exec.return_value = make_mock_result(all_values=mcps)
        result = await service.get_all_mcpservers(TENANT)
        assert len(result) == 2
