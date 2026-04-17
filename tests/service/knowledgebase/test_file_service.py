import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.knowledgebase.file_service import FileService
from db.models.knowledgebase.file import KbFileEntity
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestFileService:
    @pytest.fixture
    def service(self, mock_session):
        return FileService(session=mock_session)

    async def test_get_file_by_id_found(self, service, mock_session):
        f = KbFileEntity(id="f1", kb_id="kb1", file_name="test.pdf", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=f)
        result = await service.get_file_by_id("f1", TENANT)
        assert result == f

    async def test_get_file_by_id_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_file_by_id("bad", TENANT)
        assert result is None

    async def test_get_files_by_ids(self, service, mock_session):
        files = [KbFileEntity(id=f"f{i}", kb_id="kb1", file_name=f"f{i}.pdf", tenant_id=TENANT) for i in range(3)]
        mock_session.exec.return_value = make_mock_result(all_values=files)
        result = await service.get_files_by_ids(["f0", "f1", "f2"], TENANT)
        assert len(result) == 3

    async def test_get_files_by_ids_empty(self, service, mock_session):
        result = await service.get_files_by_ids([], TENANT)
        assert result == []

    async def test_get_file(self, service, mock_session):
        f = KbFileEntity(id="f1", kb_id="kb1", file_name="test.pdf", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=f)
        result = await service.get_file("kb1", "f1", TENANT)
        assert result == f

    async def test_get_files_by_names_empty(self, service, mock_session):
        result = await service.get_files_by_names("kb1", [], TENANT)
        assert result == []

    async def test_list_files(self, service, mock_session):
        files = [KbFileEntity(id=f"f{i}", kb_id="kb1", file_name=f"f{i}.pdf", tenant_id=TENANT) for i in range(2)]
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=2)
            else:
                return make_mock_result(all_values=files)
        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_files("kb1", TENANT)
        assert result.total == 2
