import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock
from service.knowledgebase.file_task_service import FileTaskService
from db.models.knowledgebase.file_task import KbFileTaskEntity
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestFileTaskService:
    @pytest.fixture
    def service(self, mock_session):
        return FileTaskService(session=mock_session)

    async def test_get_file_task_found(self, service, mock_session):
        task = KbFileTaskEntity(id="t1", kb_id="kb1", file_id="f1", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=task)
        result = await service.get_file_task("t1", TENANT)
        assert result == task

    async def test_get_file_task_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_file_task("bad", TENANT)
        assert result is None

    async def test_get_file_task_by_file_and_part(self, service, mock_session):
        task = KbFileTaskEntity(id="t1", kb_id="kb1", file_id="f1", file_part=0, tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=task)
        result = await service.get_file_task_by_file_and_part("kb1", "f1", 0, TENANT)
        assert result.file_part == 0

    async def test_list_file_tasks(self, service, mock_session):
        tasks = [KbFileTaskEntity(id=f"t{i}", kb_id="kb1", file_id="f1", tenant_id=TENANT) for i in range(3)]
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=3)
            else:
                return make_mock_result(all_values=tasks)
        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_file_tasks("kb1", TENANT)
        assert result.total == 3
