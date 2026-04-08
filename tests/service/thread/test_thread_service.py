import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.thread.thread_service import ThreadService
from db.models.thread import ThreadEntity, ThreadCreate
from tests.service.conftest import make_mock_result


TENANT = "test_tenant"


class TestThreadService:
    @pytest.fixture
    def service(self, mock_session):
        return ThreadService(session=mock_session)

    async def test_get_thread_found(self, service, mock_session):
        thread = ThreadEntity(id="t1", tenant_id=TENANT, title="Test")
        mock_session.exec.return_value = make_mock_result(first_value=thread)
        result = await service.get_thread("t1", TENANT)
        assert result == thread

    async def test_get_thread_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_thread("nonexistent", TENANT)
        assert result is None

    async def test_list_threads(self, service, mock_session):
        threads = [ThreadEntity(id=f"t{i}", tenant_id=TENANT) for i in range(3)]
        mock_session.exec.return_value = make_mock_result(all_values=threads)
        result = await service.list_threads(TENANT, offset=0, limit=10)
        assert len(result) == 3

    async def test_create_thread(self, service, mock_session):
        data = ThreadCreate(title="New Thread")
        mock_session.refresh = AsyncMock(side_effect=lambda t: setattr(t, 'id', 'new_id'))
        result = await service.create_thread(data, TENANT)
        assert result is not None
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    async def test_create_thread_integrity_error(self, service, mock_session):
        from sqlalchemy.exc import IntegrityError
        data = ThreadCreate(title="Dup")
        mock_session.flush.side_effect = IntegrityError("stmt", {}, Exception("UniqueViolationError"))
        with pytest.raises(ValueError, match="already exists"):
            await service.create_thread(data, TENANT)

    async def test_update_thread_title(self, service, mock_session):
        thread = ThreadEntity(id="t1", tenant_id=TENANT, title="Old")
        mock_session.exec.return_value = make_mock_result(first_value=thread)
        result = await service.update_thread_title("t1", "New Title", TENANT)
        assert result.title == "New Title"

    async def test_update_thread_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        with pytest.raises(ValueError, match="not found"):
            await service.update_thread_title("bad_id", "Title", TENANT)

    async def test_delete_thread(self, service, mock_session):
        thread = ThreadEntity(id="t1", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=thread)
        await service.delete_thread("t1", TENANT)
        mock_session.delete.assert_called_once_with(thread)
        mock_session.flush.assert_called_once()

    async def test_delete_thread_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        with pytest.raises(ValueError, match="not found"):
            await service.delete_thread("bad_id", TENANT)
