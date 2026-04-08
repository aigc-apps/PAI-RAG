import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.thread.message_service import MessageService
from db.models.message import MessageEntity, MessageCreate
from tests.service.conftest import make_mock_result


TENANT = "test_tenant"


class TestMessageService:
    @pytest.fixture
    def service(self, mock_session):
        return MessageService(session=mock_session)

    async def test_get_message_found(self, service, mock_session):
        msg = MessageEntity(id="m1", thread_id="t1", tenant_id=TENANT, role="user", content="hi")
        mock_session.exec.return_value = make_mock_result(first_value=msg)
        result = await service.get_message("m1", TENANT)
        assert result == msg

    async def test_get_message_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_message("nonexistent", TENANT)
        assert result is None

    async def test_list_messages(self, service, mock_session):
        msgs = [MessageEntity(id=f"m{i}", thread_id="t1", tenant_id=TENANT, role="user", content=f"msg{i}") for i in range(3)]
        mock_session.exec.return_value = make_mock_result(all_values=msgs)
        result = await service.list_messages("t1", TENANT)
        assert len(result) == 3

    async def test_get_message_ids_by_thread(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(all_values=["m1", "m2"])
        result = await service.get_message_ids_by_thread("t1", TENANT)
        assert result == ["m1", "m2"]

    async def test_create_message_new(self, service, mock_session):
        data = MessageCreate(thread_id="t1", role="user", content=[{"type": "text", "text": "hello"}], attachments=[])
        mock_session.refresh = AsyncMock()
        result = await service.create_message(data, TENANT)
        assert result is not None
        mock_session.add.assert_called()
        mock_session.flush.assert_called_once()

    async def test_create_message_with_local_id_update(self, service, mock_session):
        existing = MessageEntity(id="m1", thread_id="t1", tenant_id=TENANT, role="user", content=[{"type": "text", "text": "old"}], local_id="loc1")
        mock_session.exec.return_value = make_mock_result(first_value=existing)

        data = MessageCreate(thread_id="t1", role="user", content=[{"type": "text", "text": "new"}], local_id="loc1", attachments=[])
        mock_session.refresh = AsyncMock()
        result = await service.create_message(data, TENANT)
        assert result.content == [{"type": "text", "text": "new"}]

    async def test_create_message_with_attachments(self, service, mock_session):
        attachment_file = MagicMock()
        attachment_file.message_id = None
        mock_session.get = AsyncMock(return_value=attachment_file)

        data = MessageCreate(
            thread_id="t1",
            role="user",
            content=[{"type": "text", "text": "with file"}],
            attachments=[{"id": "file1", "name": "test.pdf"}],
        )
        mock_session.refresh = AsyncMock()
        result = await service.create_message(data, TENANT)
        assert result is not None

    async def test_delete_related_attachments_no_messages(self, service, mock_session):
        # No messages found
        mock_session.exec.return_value = make_mock_result(all_values=[])
        await service.delete_related_attachments("t1", TENANT)
        mock_session.delete.assert_not_called()

    async def test_delete_related_attachments_with_files(self, service, mock_session):
        attachment = MagicMock()
        # First exec: get_message_ids_by_thread returns IDs
        # Second exec: find attachment files
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(all_values=["m1"])
            else:
                return make_mock_result(all_values=[attachment])

        mock_session.exec = AsyncMock(side_effect=side_effect)
        await service.delete_related_attachments("t1", TENANT)
        mock_session.delete.assert_called_once_with(attachment)
