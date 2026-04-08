import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.knowledgebase.chunk_service import ChunkService
from db.models.knowledgebase.chunk import KbChunkEntity
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestChunkService:
    @pytest.fixture
    def service(self, mock_session):
        return ChunkService(session=mock_session)

    async def test_get_chunk_found(self, service, mock_session):
        chunk = KbChunkEntity(id="c1", kb_id="kb1", file_id="f1", tenant_id=TENANT, text="hello")
        mock_session.exec.return_value = make_mock_result(first_value=chunk)
        result = await service.get_chunk("c1", TENANT)
        assert result == chunk

    async def test_get_chunk_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_chunk("bad", TENANT)
        assert result is None

    @patch("service.knowledgebase.chunk_service.file_store")
    async def test_list_chunks(self, mock_file_store, service, mock_session):
        chunks = [KbChunkEntity(id=f"c{i}", kb_id="kb1", file_id="f1", tenant_id=TENANT, text=f"text{i}", chunk_metadata={}) for i in range(2)]
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=2)
            else:
                return make_mock_result(all_values=chunks)
        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_chunks("kb1", "f1", TENANT)
        assert result.total == 2

    @patch("service.knowledgebase.chunk_service.estimate_tokens_in_text", return_value=10)
    async def test_create_chunk(self, mock_tokens, service, mock_session):
        mock_session.refresh = AsyncMock()
        result = await service.create_chunk("kb1", "f1", "hello world", TENANT)
        assert result is not None
        mock_session.add.assert_called()
