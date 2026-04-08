import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock
from service.knowledgebase.metadata_service import MetadataService
from db.models.knowledgebase.metadata import KbMetadataEntity
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


class TestMetadataService:
    @pytest.fixture
    def service(self, mock_session):
        return MetadataService(session=mock_session)

    async def test_get_metadata_found(self, service, mock_session):
        md = KbMetadataEntity(id="md1", kb_id="kb1", name="author", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=md)
        result = await service.get_metadata("kb1", "md1", TENANT)
        assert result == md

    async def test_get_metadata_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_metadata("kb1", "bad", TENANT)
        assert result is None

    async def test_get_metadata_by_name(self, service, mock_session):
        md = KbMetadataEntity(id="md1", kb_id="kb1", name="author", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=md)
        result = await service.get_metadata_by_name("kb1", "author", TENANT)
        assert result.name == "author"

    async def test_list_metadata(self, service, mock_session):
        mds = [KbMetadataEntity(id=f"md{i}", kb_id="kb1", name=f"key{i}", tenant_id=TENANT) for i in range(2)]
        md_tuples = [(md, 3) for md in mds]
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=2)
            else:
                return make_mock_result(all_values=md_tuples)
        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_metadata("kb1", TENANT)
        assert result.total == 2
