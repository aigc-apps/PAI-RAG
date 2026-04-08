import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from db.models.knowledgebase.knowledgebase import KbEntity, KnowledgebaseCreate
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


# Mock cache_manager to avoid Redis dependency
@pytest.fixture(autouse=True)
def mock_cache():
    with patch("service.knowledgebase.knowledgebase_service.cache_manager") as mock_cm:
        mock_cache_instance = AsyncMock()
        mock_cache_instance.get.return_value = None
        mock_cache_instance.set.return_value = None
        mock_cache_instance.delete.return_value = None
        mock_cm.get_cache.return_value = mock_cache_instance
        yield mock_cm


class TestKnowledgebaseService:
    @pytest.fixture
    def service(self, mock_session):
        return KnowledgebaseService(session=mock_session)

    async def test_get_knowledgebase_found(self, service, mock_session):
        kb = KbEntity(id="kb1", name="test-kb", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=kb)
        result = await service.get_knowledgebase("kb1", TENANT)
        assert result == kb

    async def test_get_knowledgebase_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_knowledgebase("bad", TENANT)
        assert result is None

    async def test_get_knowledgebase_by_name(self, service, mock_session):
        kb = KbEntity(id="kb1", name="test-kb", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=kb)
        result = await service.get_knowledgebase_by_name("test-kb", TENANT)
        assert result.name == "test-kb"

    async def test_get_knowledgebases_by_ids(self, service, mock_session):
        kbs = [KbEntity(id=f"kb{i}", name=f"kb-{i}", tenant_id=TENANT) for i in range(2)]
        mock_session.exec.return_value = make_mock_result(all_values=kbs)
        result = await service.get_knowledgebases_by_ids(TENANT, ["kb0", "kb1"])
        assert len(result) == 2

    async def test_get_knowledgebases_by_ids_empty(self, service, mock_session):
        result = await service.get_knowledgebases_by_ids(TENANT, [])
        assert result == []

    async def test_list_knowledgebases(self, service, mock_session):
        # list_knowledgebases returns tuples of (KbEntity, file_count)
        # Use MagicMock to avoid model_dump serialization issues with lambda defaults
        mock_kbs = []
        for i in range(3):
            mock_kb = MagicMock()
            mock_kb.model_dump.return_value = {"id": f"kb{i}", "name": f"n{i}"}
            mock_kbs.append(mock_kb)
        kb_tuples = [(kb, 5) for kb in mock_kbs]
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=3)
            else:
                return make_mock_result(all_values=kb_tuples)
        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_knowledgebases(TENANT)
        assert result.total == 3

    async def test_create_knowledgebase(self, service, mock_session):
        data = KnowledgebaseCreate(name="new-kb", description="test", embedding_model="bge-m3")
        mock_session.refresh = AsyncMock()
        result = await service.create_knowledgebase(data, TENANT)
        assert result is not None
        mock_session.add.assert_called()

    async def test_delete_knowledgebase(self, service, mock_session):
        kb = KbEntity(id="kb1", name="test", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=kb)
        await service.delete_knowledgebase("kb1", TENANT)
        mock_session.delete.assert_called_once()

    async def test_delete_knowledgebase_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        with pytest.raises(ValueError):
            await service.delete_knowledgebase("bad", TENANT)
