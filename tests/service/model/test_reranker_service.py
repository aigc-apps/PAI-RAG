import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.model.reranker_service import RerankerService
from db.models.knowledgebase.reranker import RerankerModelEntity, RerankerModelCreate
from tests.service.conftest import make_mock_result


TENANT = "test_tenant"


class TestRerankerService:
    @pytest.fixture
    def service(self, mock_session):
        return RerankerService(session=mock_session)

    async def test_get_reranker_found(self, service, mock_session):
        rk = RerankerModelEntity(id="r1", model_id="qwen3-rerank", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=rk)
        result = await service.get_reranker("r1", TENANT)
        assert result == rk

    async def test_get_reranker_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_reranker("nonexistent", TENANT)
        assert result is None

    async def test_get_reranker_by_model_id(self, service, mock_session):
        rk = RerankerModelEntity(id="r1", model_id="qwen3-rerank", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=rk)
        result = await service.get_reranker_by_model_id("qwen3-rerank", TENANT)
        assert result.model_id == "qwen3-rerank"

    async def test_list_rerankers(self, service, mock_session):
        rks = [RerankerModelEntity(id=f"r{i}", model_id=f"m{i}", tenant_id=TENANT) for i in range(2)]
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=2)
            else:
                return make_mock_result(all_values=rks)
        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_rerankers(TENANT)
        assert result.total == 2

    @patch("service.model.reranker_service.encrypt_key", return_value="encrypted")
    async def test_create_reranker(self, mock_encrypt, service, mock_session):
        data = RerankerModelCreate(
            model_id="qwen3-rerank",
            model_name="qwen3-rerank",
            base_url="https://example.com",
            provider_name="dashscope",
        )
        mock_session.refresh = AsyncMock()
        result = await service.create_reranker(data, TENANT)
        assert result is not None
        mock_session.add.assert_called_once()

    async def test_delete_reranker_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        with pytest.raises(ValueError):
            await service.delete_reranker("bad_id", TENANT)

    async def test_delete_reranker_success(self, service, mock_session):
        rk = RerankerModelEntity(id="r1", model_id="m1", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=rk)
        await service.delete_reranker("r1", TENANT)
        mock_session.delete.assert_called_once_with(rk)
