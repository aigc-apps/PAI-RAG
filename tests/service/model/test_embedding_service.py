import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.model.embedding_service import EmbeddingService
from db.models.knowledgebase.embedding import EmbeddingModelEntity, EmbeddingModelCreate
from tests.service.conftest import make_mock_result


TENANT = "test_tenant"


class TestEmbeddingService:
    @pytest.fixture
    def service(self, mock_session):
        return EmbeddingService(session=mock_session)

    async def test_get_embedding_found(self, service, mock_session):
        emb = EmbeddingModelEntity(id="e1", model_id="text-embedding-v3", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=emb)
        result = await service.get_embedding("e1", TENANT)
        assert result == emb

    async def test_get_embedding_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        result = await service.get_embedding("nonexistent", TENANT)
        assert result is None

    async def test_get_embedding_by_model_id(self, service, mock_session):
        emb = EmbeddingModelEntity(id="e1", model_id="text-embedding-v3", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=emb)
        result = await service.get_embedding_by_model_id("text-embedding-v3", TENANT)
        assert result.model_id == "text-embedding-v3"

    async def test_list_embeddings(self, service, mock_session):
        embs = [EmbeddingModelEntity(id=f"e{i}", model_id=f"m{i}", tenant_id=TENANT) for i in range(2)]
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_mock_result(one_or_none_value=2)
            else:
                return make_mock_result(all_values=embs)
        mock_session.exec = AsyncMock(side_effect=side_effect)
        result = await service.list_embeddings(TENANT)
        assert result.total == 2

    @patch("service.model.embedding_service.encrypt_key", return_value="encrypted")
    async def test_create_embedding(self, mock_encrypt, service, mock_session):
        data = EmbeddingModelCreate(
            model_id="text-embedding-v3",
            model_name="text-embedding-v3",
            type="openai_like",
            endpoint="https://example.com",
            provider_name="openai_like",
        )
        mock_session.refresh = AsyncMock()
        result = await service.create_embedding(data, TENANT)
        assert result is not None
        mock_session.add.assert_called_once()

    async def test_delete_embedding_not_found(self, service, mock_session):
        mock_session.exec.return_value = make_mock_result(first_value=None)
        with pytest.raises(ValueError):
            await service.delete_embedding("bad_id", TENANT)

    async def test_delete_embedding_success(self, service, mock_session):
        emb = EmbeddingModelEntity(id="e1", model_id="m1", tenant_id=TENANT)
        mock_session.exec.return_value = make_mock_result(first_value=emb)
        await service.delete_embedding("e1", TENANT)
        mock_session.delete.assert_called_once_with(emb)
