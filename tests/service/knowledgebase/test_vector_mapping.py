import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.knowledgebase.vector_table_mapping_service import VectorTableMappingService
from db.models.knowledgebase.vector_table_mapping import VectorTableMappingEntity
from tests.service.conftest import make_mock_result

TENANT = "test_tenant"


@pytest.fixture(autouse=True)
def mock_cache():
    with patch("service.knowledgebase.vector_table_mapping_service.cache_manager") as mock_cm:
        mock_cache_instance = AsyncMock()
        mock_cache_instance.get.return_value = None
        mock_cache_instance.set.return_value = None
        mock_cm.get_cache.return_value = mock_cache_instance
        yield mock_cm


class TestVectorTableMappingService:
    @pytest.fixture
    def service(self, mock_session):
        return VectorTableMappingService(session=mock_session)

    async def test_get_existing_mapping(self, service, mock_session):
        mapping = MagicMock()
        mapping.table_name = "pai_rag_kb1_chunks"
        mock_session.exec.return_value = make_mock_result(first_value=mapping)
        result = await service.get_vector_table_name(TENANT, "kb1")
        assert result == "pai_rag_kb1_chunks"

    @patch("service.knowledgebase.vector_table_mapping_service.generate_vector_table_name", return_value="generated_name")
    async def test_create_new_mapping(self, mock_gen, service, mock_session):
        # No existing mapping
        mock_session.exec.return_value = make_mock_result(first_value=None)
        mock_session.refresh = AsyncMock()

        # Mock _create_mapping to return a mapping
        mapping = MagicMock()
        mapping.table_name = "generated_name"
        service._create_mapping = AsyncMock(return_value=mapping)

        result = await service.get_vector_table_name(TENANT, "kb1")
        assert result == "generated_name"

    async def test_cache_hit(self, service, mock_session, mock_cache):
        mock_cache_instance = mock_cache.get_cache.return_value
        mock_cache_instance.get.return_value = "cached_table_name"
        result = await service.get_vector_table_name(TENANT, "kb1")
        assert result == "cached_table_name"
        mock_session.exec.assert_not_called()
