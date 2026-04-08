import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from service.injection import get_vectordb_service
from app.main import app


class TestVectordbAPI:
    @pytest.fixture(autouse=True)
    def setup_vectordb_service(self, api_client):
        self.mock_service = AsyncMock()
        app.dependency_overrides[get_vectordb_service] = lambda: self.mock_service
        self.client = api_client

    @patch("api.v1.config_apis.vectordb._cleanup_cached_vector_stores", new_callable=AsyncMock)
    def test_add_vector_db_config_success(self, mock_cleanup, mock_session):
        mock_entity = MagicMock()
        mock_entity.model_dump = MagicMock(return_value={
            "id": "vdb1",
            "type": "elasticsearch",
            "config": {"type": "elasticsearch", "host": "localhost"},
        })
        mock_entity.__class__ = type("VectorDbConfig", (), {})
        self.mock_service.create_or_update_vectordb_config.return_value = mock_entity
        mock_session.refresh = AsyncMock()

        response = self.client.post("/v1/config/vectordb", json={
            "type": "elasticsearch",
            "config": {"host": "localhost"},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    @patch("api.v1.config_apis.vectordb._cleanup_cached_vector_stores", new_callable=AsyncMock)
    def test_add_vector_db_config_value_error(self, mock_cleanup, mock_session):
        self.mock_service.create_or_update_vectordb_config.side_effect = ValueError("invalid config")
        response = self.client.post("/v1/config/vectordb", json={
            "type": "elasticsearch",
            "config": {},
        })
        assert response.status_code == 400

    @patch("api.v1.config_apis.vectordb._cleanup_cached_vector_stores", new_callable=AsyncMock)
    def test_add_vector_db_config_error(self, mock_cleanup, mock_session):
        self.mock_service.create_or_update_vectordb_config.side_effect = Exception("db error")
        response = self.client.post("/v1/config/vectordb", json={
            "type": "elasticsearch",
            "config": {},
        })
        assert response.status_code == 500

    def test_get_vector_config_success(self):
        mock_entity = MagicMock()
        mock_entity.model_dump = MagicMock(return_value={
            "id": "vdb1",
            "type": "local",
            "config": {},
        })
        mock_entity.__class__ = type("VectorDbConfig", (), {})
        self.mock_service.get_vectordb_config.return_value = mock_entity

        response = self.client.get("/v1/config/vectordb")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

    def test_get_vector_config_error(self):
        self.mock_service.get_vectordb_config.side_effect = Exception("db error")
        response = self.client.get("/v1/config/vectordb")
        assert response.status_code == 500
