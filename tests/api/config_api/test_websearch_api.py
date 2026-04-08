import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock
from service.injection import get_websearch_service
from app.main import app


class TestWebsearchAPI:
    @pytest.fixture(autouse=True)
    def setup_websearch_service(self, api_client):
        self.mock_service = AsyncMock()
        app.dependency_overrides[get_websearch_service] = lambda: self.mock_service
        self.client = api_client

    def test_add_search_config_success(self):
        mock_entity = MagicMock()
        mock_entity.type = "tavily"
        mock_entity.endpoint = "https://api.tavily.com"
        mock_entity.search_count = 10
        mock_entity.id = "ws1"
        mock_entity.encrypted_access_key_id = None
        mock_entity.encrypted_tavily_api_key = "enc_key"
        self.mock_service.create_or_update_websearch_config.return_value = mock_entity

        response = self.client.post("/v1/config/websearch", json={
            "type": "tavily",
            "tavily_api_key": "tvly-xxx",
            "search_count": 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["type"] == "tavily"
        assert data["data"]["is_tavily_empty"] is False
        assert data["data"]["is_aliyun_empty"] is True

    def test_add_search_config_invalid_type(self):
        response = self.client.post("/v1/config/websearch", json={
            "type": "google",
            "search_count": 10,
        })
        assert response.status_code == 400

    def test_add_search_config_aliyun(self):
        mock_entity = MagicMock()
        mock_entity.type = "aliyun"
        mock_entity.endpoint = "https://search.aliyun.com"
        mock_entity.search_count = 5
        mock_entity.id = "ws2"
        mock_entity.encrypted_access_key_id = "enc_ak"
        mock_entity.encrypted_tavily_api_key = None
        self.mock_service.create_or_update_websearch_config.return_value = mock_entity

        response = self.client.post("/v1/config/websearch", json={
            "type": "aliyun",
            "access_key_id": "ak",
            "access_key_secret": "sk",
            "search_count": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["is_aliyun_empty"] is False
        assert data["data"]["is_tavily_empty"] is True

    def test_add_search_config_error(self):
        self.mock_service.create_or_update_websearch_config.side_effect = Exception("db error")
        response = self.client.post("/v1/config/websearch", json={
            "type": "tavily",
            "tavily_api_key": "tvly-xxx",
        })
        assert response.status_code == 500

    def test_add_search_config_validation_error(self):
        self.mock_service.create_or_update_websearch_config.side_effect = ValueError("invalid config")
        response = self.client.post("/v1/config/websearch", json={
            "type": "tavily",
            "tavily_api_key": "tvly-xxx",
        })
        assert response.status_code == 400

    def test_list_search_config_success(self):
        mock_entity = MagicMock()
        mock_entity.type = "tavily"
        mock_entity.endpoint = "https://api.tavily.com"
        mock_entity.search_count = 10
        mock_entity.id = "ws1"
        mock_entity.encrypted_access_key_id = None
        mock_entity.encrypted_tavily_api_key = "enc_key"
        self.mock_service.get_all_websearch_configs.return_value = [mock_entity]

        response = self.client.get("/v1/config/websearch")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert len(data["data"]) == 1

    def test_list_search_config_empty_returns_default(self):
        self.mock_service.get_all_websearch_configs.return_value = []

        response = self.client.get("/v1/config/websearch")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["type"] == "aliyun"
        assert data["data"][0]["is_aliyun_empty"] is True
        assert data["data"][0]["is_tavily_empty"] is True

    def test_list_search_config_error(self):
        self.mock_service.get_all_websearch_configs.side_effect = Exception("db error")
        response = self.client.get("/v1/config/websearch")
        assert response.status_code == 500
