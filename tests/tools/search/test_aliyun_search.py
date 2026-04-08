import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tools.search.aliyun_search import AliyunSearchTool, NodeWithScore


class TestNodeWithScore:
    def test_to_dict(self):
        node = NodeWithScore(text="hello", score=0.9, metadata={"key": "val"})
        d = node.to_dict()
        assert d["text"] == "hello"
        assert d["score"] == 0.9
        assert d["metadata"] == {"key": "val"}


class TestAliyunSearchTool:
    @patch("tools.search.aliyun_search.Client")
    @patch("tools.search.aliyun_search.open_api_models.Config")
    def test_init(self, mock_config_cls, mock_client_cls):
        tool = AliyunSearchTool(
            access_key_id="ak",
            access_key_secret="sk",
        )
        assert tool.search_count == 10
        mock_client_cls.assert_called_once()

    @patch("tools.search.aliyun_search.Client")
    @patch("tools.search.aliyun_search.open_api_models.Config")
    async def test_search_single_page_success(self, mock_config_cls, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body.to_map.return_value = {
            "pageItems": [
                {
                    "markdownText": "content here",
                    "link": "http://example.com",
                    "title": "Test Title",
                    "hostname": "example.com",
                    "score": 0.95,
                    "hostLogo": "http://logo.png",
                    "publishTime": "2024-01-01",
                },
            ]
        }
        mock_client.generic_search_async = AsyncMock(return_value=mock_response)

        tool = AliyunSearchTool(access_key_id="ak", access_key_secret="sk", search_count=5)
        result = await tool.aquery("test query")
        assert "result" in result
        assert len(result["result"]) == 1

    @patch("tools.search.aliyun_search.Client")
    @patch("tools.search.aliyun_search.open_api_models.Config")
    async def test_search_single_page_failed_status(self, mock_config_cls, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.generic_search_async = AsyncMock(return_value=mock_response)

        tool = AliyunSearchTool(access_key_id="ak", access_key_secret="sk")
        results = await tool._search_aliyun_single_page("test")
        assert results == []

    @patch("tools.search.aliyun_search.Client")
    @patch("tools.search.aliyun_search.open_api_models.Config")
    async def test_search_skips_empty_text(self, mock_config_cls, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body.to_map.return_value = {
            "pageItems": [
                {"markdownText": None, "mainText": None, "htmlSnippet": None, "link": "http://x.com", "title": "T"},
                {"markdownText": "valid", "link": "http://y.com", "title": "T2"},
            ]
        }
        mock_client.generic_search_async = AsyncMock(return_value=mock_response)

        tool = AliyunSearchTool(access_key_id="ak", access_key_secret="sk", search_count=5)
        result = await tool.aquery("test")
        assert len(result["result"]) == 1

    @patch("tools.search.aliyun_search.Client")
    @patch("tools.search.aliyun_search.open_api_models.Config")
    async def test_search_truncates_content(self, mock_config_cls, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body.to_map.return_value = {
            "pageItems": [
                {"markdownText": "x" * 2000, "link": "http://x.com", "title": "T"},
            ]
        }
        mock_client.generic_search_async = AsyncMock(return_value=mock_response)

        tool = AliyunSearchTool(access_key_id="ak", access_key_secret="sk", search_count=5)
        results = await tool._asearch("test")
        assert len(results[0].content) == 1000
