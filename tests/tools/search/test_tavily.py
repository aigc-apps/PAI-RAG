import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tools.search.tavily_search import TavilySearchTool, DEFAULT_MAX_SEARCH_RESULT, DEFAULT_MAX_SEARCH_CONTENT_LENGTH


class TestTavilySearchTool:
    @patch("tools.search.tavily_search.AsyncTavilyClient")
    def test_init_default_search_count(self, mock_client_cls):
        tool = TavilySearchTool(api_key="test-key")
        assert tool.search_count == DEFAULT_MAX_SEARCH_RESULT
        mock_client_cls.assert_called_once_with("test-key")

    @patch("tools.search.tavily_search.AsyncTavilyClient")
    def test_init_custom_search_count(self, mock_client_cls):
        tool = TavilySearchTool(api_key="test-key", search_count=5)
        assert tool.search_count == 5

    @patch("tools.search.tavily_search.AsyncTavilyClient")
    async def test_aquery_success(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {"title": "Result 1", "content": "short content", "url": "http://a.com"},
                {"title": "Result 2", "content": "short content 2", "url": "http://b.com"},
            ]
        }
        tool = TavilySearchTool(api_key="test-key")
        result = await tool.aquery("test query")
        assert "result" in result
        assert len(result["result"]) == 2
        mock_client.search.assert_called_once()

    @patch("tools.search.tavily_search.AsyncTavilyClient")
    async def test_aquery_truncates_long_content(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        long_content = "x" * 2000
        mock_client.search.return_value = {
            "results": [{"title": "R1", "content": long_content, "url": "u"}]
        }
        tool = TavilySearchTool(api_key="test-key")
        result = await tool.aquery("test")
        assert len(result["result"][0]["content"]) == DEFAULT_MAX_SEARCH_CONTENT_LENGTH

    @patch("tools.search.tavily_search.AsyncTavilyClient")
    async def test_aquery_truncates_long_query(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = {"results": []}
        tool = TavilySearchTool(api_key="test-key")
        long_query = "q" * 500
        await tool.aquery(long_query)
        call_args = mock_client.search.call_args
        assert len(call_args[0][0]) == 400

    @patch("tools.search.tavily_search.AsyncTavilyClient")
    async def test_aquery_handles_exception(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.side_effect = Exception("API error")
        tool = TavilySearchTool(api_key="test-key")
        result = await tool.aquery("test")
        assert "Error" in result["result"]
