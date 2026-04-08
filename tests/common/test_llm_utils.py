import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
import json
from common.llm.utils import parse_llm_json, get_citation_source, extract_citations


class TestParseLlmJson:
    def test_valid_json(self):
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        result = parse_llm_json('Here is the result: {"answer": 42} done')
        assert result == {"answer": 42}

    def test_no_braces_returns_empty(self):
        result = parse_llm_json("no json here")
        assert result == {}

    def test_only_opening_brace_returns_empty(self):
        result = parse_llm_json("{no closing")
        assert result == {}

    def test_nested_json(self):
        data = {"outer": {"inner": "value"}}
        result = parse_llm_json(f"prefix {json.dumps(data)} suffix")
        assert result == data

    def test_empty_string(self):
        result = parse_llm_json("")
        assert result == {}

    def test_invalid_json_content_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("{invalid: json}")


class TestGetCitationSource:
    def test_knowledgebase_tool(self):
        assert get_citation_source("search-knowledgebase-abc") == "knowledgebase"

    def test_knowledgebase_exact(self):
        assert get_citation_source("search-knowledgebase") == "knowledgebase"

    def test_other_tool(self):
        assert get_citation_source("tavily-websearch") == "web"

    def test_empty_string(self):
        assert get_citation_source("") == "web"


class TestExtractCitations:
    def _make_tool_chunk(self, tool_name, result_data):
        """Helper to create a mock ToolResultChunk-like object."""
        from unittest.mock import MagicMock
        chunk = MagicMock()
        chunk.tool.function.name = tool_name
        chunk.result = json.dumps({"result": result_data}) if result_data is not None else None
        return chunk

    def test_websearch_citations(self):
        chunk = self._make_tool_chunk("tavily-websearch", [
            {"title": "Page1", "content": "text1", "url": "http://a.com", "score": 0.9},
            {"title": "Page2", "content": "text2", "url": "http://b.com", "score": 0.8},
        ])
        citations, details = extract_citations(chunk)
        assert citations == ["Page1", "Page2"]
        assert len(details) == 2
        assert details[0]["source"] == "web"

    def test_knowledgebase_citations(self):
        chunk = self._make_tool_chunk("search-knowledgebase-kb1", [
            {"title": "Doc1", "content": "c1", "url": "", "score": 0.95},
        ])
        citations, details = extract_citations(chunk)
        assert citations == ["Doc1"]
        assert details[0]["source"] == "knowledgebase"

    def test_deduplicates_by_title(self):
        chunk = self._make_tool_chunk("tavily-websearch", [
            {"title": "Same", "content": "a", "url": "u1"},
            {"title": "Same", "content": "b", "url": "u2"},
        ])
        citations, details = extract_citations(chunk)
        assert citations == ["Same"]
        assert len(details) == 1

    def test_none_result_returns_empty(self):
        chunk = self._make_tool_chunk("tavily-websearch", None)
        chunk.result = None
        citations, details = extract_citations(chunk)
        assert citations == []
        assert details == []

    def test_unrelated_tool_returns_empty(self):
        chunk = self._make_tool_chunk("some-other-tool", [{"title": "X"}])
        citations, details = extract_citations(chunk)
        assert citations == []
        assert details == []

    def test_aliyun_websearch(self):
        chunk = self._make_tool_chunk("aliyun-websearch", [
            {"title": "A", "content": "c", "url": "http://a.com"},
        ])
        citations, details = extract_citations(chunk)
        assert citations == ["A"]
        assert details[0]["source"] == "web"

    def test_empty_title_skipped(self):
        chunk = self._make_tool_chunk("tavily-websearch", [
            {"title": "", "content": "c"},
            {"title": "Valid", "content": "c"},
        ])
        citations, details = extract_citations(chunk)
        assert citations == ["Valid"]
