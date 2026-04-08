import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import json
import pytest
from tools.think.simple_think_tool import (
    ThoughtRecord,
    record_thought,
    get_think_function,
    aget_simple_think_tool,
)


class TestThoughtRecord:
    def test_create_record(self):
        record = ThoughtRecord(timestamp="2024-01-01T00:00:00", thought="test thought")
        assert record.thought == "test thought"
        assert record.timestamp == "2024-01-01T00:00:00"


class TestRecordThought:
    def test_records_thought_to_cache(self):
        cache = {}
        result_str = record_thought(cache, "key1", "my thought")
        result = json.loads(result_str)
        assert result["type"] == "text"
        assert result["thoughts_count"] == 1
        assert len(cache["key1"]) == 1
        assert cache["key1"][0].thought == "my thought"

    def test_multiple_thoughts_same_key(self):
        cache = {}
        record_thought(cache, "key1", "thought 1")
        result_str = record_thought(cache, "key1", "thought 2")
        result = json.loads(result_str)
        assert result["thoughts_count"] == 2
        assert len(cache["key1"]) == 2

    def test_different_keys(self):
        cache = {}
        record_thought(cache, "key1", "thought A")
        record_thought(cache, "key2", "thought B")
        assert len(cache["key1"]) == 1
        assert len(cache["key2"]) == 1

    def test_long_thought_truncated_in_text(self):
        cache = {}
        long_thought = "x" * 100
        result_str = record_thought(cache, "key1", long_thought)
        result = json.loads(result_str)
        assert "..." in result["text"]


class TestGetThinkFunction:
    def test_returns_callable(self):
        fn = get_think_function("test_key")
        assert callable(fn)

    def test_function_records_thought(self):
        fn = get_think_function("test_key")
        result_str = fn(thought="hello")
        result = json.loads(result_str)
        assert result["thoughts_count"] == 1


class TestAgetSimpleThinkTool:
    async def test_returns_tools_and_mapping(self):
        openai_tools, tools_name_to_fn = await aget_simple_think_tool("test_cache")
        assert len(openai_tools) == 1
        assert "think" in tools_name_to_fn

    async def test_tool_is_callable(self):
        openai_tools, tools_name_to_fn = await aget_simple_think_tool("test_cache")
        think_tool = tools_name_to_fn["think"]
        assert think_tool is not None
