import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import json
import pytest
from tools.think.think_and_planning_tool import (
    ThoughtRecord,
    record_thought,
    get_think_function,
    aget_simple_think_tool,
)


class TestThoughtRecordPlanning:
    def test_create_record(self):
        record = ThoughtRecord(
            thought="analyze",
            thought_number=1,
            action="search",
            plan=["step1", "step2"],
        )
        assert record.thought == "analyze"
        assert record.thought_number == 1
        assert record.action == "search"
        assert record.plan == ["step1", "step2"]


class TestRecordThoughtPlanning:
    def test_records_to_cache(self):
        cache = []
        result_str = record_thought(
            think_cache=cache,
            thought="my thought",
            thought_number=1,
            action="do something",
            plan=["step1"],
        )
        result = json.loads(result_str)
        assert result["status"] == "success"
        assert len(cache) == 1
        assert cache[0].thought == "my thought"

    def test_default_values(self):
        cache = []
        result_str = record_thought(think_cache=cache)
        result = json.loads(result_str)
        assert result["status"] == "success"
        assert cache[0].thought == ""
        assert cache[0].thought_number == 1
        assert cache[0].action == ""
        assert cache[0].plan == []

    def test_multiple_records(self):
        cache = []
        record_thought(think_cache=cache, thought="t1", thought_number=1)
        record_thought(think_cache=cache, thought="t2", thought_number=2)
        assert len(cache) == 2


class TestGetThinkFunctionPlanning:
    def test_returns_callable(self):
        fn = get_think_function([])
        assert callable(fn)

    def test_function_records_to_provided_cache(self):
        cache = []
        fn = get_think_function(cache)
        fn(thought="hello", thought_number=1)
        assert len(cache) == 1


class TestAgetSimpleThinkToolPlanning:
    async def test_returns_tool(self):
        tool = await aget_simple_think_tool([])
        assert tool is not None
        assert tool.metadata.name == "think-and-planning"

    async def test_tool_description(self):
        tool = await aget_simple_think_tool([])
        assert "思考" in tool.metadata.description or "think" in tool.metadata.description.lower()
