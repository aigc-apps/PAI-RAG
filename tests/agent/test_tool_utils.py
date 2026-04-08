import json
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from agent.tool_utils import truncate_json_proportionally, smart_truncate_v2


class TestTruncateJsonProportionally:
    def test_no_long_strings_unchanged(self):
        data = {"a": "short", "b": 123, "c": [1, 2, 3]}
        original = json.dumps(data)
        result = truncate_json_proportionally(data, 100)
        assert json.dumps(result) == original

    def test_single_long_string_truncated(self):
        long_str = "x" * 500
        data = {"content": long_str}
        result = truncate_json_proportionally(data, 200)
        assert len(result["content"]) == 300  # 500 - 200

    def test_proportional_allocation(self):
        data = {"a": "x" * 600, "b": "y" * 400}
        result = truncate_json_proportionally(data, 100)
        # a gets ceil(100 * 600/1000) = 60 cut, b gets ceil(100 * 400/1000) = 40 cut
        assert len(result["a"]) == 540
        assert len(result["b"]) == 360

    def test_nested_structure_recursive(self):
        data = {"outer": {"inner": "z" * 500}}
        result = truncate_json_proportionally(data, 100)
        assert len(result["outer"]["inner"]) == 400

    def test_list_with_long_strings(self):
        data = ["a" * 500, "short", "b" * 500]
        result = truncate_json_proportionally(data, 200)
        assert len(result[0]) == 400  # 500 - ceil(200 * 500/1000)
        assert result[1] == "short"
        assert len(result[2]) == 400

    def test_empty_data(self):
        assert truncate_json_proportionally({}, 100) == {}
        assert truncate_json_proportionally([], 100) == []

    def test_no_candidates_below_threshold(self):
        data = {"a": "x" * 299}  # Below 300 threshold
        result = truncate_json_proportionally(data, 100)
        assert result["a"] == "x" * 299


class TestSmartTruncateV2:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert smart_truncate_v2(text, max_length=100) == text

    def test_empty_string(self):
        assert smart_truncate_v2("") == ""
        assert smart_truncate_v2("", max_length=10) == ""

    def test_none_input(self):
        assert smart_truncate_v2(None) is None

    def test_valid_json_uses_json_truncation(self):
        data = {"content": "x" * 20000}
        json_str = json.dumps(data)
        result = smart_truncate_v2(json_str, max_length=5000)
        assert len(result) <= len(json_str)
        # Should still be valid JSON
        parsed = json.loads(result)
        assert "content" in parsed

    def test_non_json_uses_plain_truncation(self):
        text = "a" * 20000
        result = smart_truncate_v2(text, max_length=5000)
        assert len(result) == 5000
        assert result == "a" * 5000

    def test_exactly_at_max_length(self):
        text = "x" * 15000
        assert smart_truncate_v2(text, max_length=15000) == text
