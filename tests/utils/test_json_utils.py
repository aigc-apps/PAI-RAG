import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from utils.json_utils import parse_tool_arguments


class TestParseToolArguments:
    def test_empty_string_returns_empty_dict(self):
        assert parse_tool_arguments("") == {}

    def test_valid_json(self):
        result = parse_tool_arguments('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_partial_json_parsed(self):
        result = parse_tool_arguments('{"key": "val')
        assert isinstance(result, dict)

    def test_json_with_surrounding_noise(self):
        result = parse_tool_arguments('prefix {"key": "value"} suffix')
        assert result == {"key": "value"}

    def test_completely_invalid_string_returns_empty_dict(self):
        assert parse_tool_arguments("not json at all") == {}

    def test_none_returns_empty_dict(self):
        assert parse_tool_arguments(None) == {}
