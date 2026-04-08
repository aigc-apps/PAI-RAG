import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from evaluation.utils import parse_function_call


class TestParseFunctionCall:
    def test_with_actions(self):
        json_line = {
            "actions": [{"function": "search", "arguments": {"q": "test"}}],
        }
        result = parse_function_call(json_line, "obs_text")
        assert result["function"] == "search"
        assert result["observation"] == "obs_text"

    def test_without_actions(self):
        json_line = {}
        result = parse_function_call(json_line, "obs")
        assert result == {"observation": "obs"}

    def test_none_json_line(self):
        result = parse_function_call(None, "obs")
        assert result == {"observation": "obs"}

    def test_empty_actions_list(self):
        json_line = {"actions": []}
        result = parse_function_call(json_line, "obs")
        assert result == {"observation": "obs"}

    def test_multiple_actions_takes_first(self):
        json_line = {
            "actions": [
                {"function": "first"},
                {"function": "second"},
            ]
        }
        result = parse_function_call(json_line, "obs")
        assert result["function"] == "first"
