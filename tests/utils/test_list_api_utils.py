import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from utils.list_api_utils import parse_comma_separated_list
import pytest


class TestParseCommaSeparatedList:
    def test_none_returns_empty_list(self):
        assert parse_comma_separated_list(None) == []

    def test_empty_string_returns_empty_list(self):
        assert parse_comma_separated_list("") == []

    def test_single_value(self):
        assert parse_comma_separated_list("1") == ["1"]

    def test_multiple_values(self):
        assert parse_comma_separated_list("1,2,3") == ["1", "2", "3"]

    def test_values_with_spaces(self):
        assert parse_comma_separated_list(" 1 , 2 , 3 ") == ["1", "2", "3"]

    def test_trailing_comma_ignored(self):
        assert parse_comma_separated_list("1,2,") == ["1", "2"]

    def test_list_input_passthrough(self):
        assert parse_comma_separated_list(["a", "b"]) == ["a", "b"]

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            parse_comma_separated_list(123)
