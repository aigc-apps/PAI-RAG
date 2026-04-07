import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from pairag.file.utils.text_utils import replace_consecutive_spaces


class TestReplaceConsecutiveSpaces:
    def test_multiple_spaces_collapsed(self):
        assert replace_consecutive_spaces("a   b    c") == "a b c"

    def test_multiple_tabs_collapsed(self):
        assert replace_consecutive_spaces("a\t\t\tb") == "a\tb"

    def test_excessive_newlines_collapsed(self):
        result = replace_consecutive_spaces("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_empty_string(self):
        assert replace_consecutive_spaces("") == ""

    def test_none_returns_none(self):
        assert replace_consecutive_spaces(None) is None

    def test_single_spaces_unchanged(self):
        assert replace_consecutive_spaces("a b c") == "a b c"

    def test_mixed_whitespace(self):
        result = replace_consecutive_spaces("a   b\t\tc\n\n\n\nd")
        assert "   " not in result
        assert "\t\t" not in result
        assert "\n\n\n" not in result

    def test_two_newlines_preserved(self):
        # Two newlines should be preserved (paragraph break)
        result = replace_consecutive_spaces("a\n\nb")
        assert result == "a\n\nb"
