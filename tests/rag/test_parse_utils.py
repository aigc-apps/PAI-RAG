import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from rag.parse_utils import sanitize_text


class TestSanitizeText:
    def test_nul_byte_removed(self):
        assert "\x00" not in sanitize_text("hello\x00world")

    def test_control_characters_removed(self):
        text = "a\x01b\x02c\x03d"
        result = sanitize_text(text)
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x03" not in result

    def test_tab_and_newline_preserved(self):
        text = "hello\tworld\nfoo"
        result = sanitize_text(text)
        assert "\t" in result
        assert "\n" in result

    def test_unicode_replacement_char_removed(self):
        text = "hello\uFFFDworld"
        result = sanitize_text(text)
        assert "\uFFFD" not in result

    def test_zero_width_chars_removed(self):
        text = "hello\u200Bworld\u200C\u200D\uFEFF"
        result = sanitize_text(text)
        assert "\u200B" not in result
        assert "\u200C" not in result
        assert "\u200D" not in result
        assert "\uFEFF" not in result

    def test_newline_normalization(self):
        text = "line1\r\nline2\rline3"
        result = sanitize_text(text)
        assert "\r" not in result
        assert result == "line1\nline2\nline3"

    def test_normal_text_unchanged(self):
        text = "Hello, World! This is normal text."
        assert sanitize_text(text) == text

    def test_non_string_returns_empty(self):
        assert sanitize_text(123) == ""
        assert sanitize_text(None) == ""

    def test_empty_string(self):
        assert sanitize_text("") == ""
