import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

from service.agent.agent_service import append_text


class TestAppendText:
    def test_string_content_append(self):
        msg = {"content": "Hello"}
        append_text(msg, " World")
        assert msg["content"] == "Hello World"

    def test_list_content_append_to_text_block(self):
        msg = {"content": [
            {"type": "image_url", "image_url": {"url": "http://img.png"}},
            {"type": "text", "text": "Hello"},
        ]}
        append_text(msg, " World")
        assert msg["content"][1]["text"] == "Hello World"

    def test_list_content_finds_first_text_block(self):
        msg = {"content": [
            {"type": "text", "text": "First"},
            {"type": "text", "text": "Second"},
        ]}
        append_text(msg, " appended")
        # Should append to the first text block found
        assert msg["content"][0]["text"] == "First appended"

    def test_missing_content_raises(self):
        msg = {"role": "user"}
        with pytest.raises(AssertionError):
            append_text(msg, " text")

    def test_empty_string_content(self):
        msg = {"content": ""}
        append_text(msg, "added")
        assert msg["content"] == "added"
