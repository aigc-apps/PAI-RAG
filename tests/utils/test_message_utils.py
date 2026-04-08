import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from utils.message_utils import to_chat_messages, convert_to_chat_messages, get_content_from_messages
from llama_index.core.base.llms.types import ChatMessage, MessageRole


class TestToChatMessages:
    def test_string_content(self):
        msg = {"role": "user", "content": "hello"}
        result = to_chat_messages(msg)
        assert isinstance(result, ChatMessage)
        assert str(result.content) == "hello"

    def test_text_block_content(self):
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "hello world"}],
        }
        result = to_chat_messages(msg)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_image_url_content_data_uri(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc", "detail": "high"}}
            ],
        }
        result = to_chat_messages(msg)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_image_url_content_http(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
            ],
        }
        result = to_chat_messages(msg)
        assert isinstance(result, list)

    def test_tool_call_content(self):
        msg = {
            "role": "user",
            "content": [{"type": "tool-call", "result": "tool output"}],
        }
        result = to_chat_messages(msg)
        assert isinstance(result, list)
        assert result[0].role == MessageRole.USER
        assert str(result[0].content) == "tool output"

    def test_missing_role_raises(self):
        import pytest
        with pytest.raises(AssertionError):
            to_chat_messages({"content": "hello"})

    def test_missing_content_raises(self):
        import pytest
        with pytest.raises(AssertionError):
            to_chat_messages({"role": "user"})


class TestConvertToChatMessages:
    def test_single_string_message(self):
        messages = [{"role": "user", "content": "hi"}]
        result = convert_to_chat_messages(messages)
        assert len(result) == 1

    def test_mixed_messages(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        result = convert_to_chat_messages(messages)
        assert len(result) == 2

    def test_empty_list(self):
        assert convert_to_chat_messages([]) == []


class TestGetContentFromMessages:
    def test_extracts_text(self):
        contents = [{"text": "hello"}, {"text": "world"}]
        assert get_content_from_messages(contents) == "hello\nworld"

    def test_skips_non_text_items(self):
        contents = [{"text": "hello"}, {"image": "data"}]
        assert get_content_from_messages(contents) == "hello"

    def test_empty_list(self):
        assert get_content_from_messages([]) == ""
