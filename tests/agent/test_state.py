import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from agent.state import get_message_content, AgentState


class TestGetMessageContent:
    def test_string_content(self):
        msg = {"content": "hello world"}
        assert get_message_content(msg) == "hello world"

    def test_list_content_text_block(self):
        msg = {"content": [{"type": "text", "text": "hello"}]}
        assert get_message_content(msg) == "hello"

    def test_list_content_multiple_text_blocks(self):
        msg = {"content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]}
        assert get_message_content(msg) == "hello world"

    def test_image_block_returns_empty(self):
        msg = {"content": [{"type": "image_url", "image_url": {"url": "http://img.png"}}]}
        assert get_message_content(msg) == ""

    def test_mixed_blocks(self):
        msg = {"content": [
            {"type": "image_url", "image_url": {"url": "http://img.png"}},
            {"type": "text", "text": "caption"},
        ]}
        assert get_message_content(msg) == "caption"

    def test_missing_content(self):
        msg = {}
        assert get_message_content(msg) == ""

    def test_none_content(self):
        msg = {"content": None}
        assert get_message_content(msg) is None


class TestAgentState:
    def test_from_messages_filters_system_and_orphaned_tool(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": "result"},  # orphaned: no preceding assistant with tool_calls
            {"role": "assistant", "content": "Hi there"},
        ]
        state = AgentState.from_messages(messages)
        roles = [m["role"] for m in state.messages]
        assert "system" not in roles
        assert "tool" not in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_from_messages_preserves_tool_with_preceding_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
            {"role": "tool", "content": "result", "tool_call_id": "t1"},
        ]
        state = AgentState.from_messages(messages)
        roles = [m["role"] for m in state.messages]
        assert roles == ["user", "assistant", "tool"]

    def test_from_messages_preserves_multiple_parallel_tool_results(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {"id": "t2", "type": "function", "function": {"name": "search", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "result1", "tool_call_id": "t1"},
            {"role": "tool", "content": "result2", "tool_call_id": "t2"},
        ]
        state = AgentState.from_messages(messages)
        roles = [m["role"] for m in state.messages]
        assert roles == ["user", "assistant", "tool", "tool"]

    def test_from_messages_filters_image_only_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "image_url", "image_url": {"url": "http://img.png"}},
            ]},
        ]
        state = AgentState.from_messages(messages)
        # Image-only assistant message should be filtered out (no text blocks)
        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "user"

    def test_from_messages_keeps_text_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Here is the answer"},
            ]},
        ]
        state = AgentState.from_messages(messages)
        assert len(state.messages) == 2

    def test_from_messages_string_assistant_content(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Reply"},
        ]
        state = AgentState.from_messages(messages)
        assert len(state.messages) == 2
        assert state.messages[1]["content"] == "Reply"

    def test_step_defaults_to_1(self):
        messages = [{"role": "user", "content": "Hello"}]
        state = AgentState.from_messages(messages)
        assert state.step == 1
