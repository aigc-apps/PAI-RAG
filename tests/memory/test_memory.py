import pytest
from unittest.mock import patch
from collections import deque
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from memory.base_memory import BaseMemory
from common.chat.constants import DEFAULT_MAX_INPUT_TOKENS
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from memory.utils import get_last_n_msgs_skip_first


# 测试数据
SYSTEM_MSG = ChatMessage(role=MessageRole.SYSTEM, content="system prompt")
USER_MSG = ChatMessage(role=MessageRole.USER, content="user input")
ASSISTANT_MSG = ChatMessage(role=MessageRole.ASSISTANT, content="assistant response")
arguments = "已经成功获取了上海虹桥站（station_code: AOH）和杭州西站（station_code: HVU）的车站编码。接下来，我需要使用这些信息加上当前日期作为参数，调用12306-mcp--get-tickets接口来查询两个车站之间的高铁班次情况。为了确保查询到的是今天的班次信息，我将先通过12306-mcp--get-current-date接口获取今天的具体日期。"
tool_calls = [
    ChoiceDeltaToolCall(
        index=1,
        id="1",
        type="function",
        function=ChoiceDeltaToolCallFunction(name="test", arguments=str(arguments)),
    )
]

# 工具调用消息
TOOL_CALL_MSG = ChatMessage(
    role=MessageRole.ASSISTANT,
    content="",
    additional_kwargs={
        "tool_calls": tool_calls,
    },
)

TOOL_RESPONSE_MSG = ChatMessage(
    role=MessageRole.TOOL, content='{"result": "tool_response"}', tool_call_id="1"
)

class TestBaseMemory:
    @pytest.fixture
    def memory(self):
        return BaseMemory(max_tokens=100)

    def test_initialization(self):
        """测试初始化逻辑"""
        memory = BaseMemory()
        assert memory.max_tokens == DEFAULT_MAX_INPUT_TOKENS
        assert isinstance(memory.messages, list)
        assert isinstance(memory.queue, deque)
        assert memory.queue_tokens_num == 0
        assert memory.history_tokens_num == 0

    def test_from_messages_basic(self, memory):
        """测试基础消息处理"""
        messages = [SYSTEM_MSG, USER_MSG, ASSISTANT_MSG]
        memory.from_messages(messages)

        assert len(memory.messages) == 3
        assert memory.messages == messages
        assert len(memory.history_messages) == 3

    def test_from_messages_history_limit(self, memory):
        """测试历史消息截取逻辑"""
        memory.max_tokens = 20000
        long_messages = [SYSTEM_MSG] + [USER_MSG] * 80  # 20条消息
        memory.from_messages(long_messages)
        assert len(memory.history_messages) == 51  # 系统消息 + 最后10条

    def test_from_messages_token_limit_exceeded(self, memory):
        """测试历史token超限时抛出异常"""
        with patch.object(memory, "count_tokens", return_value=10):
            with pytest.raises(Exception) as exc_info:
                memory.max_tokens = 2  # 设置较小的max_tokens
                memory.from_messages([SYSTEM_MSG, USER_MSG])
            assert "exceed the maximum context length" in str(exc_info.value)

    def test_add_normal_message(self, memory):
        """测试添加普通消息"""

        # 添加普通消息
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="test content")
        memory.add(msg)

        assert msg in memory.messages
        assert len(memory.queue) == 1
        assert memory.queue_tokens_num == 3

    def test_add_tool_message(self, memory):
        """测试添加工具消息"""
        memory.add(TOOL_CALL_MSG)

        # 添加工具响应消息
        memory.add(TOOL_RESPONSE_MSG)

        assert len(memory.queue) == 2

    def test_queue_truncation(self, memory):
        """测试队列pop逻辑"""
        memory.max_tokens = 10  # 设置较小的max_tokens

        # 添加第一条消息
        msg1 = ChatMessage(role=MessageRole.ASSISTANT, content="msg1")
        memory.add(msg1)

        # 添加第二条消息触发截断
        msg2 = ChatMessage(role=MessageRole.ASSISTANT, content="msg2" * 100)
        memory.add(msg2)

        assert len(memory.queue) == 1  # 第一条被pop

    def test_tool_call_pairing(self, memory):
        """测试工具调用/响应必须成对出现"""
        memory.max_tokens = 100

        # 添加工具调用消息
        memory.add(TOOL_CALL_MSG)

        # 添加工具响应消息
        memory.add(TOOL_RESPONSE_MSG)
        assert len(memory.queue) % 2 == 0  # 确保成对存在

    def test_get_truncated_messages(self, memory):
        """测试获取截断后的消息"""
        # 添加系统消息
        memory.history_messages = [SYSTEM_MSG]

        # 添加队列消息
        msg1 = ChatMessage(role=MessageRole.USER, content="msg1")
        msg2 = ChatMessage(role=MessageRole.ASSISTANT, content="msg2")
        memory.add(msg1)
        memory.add(msg2)

        result = memory.get_truncated_messages()
        assert len(result) == 3  # 系统消息 + 2条队列消息
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].content == "msg1"

    def test_get_truncated_messages_invalid_system(self, memory):
        """测试系统消息校验"""
        # 添加两个系统消息
        memory.history_messages = [SYSTEM_MSG, SYSTEM_MSG]

        with pytest.raises(Exception) as exc_info:
            memory.get_truncated_messages()
        assert "must contain only one system message" in str(exc_info.value)

    def test_truncate_tool_call(self, memory):
        """测试工具调用参数截断"""
        truncated_msg, token_num = memory.truncate_message(TOOL_CALL_MSG, 5)

        assert (
            truncated_msg.additional_kwargs["tool_calls"][0].function.arguments
            != TOOL_CALL_MSG.additional_kwargs["tool_calls"][0].function.arguments
        )
        assert truncated_msg.role == MessageRole.ASSISTANT

    def test_truncate_normal_message(self, memory):
        """测试普通消息截断"""
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="original long content")
        truncated_msg, token_num = memory.truncate_message(msg, 2)
        assert truncated_msg.content != msg.content

    def test_get_last_n_msgs_skip_first(self):
        """测试取跳过第一条消息"""

        msg_list = [ASSISTANT_MSG] * 5

        new_msg_list = get_last_n_msgs_skip_first(msg_list, 7)

        assert new_msg_list == [ASSISTANT_MSG] * 4
