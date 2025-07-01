from typing import List, Optional
from pairag.mcp.constants import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_HISTORY_MESSAGES_COUNT,
    DEFAULT_HISTORY_MESSAGES_INPUT_TOKENS,
)
from pairag.memory.utils import truncate, get_message_context
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.utilities.token_counting import TokenCounter
from llama_index.core.utils import get_tokenizer
from llama_index.core.bridge.pydantic import Field, BaseModel
from collections import deque
from loguru import logger
import copy


class MessageInfo(BaseModel):
    message: ChatMessage = Field(description="Original message.")
    tokens_num: int = Field(description="message token.", default=0)


class BaseMemory:
    """
    Memory is a class for managing the conversation memory
    It provides methods to truncate the memory.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
    ):
        self.max_tokens = max_tokens or DEFAULT_MAX_INPUT_TOKENS
        self.messages = []
        self.tokenizer = get_tokenizer()
        self.token_counter = TokenCounter(tokenizer=self.tokenizer)
        self.queue = deque()
        self.tokens_in_queue = 0
        self.history_messages = []
        self.history_token = 0

    def count_tokens(self, msg: ChatMessage) -> int:
        return self.token_counter.estimate_tokens_in_messages([msg])

    def from_messages(self, msgs: List[ChatMessage]) -> List[ChatMessage]:
        if not msgs:
            return []
        self.messages.extend(msgs)
        if len(msgs) <= DEFAULT_HISTORY_MESSAGES_COUNT:
            self.history_messages.extend(msgs)
        else:
            # 传入system messages
            self.history_messages.append(msgs[0])
            # 传入history messages
            self.history_messages.extend(msgs[-DEFAULT_HISTORY_MESSAGES_COUNT:])
        for i, message in enumerate(self.history_messages):
            new_msg = self.truncate_message(
                message, max_tokens=DEFAULT_HISTORY_MESSAGES_INPUT_TOKENS
            )
            self.history_messages[i] = new_msg
            tokens_num = self.count_tokens(new_msg)
            self.history_token += tokens_num

        if self.history_token > self.max_tokens:
            raise Exception(
                "The input messages exceed the maximum context length ({self.max_tokens} tokens)"
            )

        self.max_tokens = self.max_tokens - self.history_token

    def add(self, msg: ChatMessage):
        self.messages.append(msg)

        tokens_num = self.count_tokens(msg)
        available_tokens = self.max_tokens - self.tokens_in_queue
        # tokens余额足够,直接进入queue
        if tokens_num <= available_tokens:
            queue_message = MessageInfo(message=msg, tokens_num=tokens_num)
            self.queue.append(queue_message)
            self.tokens_in_queue += tokens_num
        # tokens余额不足,需要FIFO truncate和pop
        else:
            available_tokens = self.max_tokens
            # 如果是tool,不能把preceeding message with "tool_calls" pop出queue,必须成对出现
            if msg.role == MessageRole.TOOL:
                available_tokens = available_tokens - self.queue[-1].tokens_num
            new_msg = self.truncate_message(msg, max_tokens=available_tokens)
            new_tokens_num = self.count_tokens(new_msg)
            queue_message = MessageInfo(message=new_msg, tokens_num=new_tokens_num)
            tokens_to_pop = self.tokens_in_queue + new_tokens_num - self.max_tokens
            self.pop(tokens_to_pop)
            self.queue.append(queue_message)
            self.tokens_in_queue += new_tokens_num

    def pop(self, tokens_to_pop):
        while tokens_to_pop > 0 and self.queue:
            first_msg_info = self.queue[0]
            # 直接pop出第一条消息
            if first_msg_info.tokens_num <= tokens_to_pop:
                self.tokens_in_queue -= first_msg_info.tokens_num
                tokens_to_pop -= first_msg_info.tokens_num
                logger.info(f"Pop out the first message {self.queue[0]}")
                self.queue.popleft()
                # 如果是tool,不能把preceeding message with "tool_calls" pop出queue,必须成对出现
                if self.queue and self.queue[0].message.role == MessageRole.TOOL:
                    self.tokens_in_queue -= self.queue[0].tokens_num
                    tokens_to_pop -= self.queue[0].tokens_num
                    logger.info(f"Pop out the first message {self.queue[0]}")
                    self.queue.popleft()
            # 截断第一条消息
            else:
                logger.info(f"truncate first message {self.queue[0]}")
                new_first_msg = self.truncate_message(
                    first_msg_info.message,
                    max_tokens=self.queue[0].tokens_num - tokens_to_pop,
                )
                new_first_tokens_num = self.count_tokens(new_first_msg)
                self.queue[0].tokens_num = new_first_tokens_num
                self.queue[0].message = new_first_msg
                self.tokens_in_queue -= new_first_tokens_num
                break

    def get_context(self) -> List[ChatMessage]:
        return self.get_truncated_messages()

    def get_truncated_messages(self) -> List[ChatMessage]:
        messages = copy.deepcopy(self.history_messages)
        for item in self.queue:
            messages.append(item.message)

        if len([m for m in messages if m.role == MessageRole.SYSTEM]) != 1:
            raise Exception(
                "The input messages must contain only one system message. "
                " And the system message, if exists, must be the first message."
            )
        return messages

    def get(self) -> List[ChatMessage]:
        return self.messages

    def truncate_message(self, msg: ChatMessage, max_tokens: int):
        # 仅处理包含 tool_calls 的消息
        if "tool_calls" in msg.additional_kwargs:
            tool_calls = copy.deepcopy(msg.additional_kwargs["tool_calls"])
            # 截断 arguments 字段
            for call in tool_calls:
                if "arguments" in call["function"]:
                    args = str(call["function"]["arguments"])
                    # 截断 arguments 字符串
                    truncated_args = truncate(
                        args, max_tokens, tokenizer=self.tokenizer
                    )
                    call["function"]["arguments"] = truncated_args
                    estimated_arg_tokens = self.count_tokens(
                        ChatMessage(role="assistant", content=truncated_args)
                    )
                    max_tokens -= estimated_arg_tokens
                    if max_tokens <= 0:
                        break
            return ChatMessage(
                role=msg.role,
                content=msg.content,
                additional_kwargs={"tool_calls": tool_calls},
            )
        else:
            # 普通消息按 content 截断
            text = get_message_context(msg)
            content = truncate(text, max_token=max_tokens, tokenizer=self.tokenizer)
            return ChatMessage(role=msg.role, content=content)
