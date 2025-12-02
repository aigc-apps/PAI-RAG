from typing import List, Optional, Tuple
from common.chat.constants import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_HISTORY_MESSAGES_COUNT,
    DEFAULT_HISTORY_MESSAGES_INPUT_TOKENS,
)
from memory.utils import (
    truncate,
    get_message_context,
    estimate_tokens_in_message,
    get_last_n_msgs_skip_first,
    get_tokenizer,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, BaseModel
from collections import deque
from loguru import logger
import copy
from llama_index.core.base.llms.types import ImageBlock


class QueueMessageItem(BaseModel):
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
        self.queue = deque()
        self.queue_tokens_num = 0
        self.history_messages = []
        self.history_tokens_num = 0
        self.tokenizer = get_tokenizer()

    def count_tokens(self, msg: ChatMessage) -> int:
        return estimate_tokens_in_message(msg, tokenizer=self.tokenizer)

    def from_messages(self, msgs: List[ChatMessage]) -> List[ChatMessage]:
        if not msgs:
            return []
        self.messages.extend(msgs)
        if msgs[0].role != MessageRole.SYSTEM:
            raise Exception("The system message must be the first message.")
        else:
            self.history_messages.append(msgs[0])
            self.history_tokens_num += self.count_tokens(msgs[0])
        last_n_history_messages = get_last_n_msgs_skip_first(
            msgs, DEFAULT_HISTORY_MESSAGES_COUNT
        )
        for message in last_n_history_messages:
            new_msg, tokens_num = self.truncate_message(
                message, max_tokens=DEFAULT_HISTORY_MESSAGES_INPUT_TOKENS
            )
            self.history_messages.append(new_msg)
            self.history_tokens_num += tokens_num

        if self.history_tokens_num > self.max_tokens:
            raise Exception(
                f"The input messages exceed the maximum context length ({self.max_tokens} tokens)"
            )

        self.max_tokens = self.max_tokens - self.history_tokens_num

    def add(self, msg: ChatMessage):
        self.messages.append(msg)

        tokens_num = self.count_tokens(msg)
        available_tokens = self.max_tokens - self.queue_tokens_num
        # 按照msg token截断message
        # 如果是tool,不能把preceeding message with "tool_calls" pop出queue,必须成对出现
        max_message_tokens = self.max_tokens
        if (
            msg.role == MessageRole.TOOL
            and max_message_tokens - self.queue[-1].tokens_num > 0
        ):
            max_message_tokens = max_message_tokens - self.queue[-1].tokens_num
        if tokens_num > max_message_tokens:
            msg, tokens_num = self.truncate_message(msg, max_tokens=max_message_tokens)
        # tokens余额足够,直接进入queue
        if tokens_num <= available_tokens:
            queue_message = QueueMessageItem(message=msg, tokens_num=tokens_num)
            self.queue.append(queue_message)
            self.queue_tokens_num += tokens_num
        # tokens余额不足,需要FIFO truncate和pop
        else:
            available_tokens = self.max_tokens
            queue_message = QueueMessageItem(message=msg, tokens_num=tokens_num)
            tokens_to_pop = self.queue_tokens_num + tokens_num - self.max_tokens
            self.pop(tokens_to_pop)
            self.queue.append(queue_message)
            self.queue_tokens_num += tokens_num

    def pop(self, tokens_to_pop):
        while tokens_to_pop > 0 and self.queue:
            first_msg_info = self.queue[0]
            # 直接pop出第一条消息
            if first_msg_info.tokens_num <= tokens_to_pop:
                self.queue_tokens_num -= first_msg_info.tokens_num
                tokens_to_pop -= first_msg_info.tokens_num
                logger.info(f"Pop out the first message {self.queue[0]}")
                self.queue.popleft()
                # 如果是tool,不能把preceeding message with "tool_calls" pop出queue,必须成对出现
                if self.queue and self.queue[0].message.role == MessageRole.TOOL:
                    self.queue_tokens_num -= self.queue[0].tokens_num
                    tokens_to_pop -= self.queue[0].tokens_num
                    logger.info(f"Pop out the first message {self.queue[0]}")
                    self.queue.popleft()
            # 截断第一条消息
            else:
                logger.info(f"truncate first message {self.queue[0]}")
                new_first_msg, new_first_tokens_num = self.truncate_message(
                    first_msg_info.message,
                    max_tokens=first_msg_info.tokens_num - tokens_to_pop,
                )
                self.queue[0].tokens_num = new_first_tokens_num
                self.queue[0].message = new_first_msg
                self.queue_tokens_num -= new_first_tokens_num
                break

    def get_context(self, image_urls : List[str] = None) -> List[ChatMessage]:
        return self.get_truncated_messages(image_urls)

    def get_truncated_messages(self, image_urls : List[str] = None) -> List[ChatMessage]:
        messages = []
        messages.extend(self.history_messages)
        for item in self.queue:
            messages.append(item.message)

        if len([m for m in messages if m.role == MessageRole.SYSTEM]) != 1:
            raise Exception("The input messages must contain only one system message. ")

        if image_urls:
            for message in reversed(messages):
                if message.role == MessageRole.USER:
                    for url in image_urls:
                        message.blocks.append(ImageBlock(url=url))
                    break
        return messages

    def get(self) -> List[ChatMessage]:
        return self.messages

    def truncate_message(
        self, msg: ChatMessage, max_tokens: int
    ) -> Tuple[ChatMessage, int]:
        # 仅处理包含 tool_calls 的消息
        if "tool_calls" in msg.additional_kwargs:
            tool_calls = copy.deepcopy(msg.additional_kwargs["tool_calls"])
            # 截断 arguments 字段
            for call in tool_calls:
                if call.function.arguments:
                    args = str(call.function.arguments)
                    # 截断 arguments 字符串
                    truncated_args, estimated_arg_tokens = truncate(
                        args, max_tokens, tokenizer=self.tokenizer
                    )
                    call.function.arguments = truncated_args
                    max_tokens -= estimated_arg_tokens
                    if max_tokens <= 0:
                        break
            new_msg = ChatMessage(
                role=msg.role,
                content=msg.content,
                additional_kwargs={"tool_calls": tool_calls},
            )
            new_tokens_num = self.count_tokens(new_msg)
            return new_msg, new_tokens_num

        else:
            # 普通消息按 content 截断
            text = get_message_context(msg)
            content, token = truncate(
                text, max_token=max_tokens, tokenizer=self.tokenizer
            )
            return ChatMessage(role=msg.role, content=content), token
