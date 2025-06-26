from typing import List, Optional, Union
from pairag.mcp.constants import DEFAULT_MAX_INPUT_TOKENS
from pairag.memory.utils import truncate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.utilities.token_counting import TokenCounter


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

    def count_tokens(self, msg: ChatMessage) -> int:
        return TokenCounter().estimate_tokens_in_messages([msg])

    def add(self, msg: Union[List[ChatMessage], ChatMessage]):
        new_messages = [msg] if isinstance(msg, ChatMessage) else msg
        self.messages.extend(new_messages)

    def get(self) -> List[ChatMessage]:
        return self.messages

    def get_truncated_messages(self) -> List[ChatMessage]:
        return self.truncate_messages(self.messages)

    def truncate_messages(
        self, messages: List[ChatMessage], max_tokens: Optional[int] = None
    ) -> List[ChatMessage]:
        max_tokens = max_tokens or self.max_tokens
        if len([m for m in messages if m.role == MessageRole.SYSTEM]) >= 2:
            raise Exception(
                code="400",
                message="The input messages must contain no more than one system message. "
                " And the system message, if exists, must be the first message.",
            )
        if messages and messages[0].role == MessageRole.SYSTEM:
            sys_msg = messages[0]
            available_token = max_tokens - self.count_tokens(sys_msg)
        else:
            sys_msg = None
            available_token = max_tokens
        token_cnt = 0
        new_messages = []
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == MessageRole.SYSTEM:
                continue
            cur_token_cnt = self.count_tokens(messages[i])
            if cur_token_cnt <= available_token:
                new_messages = [messages[i]] + new_messages
                available_token -= cur_token_cnt
            else:
                if (messages[i].role == MessageRole.USER) and (i != len(messages) - 1):
                    _msg = self.truncate_message(
                        messages[i], max_tokens=available_token
                    )
                    if _msg:
                        new_messages = [_msg] + new_messages
                    break
                elif messages[i].role in (MessageRole.TOOL, MessageRole.ASSISTANT):
                    _msg = self.truncate_message(
                        messages[i], max_tokens=available_token
                    )
                    if _msg:
                        new_messages = [_msg] + new_messages
                    else:
                        break
                else:
                    token_cnt = (max_tokens - available_token) + cur_token_cnt
                    break

        if sys_msg is not None:
            new_messages = [sys_msg] + new_messages

        if (sys_msg is not None and len(new_messages) < 2) or (
            sys_msg is None and len(new_messages) < 1
        ):
            raise Exception(
                code="400",
                message=f"The input messages exceed the maximum context length ({max_tokens} tokens) after "
                f"keeping only the system message (if exists) and the latest one user message (around {token_cnt} tokens). ",
            )
        return new_messages

    def truncate_message(self, msg: ChatMessage, max_tokens: int):
        if isinstance(msg.content, str):
            content = truncate(msg.content, max_token=max_tokens)
        else:
            text = []
            for item in msg.content:
                if not item.text:
                    return None
                text.append(item.text)
            text = "\n".join(text)
            content = truncate(text, max_token=max_tokens)
        return ChatMessage(role=msg.role, content=content)
