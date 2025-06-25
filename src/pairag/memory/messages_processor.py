from typing import List, Optional
from pairag.mcp.constants import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_SUMMARIZE_LENGTH
from pairag.memory.utils import truncate
from llama_index.core.llms import ChatMessage, MessageRole
from pairag.mcp.prompts import DEFAULT_SUMMARIZE_PROMPT
from llama_index.core.utilities.token_counting import TokenCounter
from loguru import logger


class MessagesProcessor:
    """
    Memory is a class for managing the conversation memory
    It provides methods to truncate and compress the memory using summarization model.
    """

    def __init__(
        self,
        llm,
        max_tokens: Optional[int] = None,
        memory_summarize: Optional[bool] = False,
        summarize_length: Optional[int] = None,
        summarize_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.summarize_prompt = summarize_prompt or DEFAULT_SUMMARIZE_PROMPT
        self.memory_summarize = memory_summarize
        self.max_tokens = max_tokens or DEFAULT_MAX_INPUT_TOKENS
        self.summarize_length = summarize_length or DEFAULT_SUMMARIZE_LENGTH

    def count_tokens(self, msg: ChatMessage) -> int:
        return TokenCounter().estimate_tokens_in_messages([msg])

    def summarize(
        self, msg: ChatMessage, messages: List[ChatMessage], index: Optional[int] = None
    ) -> ChatMessage:
        """
        Summarize the text using the AI model.
        Args:
            msg (ChatMessage): The text to summarize
            messages (List[ChatMessage]): The list of messages
            index (int, optional): The index of the message in the messages.
        Returns:
            message (ChatMessage): The summarized message
        """
        if not self.llm:
            logger.warning("No model to perform summarization.")
            return msg
        if not self.memory_summarize:
            logger.info("not perform summarization.")
            return msg
        if self.count_tokens(msg) < self.summarize_length:
            logger.info("not need summarization.")
            return msg

        logger.info("perform summarization.")

        context_prompt_str = """
        # 下面是用户输入
        {query_str}
        # 下面是上下文
        {context_str}
        """
        if index:
            query_str = self.get_nearest_user_message(messages, index)
            context_str = self.get_nearest_assistant_message(messages, index)
        if query_str or context_str:
            prompt = self.summarize_prompt + context_prompt_str.format(
                query_str=query_str, context_str=context_str
            )
        else:
            prompt = self.summarize_prompt
        summary_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt),
            ChatMessage(role=MessageRole.USER, content=msg.content),
        ]

        response = self.llm.chat(summary_messages)
        return ChatMessage(role=msg.role, content=response.message.content)

    def get_nearest_user_message(self, messages, index):
        while index >= 0:
            if messages[index].role == MessageRole.USER and messages[index].content:
                return messages[index].content
            index -= 1
        return None

    def get_nearest_assistant_message(self, messages, index):
        while index > 0:
            if (
                messages[index - 1].role in (MessageRole.TOOL, MessageRole.ASSISTANT)
                and messages[index - 1].content
            ):
                return messages[index - 1].content
            index -= 1
        return None

    def compress_messages(self, messages) -> str:
        """
        Compress (truncate summarize) the memory using the model.
        """
        messages = self.truncate_messages(messages)
        if not self.llm:
            logger.info("No model to perform summarization.")
            return messages
        if not self.memory_summarize:
            logger.info("not perform summarization.")
            return messages
        new_messages = []
        for index, message in enumerate(messages):
            if message.role == MessageRole.SYSTEM:
                new_messages.append(message)
                continue
            if self.memory_summarize:
                new_messages.append(self.summarize(message, messages, index))
        return new_messages

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
