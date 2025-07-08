from llama_index.core.llms import ChatMessage, MessageRole
from typing import List, Tuple


def truncate(
    text: str,
    max_token: int,
    start_token: int = 0,
) -> Tuple[str, int]:
    if not text:
        return text, 0

    truncated_text = text[start_token:max_token]

    return truncated_text, len(truncated_text)


def get_message_context(msg: ChatMessage) -> str:
    if not msg.content:
        return ""
    if isinstance(msg.content, str):
        return msg.content
    else:
        text = []
        for item in msg.content:
            if not item.text:
                return None
            text.append(item.text)
        text = "\n".join(text)
        return text


def estimate_tokens_in_message(message: ChatMessage) -> str:
    """
    Estimate string length for a single message.

    Args:
        message (OpenAIMessage): The message to estimate the string length for.

    Returns:
        int: The estimated string length.

    """
    tokens = 0

    if message.role:
        tokens += len(message.role)

    text = get_message_context(message)
    tokens += len(text)

    additional_kwargs = {**message.additional_kwargs}

    if "tool_calls" in additional_kwargs:
        for tool_call in additional_kwargs["tool_calls"]:
            tokens += len(str(tool_call))

    return tokens


def get_last_n_msgs_skip_first(msgs: List[ChatMessage], n) -> List[ChatMessage]:
    last_n_messages = msgs[1:][-n:]
    if last_n_messages and last_n_messages[0].role == MessageRole.TOOL:
        return last_n_messages[1:]
    return last_n_messages
