from typing import List, Tuple

try:
    # Optional message types used only by the legacy message-context helpers.
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
except ImportError:  # pragma: no cover - exercised only in lean envs
    ChatMessage = None
    MessageRole = None

CJK_CHARS_PER_TOKEN = 1.5
OTHER_CHARS_PER_TOKEN = 2.5


def estimate_tokens_in_text(text: str) -> int:
    """Conservatively estimate tokens from Latin/CJK character counts."""
    if not text:
        return 0
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    cjk = min((n_bytes - n_chars) / 2, n_chars)
    other = n_chars - cjk
    return int(cjk / CJK_CHARS_PER_TOKEN + other / OTHER_CHARS_PER_TOKEN)

def truncate(
    text: str,
    max_token: int,
    start_token: int = 0,
) -> Tuple[str, int]:
    if not text:
        return text, 0
    if max_token <= start_token:
        return "", 0
    assert start_token >= 0, "start_token must be >= 0"

    total = estimate_tokens_in_text(text)
    if total <= start_token:
        return "", 0
    start_char = int(len(text) * start_token / max(total, 1))
    end_char = int(len(text) * min(max_token, total) / max(total, 1))
    result = text[start_char:end_char]
    return result, estimate_tokens_in_text(result)


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


def estimate_tokens_in_message(message: ChatMessage) -> int:
    """
    Estimate tokens length for a single message.

    Args:
        message (OpenAIMessage): The message to estimate the tokens length for.

    Returns:
        int: The estimated tokens length.

    """
    tokens = 0

    if message.role:
        tokens += estimate_tokens_in_text(str(message.role))

    text = get_message_context(message)
    tokens += estimate_tokens_in_text(text)

    additional_kwargs = {**message.additional_kwargs}

    if "tool_calls" in additional_kwargs:
        for tool_call in additional_kwargs["tool_calls"]:
            tokens += estimate_tokens_in_text(str(tool_call))

    return tokens


def get_last_n_msgs_skip_first(msgs: List[ChatMessage], n) -> List[ChatMessage]:
    last_n_messages = msgs[1:][-n:]
    if last_n_messages and last_n_messages[0].role == MessageRole.TOOL:
        return last_n_messages[1:]
    return last_n_messages
