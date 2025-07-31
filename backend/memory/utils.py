from llama_index.core.llms import ChatMessage, MessageRole
from typing import List, Tuple, Any
from transformers import AutoTokenizer
from utils.constants import DEFAULT_MODEL_DIR
import os

TOKENIZATION_MODEL = "Qwen3-32B-Tokenization"


def get_tokenizer():
    model_path = os.path.join(DEFAULT_MODEL_DIR, TOKENIZATION_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return tokenizer


def estimate_tokens_in_text(
    text: str,
    tokenizer: Any = None,
) -> int:
    """
    Estimate token length for a given text.

    Args:
        text (str): The text to estimate the tokens length for.

    Returns:
        int: The estimated tokens length.

    """
    if not text:
        return 0
    tokenizer = tokenizer or get_tokenizer()
    result = tokenizer(text, return_offsets_mapping=True)
    token_ids = result["input_ids"]
    return len(token_ids)


def truncate(
    text: str,
    max_token: int,
    start_token: int = 0,
    tokenizer: Any = None,
) -> Tuple[str, int]:
    if not text:
        return text, 0
    if max_token <= start_token:
        return "", 0
    assert start_token >= 0, "start_token must be >= 0"

    tokenizer = tokenizer or get_tokenizer()
    result = tokenizer(text, return_offsets_mapping=True)
    token_ids, offset_mapping = result["input_ids"], result["offset_mapping"]
    start_token_offset_mapping_left = offset_mapping[start_token][0]
    if max_token > len(token_ids):
        text = text[start_token_offset_mapping_left:]
        return text, len(token_ids[start_token:])
    else:
        # start_token大于等于0, 此时max_token大于start_token,大于等于1
        last_token_offset_mapping_right = offset_mapping[max_token - 1][1]
        text = text[start_token_offset_mapping_left:last_token_offset_mapping_right]
        return text, max_token - start_token


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


def estimate_tokens_in_message(message: ChatMessage, tokenizer: Any = None) -> str:
    """
    Estimate tokens length for a single message.

    Args:
        message (OpenAIMessage): The message to estimate the tokens length for.

    Returns:
        int: The estimated tokens length.

    """
    tokens = 0

    if message.role:
        tokens += estimate_tokens_in_text(message.role, tokenizer)

    text = get_message_context(message)
    tokens += estimate_tokens_in_text(text, tokenizer)

    additional_kwargs = {**message.additional_kwargs}

    if "tool_calls" in additional_kwargs:
        for tool_call in additional_kwargs["tool_calls"]:
            tokens += estimate_tokens_in_text(str(tool_call), tokenizer)

    return tokens


def get_last_n_msgs_skip_first(msgs: List[ChatMessage], n) -> List[ChatMessage]:
    last_n_messages = msgs[1:][-n:]
    if last_n_messages and last_n_messages[0].role == MessageRole.TOOL:
        return last_n_messages[1:]
    return last_n_messages
