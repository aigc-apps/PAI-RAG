from llama_index.core.utils import get_tokenizer
from typing import Optional, Callable, List


def truncate(
    text: str,
    max_token: int,
    start_token: int = 0,
    tokenizer: Optional[Callable[[str], List]] = None,
) -> str:
    tokenizer = tokenizer or get_tokenizer()

    token_ids = tokenizer(text)[start_token:]

    if len(token_ids) <= max_token:
        return text

    truncated_ids = token_ids[:max_token]

    encoding = tokenizer.func.__self__
    truncated_text = encoding.decode(truncated_ids)

    return truncated_text
