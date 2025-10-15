import os
from pathlib import Path
from typing import List
import pandas as pd
from transformers import AutoTokenizer


current_dir_path = Path(__file__).parent.parent
TOKENIZATION_MODEL = os.path.join(current_dir_path, "resources/tokenizer/Qwen3-32B-Tokenizer")

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZATION_MODEL, local_files_only=True, use_fast=True)
    return tokenizer

default_tokenizer = get_tokenizer()

def estimate_tokens_in_text(
    text: str,
    tokenizer: AutoTokenizer = default_tokenizer,
    return_offsets_mapping: bool = True,
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
    
    result = tokenizer(text, return_offsets_mapping=return_offsets_mapping, return_attention_mask=False, add_special_tokens=False)
    token_ids = result["input_ids"]
    return len(token_ids)

