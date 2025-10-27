# 将DataFrame 切分为固定小于Token Count的文本块


from typing import List
import pandas as pd
from pairag.file.utils.tokenization import estimate_tokens_in_text


def split_dataframe(df: pd.DataFrame, max_tokens: int) -> List[str]:
    """
    将DataFrame切分为固定小于Token Count的文本块
    """
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return []

    headers = df.columns

    header_length = estimate_tokens_in_text(text=" ".join([str(col) for col in headers]), return_offsets_mapping=False)
    max_chunk_size = max_tokens - header_length - 10

    chunks = []
    start_index = 0
    current_index = start_index
    current_chunk_size = 0
    while current_index < rows:
        # at least one row in each chunk
        row = df.iloc[start_index]
        row_size = estimate_tokens_in_text(text=" ".join([str(cell) for cell in row]), return_offsets_mapping=False)

        # 如果未超出token上限或者是第一行
        if row_size + current_chunk_size <= max_chunk_size or current_index == start_index:
            current_chunk_size += row_size
            current_index += 1
        else:
            chunk = df.iloc[start_index:current_index].to_markdown(tablefmt="github", index=False)
            chunks.append(chunk)
            start_index = current_index
            current_chunk_size = 0        

    if start_index < current_index:
        chunk = df.iloc[start_index:current_index].to_markdown(tablefmt="github", index=False)
        chunks.append(chunk)
        
    return chunks