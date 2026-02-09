"""Utility functions for handling tool calls and results."""

import json
import math
from typing import Optional
from llama_index.core.tools.function_tool import FunctionTool
from common.llm.models import TextChunk
from loguru import logger

TRUNCATE_DECAY_FACTOR = 0.8

def check_and_handle_return_direct(
    tool_obj: FunctionTool,
    tool_name: str,
    tool_content: Optional[str],
    tool_error: Optional[str],
) -> Optional[TextChunk]:
    """
    Check if a tool has return_direct=True and format the result accordingly.

    Args:
        tool_obj: The FunctionTool object
        tool_name: Name of the tool
        tool_content: Content returned by the tool (None if error)
        tool_error: Error message if tool call failed (None if success)
        agent_name: Name of the agent (for logging)

    Returns:
        TextChunk if return_direct=True, None otherwise
    """
    return_direct = getattr(tool_obj.metadata, 'return_direct', False)

    if not return_direct:
        return None

    logger.info(f"Tool {tool_name} has return_direct=True, returning tool result directly.")

    if tool_error:
        return TextChunk(delta="Tool call failed: {tool_error}")

    if not tool_content:
        return TextChunk(delta="Tool call successful, but no content returned.")

    try:
        result_data = json.loads(tool_content)
        if isinstance(result_data, dict) and "result" in result_data:
            # Format FAQ results or similar structured results
            formatted_result = ""
            for item in result_data.get("result", []):
                if isinstance(item, dict):
                    content = item.get("content", "")
                    if content:
                        formatted_result += content + "\n\n"
            if formatted_result:
                return TextChunk(delta=formatted_result.strip())
            else:
                return TextChunk(delta=tool_content)
        else:
            return TextChunk(delta=tool_content)
    except (json.JSONDecodeError, Exception):
        return TextChunk(delta=tool_content)





def truncate_json_proportionally(data, cut_size):
    """
    Truncates strings over 300 characters in a JSON-like object
    proportionally based on their length.
    """
    threshold = 300
    candidates = [] # Stores (parent_container, key_or_index, length)
    total_len = 0

    # --- Pass 1: Recursive search to find candidates and total length ---
    def find_candidates(obj):
        nonlocal total_len

        # If it's a dictionary, iterate through keys and values
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and len(value) > threshold:
                    candidates.append((obj, key, len(value)))
                    total_len += len(value)
                else:
                    find_candidates(value)

        # If it's a list, iterate through indices and values
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, str) and len(value) > threshold:
                    candidates.append((obj, i, len(value)))
                    total_len += len(value)
                else:
                    find_candidates(value)

    find_candidates(data)

    # If no strings meet the criteria, return early
    if not candidates or total_len == 0:
        return data

    # --- Pass 2: Calculate weights and truncate ---
    for parent, key, original_len in candidates:
        # Calculate weight: cut_size * (string_len / total_len)
        # Using math.ceil as requested
        reduction = math.ceil(cut_size * (original_len / total_len))

        # Calculate the new length
        new_len = max(0, original_len - reduction)

        # Update the string in the actual container
        # parent[key] accesses the dict key or list index directly
        parent[key] = parent[key][:new_len]

    return data

def smart_truncate_v2(output: str, max_length: int = 15000) -> str:
    """
    按照结构折叠和截断输出字符串
    """
    if not output or len(output) <= max_length:
        return output

    try:
        # 1. 尝试解析 JSON
        data = json.loads(output)
    except Exception:
        # 解析失败，直接按长度截断
        return output[:max_length]


    length_gap =  len(output) - max_length
    result_obj = truncate_json_proportionally(data, length_gap)

    # 最终转换为字符串返回
    try:
        return json.dumps(result_obj, ensure_ascii=False)
    except Exception:
        return str(result_obj)


if __name__ == "__main__":
    # 模拟一个有 20 个搜索结果的工具输出
    mock_results = {
        "results": [{
            "title": f"Result {i}",
            "content": f"Index {i}: " + "Very long content... " * 50, # 原始长度约 1000
            "url": f"http://example.com/{i}"
        } for i in range(20)]
    }

    raw_json = json.dumps(mock_results)
    print(f"原始 JSON 长度: {len(raw_json)}")

    # 执行渐进式截断
    truncated_json = smart_truncate_v2(raw_json, max_length=10000)

    print(f"截断后 JSON 长度: {len(truncated_json)}")

    # 解析回来看看每项的长度变化
    result_data = json.loads(truncated_json)
    for i, item in enumerate(result_data["results"]):
        content_len = len(item['content'])
        print(f"Item {i:02d} content 长度: {content_len}")

    mock_results_str = json.dumps({"result": "abcabc" * 30000})
    print(f"原始 JSON 长度: {len(mock_results_str)}")
    truncated_json = smart_truncate_v2(mock_results_str)
    print(f"截断后 JSON 长度: {len(truncated_json)}")
