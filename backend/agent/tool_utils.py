"""Utility functions for handling tool calls and results."""

import json
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



def smart_truncate_v2(output: str, max_total_len: int = 10000) -> str:
    """
    自适应渐进式截断：
    1. 尽量保留列表中的所有项。
    2. 越靠后的项，其内部字符串被截断得越严重。
    """
    if not output or len(output) <= max_total_len:
        return output

    try:
        data = json.loads(output)
        # 开始递归处理
        # base_str_limit: 第一个元素允许的字符串长度
        # decay_factor: 每一项比前一项缩减的比例
        processed_data = _progressive_truncate_recursive(
            data,
            base_str_limit=2000,
            decay_factor=1,
        )

        final_json = json.dumps(processed_data, ensure_ascii=False, indent=2)

        # 最后的保险：如果还是超长，进行硬截断
        if len(final_json) > max_total_len:
            return final_json[:max_total_len] + "\n... [Hard Truncated]"
        return final_json

    except (json.JSONDecodeError, TypeError):
        return output[:max_total_len] + "... [Truncated]"

def _progressive_truncate_recursive(obj, base_str_limit, decay_factor):
    """
    递归函数
    :param item_index: 当前元素在所属列表中的索引（如果不在列表中则为0）
    """
    # 计算当前深度/位置下的字符串长度限制
    # 随着 index 增加，限制呈指数级下降，最小保留 100 字符
    current_str_limit = max(100, int(base_str_limit * decay_factor))

    # 1. 处理字典
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _progressive_truncate_recursive(v, base_str_limit, decay_factor)
            decay_factor *= TRUNCATE_DECAY_FACTOR
        return obj

    # 2. 处理列表
    elif isinstance(obj, list):
        new_list = []
        for i, item in enumerate(obj):
            # 对列表里的每个 item，递归调用，并传入它自己的索引 i
            decay_factor *= TRUNCATE_DECAY_FACTOR
            new_list.append(_progressive_truncate_recursive(item, base_str_limit, decay_factor))
        return new_list

    # 3. 处理字符串
    elif isinstance(obj, str):
        if len(obj) <= current_str_limit:
            return obj
        else:
            return obj[:current_str_limit] + f"...[Truncated. Total:{len(obj)} chars]"

    # 4. 其他类型
    return obj

# --- 测试演示 ---
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
    truncated_json = smart_truncate_v2(raw_json, max_total_len=10000)

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
