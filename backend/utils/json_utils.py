import json
from typing import Dict
from llama_index.core.llms.utils import parse_partial_json
from loguru import logger


def parse_tool_arguments(json_str: str) -> Dict:
    """
    解析工具参数 JSON 字符串，尝试多种方法处理格式不正确的 JSON。

    解析策略（按顺序尝试）：
    1. 标准 JSON 解析
    2. 使用 parse_partial_json 处理部分 JSON
    3. 尝试修复常见的 JSON 格式问题（移除末尾的额外字符等）

    Args:
        json_str: 要解析的 JSON 字符串
        agent_name: Agent 名称，用于日志记录

    Returns:
        解析后的字典，如果所有方法都失败则返回空字典
    """
    if not json_str:
        return {}

    # 首先尝试标准 JSON 解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 如果标准解析失败，尝试使用 parse_partial_json 处理部分 JSON
    try:
        result = parse_partial_json(json_str)
        if result:
            return result
    except Exception:
        pass

    # 如果都失败，尝试简单的字符串修复
    try:
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')

        if first_brace >= 0 and last_brace > first_brace:
            fixed_json = json_str[first_brace:last_brace + 1]
            return json.loads(fixed_json)
        else:
            raise ValueError("No valid JSON braces found")
    except Exception:
        logger.warning(f"Invalid JSON args: {json_str[:200]}")
        return {}
