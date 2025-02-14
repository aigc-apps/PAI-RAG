import json
from loguru import logger


def parse_json_from_code_block_str(input_str):
    start = input_str.find("{")
    end = input_str.find("}", start + 1)
    if start != -1 and end != -1:
        content = input_str[start : end + 1]
        try:
            data = json.loads(content)
            logger.debug("解析后的 JSON 对象：", data)
            return data
        except json.JSONDecodeError as e:
            logger.debug("JSON 解码错误:", e)
            return json.loads('{ "queries": [] }')
    else:
        logger.debug("未找到有效的JSON对象。")
        return json.loads('{ "queries": [] }')
