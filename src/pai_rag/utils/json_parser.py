import json
import re


def parse_json_from_code_block_str(input_str):
    pattern = r"\{.*\}"
    match = re.search(pattern, input_str, re.DOTALL)

    if match:
        json_str = match.group()
        print(f"提取的 JSON 字符串:\n{json_str}\n")

        try:
            # 解析 JSON 字符串
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return json.loads('{ "queries": [] }')
    else:
        print("未找到匹配的 JSON 对象。")
        return json.loads('{ "queries": [] }')
