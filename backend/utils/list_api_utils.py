from typing import List

# 1. 定义解析函数：将 "1,2,3" 转换为 ["1", "2", "3"]
# Pydantic 后续会自动将字符串列表转换为整数列表
def parse_comma_separated_list(v: str | List[str] | None) -> List[str]:
    if not v:
        return []

    # 如果已经是列表（防御性编程），直接返回
    if isinstance(v, list):
        return v
    # 如果是字符串，按逗号分割并去除空格
    elif isinstance(v, str):
        return [i.strip() for i in v.split(",") if i.strip()]
    else:
        raise ValueError(f"Invalid input: {v}")
