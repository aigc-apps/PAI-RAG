import re


def extract_image_links(query: str) -> list:
    # 匹配以常见图片扩展名结尾的 HTTP/HTTPS 链接
    http_image_pattern = r"https?://[^\s'\"<>]+\.(?:jpe?g|png|gif|webp|svg)"

    # 提取所有匹配项
    matches = re.findall(http_image_pattern, query)

    # 去重并返回
    return list(set(matches))
