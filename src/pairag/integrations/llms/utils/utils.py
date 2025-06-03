import re
from llama_index.core.schema import (
    ImageNode,
)


def extract_image_links(query: str) -> list:
    # 正则表达式匹配 HTTP/HTTPS 链接（以常见图片扩展名结尾）
    http_link_pattern = r"https?://[^\s()<>]+(?:\.(?:jpe?g|png|gif|webp|svg))"

    # 正则表达式匹配 Markdown 图片语法中的链接
    markdown_pattern = r"!\[.*?\]\((https?://[^\)]+)\)"

    # 合并所有模式
    combined_pattern = f"({http_link_pattern})|({markdown_pattern})"

    # 提取所有匹配项
    matches = re.findall(combined_pattern, query)

    # 去重并返回非空结果
    return list(set([match for group in matches for match in group if match]))


def transform_to_image_nodes(query: str) -> list[ImageNode]:
    image_list = extract_image_links(query)
    image_node_list = []
    for image_url in image_list:
        image_node_list.append(ImageNode(image_url=image_url))
    return image_node_list
