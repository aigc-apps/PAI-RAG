import re


MARKDOWN_IMAGE_PATTERN = r'!\[.*?\]\((.*?)\)\s*\n*\s*图片的描述:\s*(.*?)(?=\n\n|$)'
MAX_TRUNCATED_CHUNK_LEN = 8000


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. 移除 NUL 和其他控制字符 (保留 \t \n \r)
    # 允许 0x09 (tab), 0x0A (LF), 0x0D (CR)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)

    # 2. 移除 Unicode 替换字符（解码失败标志）
    text = text.replace('\uFFFD', ' ')

    # 3. （可选）移除零宽字符
    text = re.sub(r'[\u200B-\u200D\uFEFF]', ' ', text)

    # 4. （可选）规范化换行：\r\n 或 \r → \n
    text = re.sub(r'\r\n?', '\n', text)

    return text


def get_node_texts_for_embedding(nodes) -> list[str]:
    texts = []
    for node in nodes:
        # FAQ nodes need pure question/answer text for precise matching —
        # skip file_name/title prefixes that would add irrelevant noise.
        if node.metadata.get('is_faq'):
            texts.append(node.text[:3000])
            continue

        # The document title is already prepended to node.text by
        # process_file_async(), so we only prefix the file_name here
        # for additional source traceability.
        base_text = ""
        file_name = node.metadata.get('file_name', '').strip()
        if file_name:
            base_text += f"file_name: {file_name}\n\n"

        base_text += node.text

        texts.append(base_text[:3000])
    return texts
