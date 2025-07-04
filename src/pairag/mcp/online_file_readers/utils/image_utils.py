from loguru import logger
import re
from pairag.file.readers.pai.utils.image_utils import image_from_url


def replace_markdown_images_with_oss_urls(
    markdown_text: str, image_store, pdf_name: str
) -> str:
    """
    将 markdown 中的图片链接 ![](xxx) 替换为 OSS 的 URL。

    :param markdown_text: 原始 Markdown 文本
    :param image_store: 包含 upload_image 方法的对象
    :param pdf_name: 用于命名上传文件的标识（如 PDF 文件名）
    :return: 替换后的 Markdown
    """

    def replace_image(match):
        image_path = match.group(1)

        # 加载图像
        image = image_from_url(image_path)
        if image is None:
            return f"<!-- Failed to load image: {image_path} -->"

        # 上传图像到 OSS
        try:
            oss_url = image_store.upload_image(image, pdf_name)
            if oss_url:
                return f"![]({oss_url})"
            else:
                return f"<!-- Upload failed for image: {image_path} -->"
        except Exception as ex:
            logger.warning(f"Error uploading image {image_path}: {ex}")
            return f"<!-- Upload error: {image_path} -->"

    # 正则匹配 Markdown 中的图片格式 ![](xxx)
    pattern = r"!\$$(.*?)\$$"
    result = re.sub(pattern, replace_image, markdown_text)

    return result
