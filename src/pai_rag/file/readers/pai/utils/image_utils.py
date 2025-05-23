from PIL import Image
from io import BytesIO
from loguru import logger
import requests
from urllib.parse import urlparse


def is_remote_url(url_or_path: str) -> bool:
    result = urlparse(url_or_path)
    return result.scheme in ("http", "https", "ftp", "s3", "gs")


def image_from_bytes(
    image_blob: bytes,
    image_filename_or_extension: str,
):
    if image_filename_or_extension.lower().endswith(
        ".emf"
    ) or image_filename_or_extension.lower().endswith(".wmf"):
        # 暂时不处理Windows图元文件
        logger.warning(
            f"Skip processing EMF or WMF image: {image_filename_or_extension}"
        )
        return None

    return Image.open(BytesIO(image_blob))


def image_from_url(image_url: str):
    if is_remote_url(image_url):
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # 检查请求是否成功

            # 将二进制数据转换为图像对象
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as ex:
            logger.warning(
                f"Failed to download image from URL: {image_url}. Error: {ex}"
            )
            return None
    else:
        try:
            image = Image.open(image_url)
            return image
        except Exception as ex:
            logger.warning(f"Failed to open image from file: {image_url}. Error: {ex}")
            return None
