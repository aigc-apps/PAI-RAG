import hashlib
from io import BytesIO
import math
from pathlib import Path
from typing import BinaryIO, Optional
from urllib.parse import urlparse
from PIL.PngImagePlugin import PngImageFile
from PIL import Image
import requests
from loguru import logger

MARKDOWN_IMAGE_PATTERN = r'!\[([^\]]*)\]\(([^)]+)\)'

IMAGE_MAX_PIXELS = 512 * 512
UNSUPPORTED_FORMATS = {"WMF", "EMF", "WMZ", "EMZ", "SVG", "EPS"}


def is_remote_url(url_or_path: str | Path) -> bool:
    result = urlparse(str(url_or_path))
    is_remote = result.scheme in ("http", "https", "ftp", "s3", "gs")
    return is_remote


def get_image_from_url(image_url: str) -> BytesIO:
    if is_remote_url(image_url):
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # 检查请求是否成功

            image_file = BytesIO(response.content)
            image_file = compress_image_if_needed(image_file)
            image_md5 = hashlib.md5(image_file.getvalue()).hexdigest()
            image_name = f"{image_md5}.jpg"
            return image_file, image_name
        except Exception as ex:
            logger.warning(
                f"Failed to download image from URL: {image_url}. Error: {ex}"
            )
            return None, None
    else:
        try:
            with open(image_url, "rb") as image_file:
                image_data = image_file.read()
                image_file = BytesIO(image_data)
                image_file = compress_image_if_needed(image_file)
                image_md5 = hashlib.md5(image_file.getvalue()).hexdigest()
                image_name = f"{image_md5}.jpg"
                return image_file, image_name

        except Exception as ex:
            logger.warning(f"Failed to open image from file: {image_url}. Error: {ex}")
            return None, None


def compress_image_if_needed(image_file: BinaryIO) -> BinaryIO:
    try:
        image: PngImageFile = Image.open(fp=image_file)
        if image.format in UNSUPPORTED_FORMATS:
            logger.warning(f"Skipping unsupported image format: {image.format}")
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_stream = BytesIO()
        image.save(image_stream, format="jpeg")

        image_stream.seek(0)

        return image_stream
    except (OSError, ValueError) as e:
        logger.warning(f"Cannot load image: {e}")
        return None



def to_markdown_image_text(image_url: str, alt: Optional[str] = "") -> str:
    return f'\n![{alt}]({image_url})\n'

def markdown_image_text_to_chunk(image_url: str, alt: Optional[str] = "") -> str:
    if alt:
        return f"![]({image_url})\n图片的描述: {alt}"
    else:
        return f"![]({image_url})\n"