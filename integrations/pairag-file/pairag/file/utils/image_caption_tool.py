from typing import Optional
from loguru import logger
import base64

from pairag.file.utils.multimodal_llm import OpenAIMultimodalLLM


system_prompt_str = """
你是一个图片处理专家，善于找到文档中包含信息的图片，并提取图片里的文字信息，给图片生成简洁完整的描述。

## 任务描述
你需要判断图片是否包含有用的信息：
- 如果图片仅包含常见的图标、空白图片，商标，简短词语等，不适合用于展示给用户。请直接返回`[NO_IMAGE_CONTENT]`，不要返回任何其他内容。
- 如果图片包含适合展示给用户浏览的有用信息，如产品说明、操作步骤、截图等，请生成该图片的简要描述，用上图描述了/上图展示了xx开头， 不要超过300个字符。
"""

# Image MIME type mapping
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".heic": "image/heic",
}


def get_image_mime_type(file_extension: str) -> str:
    """
    Get the MIME type for an image file extension.
    
    Args:
        file_extension: File extension (e.g., '.jpg', '.png')
        
    Returns:
        MIME type string (e.g., 'image/jpeg')
    """
    ext = file_extension.lower() if file_extension.startswith('.') else f'.{file_extension.lower()}'
    return IMAGE_MIME_TYPES.get(ext, "image/jpeg")  # Default to jpeg


def encode_image_to_data_url(image_data: bytes, file_extension: str = ".jpg") -> str:
    """
    Encode image data to a base64 data URL with proper MIME type prefix.
    
    Args:
        image_data: Raw image bytes
        file_extension: File extension to determine MIME type
        
    Returns:
        Data URL string (e.g., 'data:image/jpeg;base64,...')
    """
    mime_type = get_image_mime_type(file_extension)
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{image_base64}"


class ImageCaptionTool:
    def __init__(self, multimodal_llm: OpenAIMultimodalLLM):
        assert (
            multimodal_llm is not None
        ), "Must provide a multimodal_llm for the image captioning tool."

        self.multimodal_llm = multimodal_llm

    def extract_image(self, image_data: bytes, file_extension: str = ".jpg") -> Optional[str]:
        """
        Run the image captioning model on the given image data.
        
        Args:
            image_data: Raw image bytes
            file_extension: File extension (e.g., '.jpg', '.png') for MIME type detection
            
        Returns:
            Image caption/description string, or None if no content detected
        """
        logger.info(f"[图像解析] 正在解析图片, 文件类型: {file_extension}")
        
        # Encode image with proper MIME type prefix
        image_data_url = encode_image_to_data_url(image_data, file_extension)

        caption_result = self.multimodal_llm.chat_with_images(
            system_prompt=system_prompt_str,
            image_urls=[image_data_url],
        )

        logger.info(f"[图像解析] 解析图片结果: {caption_result}")
        if not caption_result or "NO_IMAGE_CONTENT" in caption_result:
            return None
        return caption_result
