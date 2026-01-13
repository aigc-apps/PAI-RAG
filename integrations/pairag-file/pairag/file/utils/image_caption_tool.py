import traceback
from typing import List
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ImageBlock,
    TextBlock,
    ChatResponse,
)
from llama_index.core.multi_modal_llms import MultiModalLLM
from loguru import logger
import base64


system_prompt_str = """
你是一个图片处理专家，善于找到文档中包含信息的图片，并提取图片里的文字信息，给图片生成简洁完整的描述。

## 任务描述
你需要判断图片是否包含有用的信息：
- 如果图片仅包含常见的图标、空白图片，商标，简短词语等，不适合用于展示给用户。请直接返回`[NO_IMAGE_CONTENT]`，不要返回任何其他内容。
- 如果图片包含适合展示给用户浏览的有用信息，如产品说明、操作步骤、截图等，请生成该图片的简要描述，用上图描述了/上图展示了xx开头， 不要超过300个字符。
"""


class ImageCaptionTool:
    def __init__(self, multimodal_llm: MultiModalLLM):
        assert (
            multimodal_llm is not None
        ), "Must provide a multimodal_llm for the image captioning tool."

        self.multimodal_llm = multimodal_llm

    def _get_result(self, messages: List[ChatMessage]) -> str:
        try:
            response: ChatResponse = self.multimodal_llm.chat(messages)
            return response.message.content
        except Exception:
            logger.error(f"解析图片出错: {traceback.format_exc()}")
            raise

    def extract_url(self, image_url: str, context_str=None) -> str:
        """
        Run the image captioning model on the given image URL.
        image_url: 图片链接
        context_str: 上下文描述。
        """
        logger.info(f"[图像解析] 正在解析图片: {image_url}")
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    TextBlock(text=system_prompt_str),
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    ImageBlock(url=image_url),
                ],
            ),
        ]
        result = self._get_result(messages)
        logger.info(f"[图像解析] 图片链接: {image_url} \n图片描述: {result}")

        if "NO_IMAGE_CONTENT" in result:
            return None
        return result


    def extract_image(self, image_data: bytes, context_str=None) -> str:
        """
        Run the image captioning model on the given image URL.
        image_data: 图片数据
        context_str: 上下文描述。
        """
        logger.info(f"[图像解析] 正在解析图片")
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    TextBlock(text=system_prompt_str),
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    ImageBlock(image=image_base64),
                ],
            ),
        ]
        result = self._get_result(messages)
        logger.info(f"[图像解析] 解析图片结果: {result}")
        if "NO_IMAGE_CONTENT" in result:
            return None
        return result
