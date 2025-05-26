from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ImageBlock,
    TextBlock,
    ChatResponse,
)
from llama_index.core.multi_modal_llms import MultiModalLLM
from loguru import logger


context_prompt_str = """
# 下面是图片上下文描述，注意这段信息可能与图片内容矛盾或者无关，以图片内容为准。
{context_str}

"""

caption_prompt_str = """
# 任务
请使用中文为下面的图片生成简要且完整的描述。请用上图描述了/上图展示了xx开头。

"""


class ImageCaptionTool:
    def __init__(self, multimodal_llm: MultiModalLLM):
        assert (
            multimodal_llm is not None
        ), "Must provide a multimodal_llm for the image captioning tool."

        self.multimodal_llm = multimodal_llm

    def extract_url(self, image_url: str, context_str=None) -> str:
        """
        Run the image captioning model on the given image URL.
        image_url: 图片链接
        context_str: 上下文描述。
        """
        prompt = caption_prompt_str
        if context_str:
            prompt += context_prompt_str.format(context_str=context_str)

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    TextBlock(text="你是一个图片处理专家，善于提取图片里的文字信息，并给图片生成详细的描述和标签。"),
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    TextBlock(text=prompt),
                    ImageBlock(url=image_url),
                ],
            ),
        ]
        response: ChatResponse = self.multimodal_llm.chat(messages)
        logger.info("上下文介绍: ", context_str)
        logger.info("图片描述: ", response.message.content)

        return response.message.content

    async def aextract_path(self, local_image_path: str) -> str:
        """
        Run the image captioning model on the given local image path.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    TextBlock(text="你是一个图片处理专家，善于提取图片里的文字信息，并给图片生成详细的描述和标签。"),
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    TextBlock(text="请使用中文为下面的图片生成简要且完整的描述。请用上图描述了/上图展示了xx开头。"),
                    ImageBlock(path=local_image_path),
                ],
            ),
        ]

        response: ChatResponse = await self.multimodal_llm.achat(messages)
        return response.message.content

    def extract_path(self, local_image_path: str) -> str:
        """
        Run the image captioning model on the given local image path.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    TextBlock(text="你是一个图片处理专家，善于提取图片里的文字信息，并给图片生成详细的描述和标签。"),
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    TextBlock(text="请使用中文为下面的图片生成简要且完整的描述。请用上图描述了/上图展示了xx开头。"),
                    ImageBlock(path=local_image_path),
                ],
            ),
        ]

        response: ChatResponse = self.multimodal_llm.chat(messages)
        return response.message.content
