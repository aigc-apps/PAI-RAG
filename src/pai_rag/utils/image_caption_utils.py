from pai_rag.integrations.llms.pai.pai_multi_modal_llm import PaiMultiModalLlm
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ImageBlock,
    TextBlock,
    ChatResponse,
)


class ImageCaptionTool:
    def __init__(self, multimodal_llm: PaiMultiModalLlm):
        assert (
            multimodal_llm is not None
        ), "Must provide a multimodal_llm for the image captioning tool."

        self.multimodal_llm = multimodal_llm

    def extract_url(self, image_url: str, pre_content=None) -> str:
        """
        Run the image captioning model on the given image URL.
        image_url: 图片链接
        pre_content: 上下文描述。
        """
        if not pre_content:
            prompt = """1.请提取图片里的文字信息。
                        2.请使用中文为下面的图片生成简要且完整的描述。请用上图描述了/上图展示了xx开头。"""
        else:
            prompt = f"""
# 图片上下文描述
{pre_content}

# 任务
1. 请提取图片里的文字信息。
2. 请参考图片的上下文材料，使用中文为下面的图片生成简要且完整的描述。请用上图描述了/上图展示了xx开头。"""
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
        print("上下文介绍: ", pre_content)
        print("图片描述: ", response.message.content)

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
                    TextBlock(text="请提取下面图片里的文字信息，并为下面的图片生成详细的描述和标签信息:"),
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
                    TextBlock(text="请提取下面图片里的文字信息，并为下面的图片生成详细的描述和标签信息:"),
                    ImageBlock(path=local_image_path),
                ],
            ),
        ]

        response: ChatResponse = self.multimodal_llm.chat(messages)
        return response.message.content
