from pairag.file.nodeparsers.pai.image_caption_tool import ImageCaptionTool
from tests.openailike_multimodal import OpenAIAlikeMultiModal
import os
import pytest


if os.environ.get("DASHSCOPE_API_KEY") is None:
    pytest.skip(reason="DASHSCOPE_API_KEY not set", allow_module_level=True)


@pytest.fixture
def multimodal_llm() -> OpenAIAlikeMultiModal:
    return OpenAIAlikeMultiModal(
        model="qwen-vl-max",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )


def test_image_caption_tool(multimodal_llm: OpenAIAlikeMultiModal):
    test_image_url = "https://feiyue-test.oss-cn-hangzhou.aliyuncs.com/pai_oss_images/EAS%E8%AE%A1%E8%B4%B9%E8%AF%B4%E6%98%8E/8ea0d46984ba9166e96f8afb58132dc6.jpeg"

    image_caption_tool = ImageCaptionTool(multimodal_llm=multimodal_llm)
    caption = image_caption_tool.extract_url(test_image_url)
    print(caption)
