import json
from typing import Optional, List
from llama_index.core.tools import FunctionTool
import traceback
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ImageBlock,
    TextBlock,
    ChatResponse,
)
from common.llm.llm_model import PaiLlm
from service.factory.model_factory import create_llm
from service.model.llm_service import LlmService
from loguru import logger


async def analyze_image(
    image_url_list: List[str],
    question: str = "",
    multimodal_llm: PaiLlm = None,
) -> str:
    system_prompt = (
        "你是一个图片理解专家。"
        "请结合用户输入的问题，对图片生成尽量简洁明确的描述，不超过200字。"
    )
    user_prompt = question or "请描述图片的内容。"
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=[
                TextBlock(text=system_prompt),
            ],
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=[
                TextBlock(text=user_prompt),
                *[ImageBlock(url=image_url) for image_url in image_url_list],
            ],
        ),
    ]
    try:
        response: ChatResponse = await multimodal_llm.achat(messages)
        raw_output = response.message.content.strip()
        return raw_output
    except Exception:
        logger.error(f"解析图片出错: {traceback.format_exc()}")
        raise


async def aget_image_analysis_from_db(
    image_url_list: List[str],
    question: Optional[str] = None,
    llm_service: LlmService = None,
) -> str:
    if not image_url_list:
        return json.dumps({"error": "无法获取图片访问链接"}, ensure_ascii=False)

    try:
        answer = await analyze_image(image_url_list, question)

        return json.dumps({
            "image_url_list": image_url_list,
            "question": question,
            "answer": answer,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"VLM 解析失败: {str(e)}")
        return json.dumps({
            "error": f"VLM 解析失败: {str(e)}"
        }, ensure_ascii=False)

async def aget_image_analysis(
    image_url_list: List[str],
    question: Optional[str] = None,
    llm_service: LlmService = None):
    """Get read file tool"""
    if not llm_service:
        raise ValueError("llm_service is required")

    multimodal_llm_config = await llm_service.get_multimodal_llm()
    if not multimodal_llm_config:
        raise ValueError("要使用图片解析工具，请在模型配置页面配置多模态大模型。")
    multimodal_llm = create_llm(multimodal_llm_config)
    content = await aget_image_analysis_from_db(image_url_list=image_url_list, question=question, multimodal_llm=multimodal_llm)
    return content

async def aget_image_parser_tool(
    llm_service: LlmService = None,
):
    """
    创建 image-parser 工具，用于解析上传的图片内容。
    """
    async def aget_image_analysis_func(
        image_url_list: List[str],
        question: Optional[str] = None,
    ):
        return await aget_image_analysis(image_url_list=image_url_list, question=question, llm_service=llm_service)
    image_parser_tool = FunctionTool.from_defaults(
        async_fn=aget_image_analysis_func,
        name="image-parser",
        description="""解析上传的图片内容。适用于用户提问涉及图片中的信息（如图表、文字、产品图等）。
参数：
- image_url_list: 必填，图片的url列表。
- question: 可选，用户想问的具体问题，例如“图中智能床的价格是多少？”、“请提取表格数据”等。
返回：包含图片分析结果的 JSON 对象。""",
    )
    return image_parser_tool
