import json
from typing import Optional, List
from llama_index.core.tools import FunctionTool
from sqlmodel.ext.asyncio.session import AsyncSession

from db.db_context import with_async_db_session
from tools.llm_utils import get_multimodal_llm_from_db
import traceback
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ImageBlock,
    TextBlock,
    ChatResponse,
)
from loguru import logger


async def analyze_image(image_url_list: List[str], question: str = "") -> str:
    multimodal_llm = await get_multimodal_llm_from_db()
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


@with_async_db_session
async def aget_image_analysis_from_db(
    session: AsyncSession,
    image_url_list: List[str],
    question: Optional[str] = None
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
    question: Optional[str] = None):
    """Get read file tool"""
    content = await aget_image_analysis_from_db(image_url_list=image_url_list, question=question)
    return content

async def aget_image_parser_tool():
    """
    创建 image-parser 工具，用于解析上传的图片内容。
    """
    image_parser_tool = FunctionTool.from_defaults(
        async_fn=aget_image_analysis,
        name="image-parser",
        description="""解析上传的图片内容。适用于用户提问涉及图片中的信息（如图表、文字、产品图等）。
参数：
- image_url_list: 必填，图片的url列表。
- question: 可选，用户想问的具体问题，例如“图中智能床的价格是多少？”、“请提取表格数据”等。
返回：包含图片分析结果的 JSON 对象。""",
    )
    return image_parser_tool
