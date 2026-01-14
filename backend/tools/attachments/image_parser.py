import json
from typing import Optional, List
from llama_index.core.tools import FunctionTool
import traceback
from common.llm.llm_model import PaiLlm
from service.factory.model_factory import create_llm
from service.model.llm_service import LlmService
from loguru import logger


async def analyze_image(
    image_base64_list: List[str],
    question: str = "",
    multimodal_llm: PaiLlm = None,
) -> str:
    system_prompt = (
        "你是一个图片理解专家。"
        "请结合用户输入的问题，对图片生成尽量简洁明确的描述，不超过200字。"
    )
    user_prompt = question or "请描述图片的内容。"
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                *[{"type": "image_url", "image_url": image} for image in image_base64_list],
            ],
        }
    ]
    try:
        response = ""
        response_gen = await multimodal_llm.astream(messages)
        async for chunk in response_gen:
            response += chunk.delta
        return response
    except Exception as e:
        logger.error(f"解析图片出错: {traceback.format_exc()}")
        return json.dumps({
            "error": f"解析图片出错: {str(e)}"
        }, ensure_ascii=False)


async def aget_image_analysis_from_db(
    image_list: List[str],
    question: Optional[str] = None,
    multimodal_llm: PaiLlm = None,
) -> str:
    if not image_list:
        return json.dumps({"error": "无法获取图片访问链接"}, ensure_ascii=False)

    try:
        answer = await analyze_image(image_list, question, multimodal_llm)

        return json.dumps({
            "question": question,
            "answer": answer,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"VLM 解析失败: {str(e)}")
        return json.dumps({
            "error": f"VLM 解析失败: {str(e)}"
        }, ensure_ascii=False)

async def aget_image_analysis(
    image_list: List[str],
    question: str = None,
    llm_service: LlmService = None,
    tenant_id: str = None):
    """Get read file tool"""
    if not llm_service:
        raise ValueError("llm_service is required")

    multimodal_llm_config = await llm_service.get_multimodal_llm(tenant_id=tenant_id)
    if not multimodal_llm_config:
        raise ValueError("要使用图片解析工具，请在模型配置页面配置多模态大模型。")
    multimodal_llm = create_llm(multimodal_llm_config)
    content = await aget_image_analysis_from_db(image_list=image_list, question=question, multimodal_llm=multimodal_llm)
    return content

async def aget_image_parser_tool(
    image_list: List[str],
    llm_service: LlmService = None,
    tenant_id: str = None,
):
    """
    创建 image-parser 工具，用于解析上传的图片内容。
    """
    async def aget_image_analysis_func(
        query: str = "请描述图片的内容。",
    ):
        return await aget_image_analysis(image_list=image_list, question=query, llm_service=llm_service, tenant_id=tenant_id)
    image_parser_tool = FunctionTool.from_defaults(
        async_fn=aget_image_analysis_func,
        name="image-parser",
        description="""解析上传的图片内容。适用于用户提问涉及图片中的信息（如图表、文字、产品图等）。
参数：
- query: 用户的查询意图，例如"图中智能床的价格是多少？"、"请提取表格数据"等"。
返回：包含图片分析结果的 JSON 对象。""",
        return_direct=False,
    )
    return image_parser_tool
