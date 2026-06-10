import json
from typing import Optional, List
from llama_index.core.tools import FunctionTool
import traceback
from common.llm.llm_model import PaiLlm
from service.factory.model_factory import create_llm
from service.model.llm_service import LlmService
from loguru import logger


async def analyze_multimodal(
    image_base64_list: List[str],
    video_base64_list: List[str],
    question: str = "",
    multimodal_llm: PaiLlm = None,
) -> str:
    data_list = []
    if image_base64_list:
        # Images and videos both ride as base64 data URIs in dashscope's
        # OpenAI-compatible shorthand: the value is the data URI string (not
        # a `{url: ...}` object). This was already verified on the PAI-RAG
        # feature branch and keeps local dev working without requiring an
        # externally-reachable OSS endpoint. Vision pricing is media-based,
        # so base64 doesn't cost more tokens than passing a URL.
        data_list.extend([{"type": "image_url", "image_url": image} for image in image_base64_list])
    if video_base64_list:
        data_list.extend([{"type": "video_url", "video_url": video} for video in video_base64_list])

    system_prompt = (
        "你是一个图片和视频多模态数据理解专家。"
        "请结合用户输入的问题，对图片和视频生成尽量简洁明确的描述，不超过200字。"
    )
    user_prompt = question or "请描述图片和视频中的内容。"
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
                *data_list,
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
        logger.error(f"解析图片和视频出错: {traceback.format_exc()}")
        return json.dumps({
            "error": f"解析图片和视频出错: {str(e)}"
        }, ensure_ascii=False)


async def aget_multimodal_analysis_from_db(
    image_list: List[str],
    video_list: List[str],
    question: Optional[str] = None,
    multimodal_llm: PaiLlm = None,
) -> str:
    if not image_list and not video_list:
        return json.dumps({"error": "无法获取图片或者视频访问链接"}, ensure_ascii=False)

    try:
        answer = await analyze_multimodal(image_base64_list=image_list, video_base64_list=video_list, question=question, multimodal_llm=multimodal_llm)

        return json.dumps({
            "question": question,
            "answer": answer,
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"VLM 解析失败: {str(e)}")
        return json.dumps({
            "error": f"VLM analysis failed: {str(e)}"
        }, ensure_ascii=False)

async def aget_multimodal_analysis(
    image_list: List[str] = [],
    video_list: List[str] = [],
    question: str = None,
    llm_service: LlmService = None,
    tenant_id: str = None,
    vision_model_id: Optional[str] = None):
    """Get read file tool"""
    if not llm_service:
        raise ValueError("llm_service is required")

    multimodal_llm_config = None
    if vision_model_id:
        multimodal_llm_config = await llm_service.get_llm_by_model_id(
            model_id=vision_model_id,
            tenant_id=tenant_id,
        )
        if not multimodal_llm_config:
            raise ValueError(f"图片理解模型 {vision_model_id} 不存在。")
        if not multimodal_llm_config.enabled:
            raise ValueError(f"图片理解模型 {vision_model_id} 已禁用。")
        if not multimodal_llm_config.vision_support:
            raise ValueError(f"图片理解模型 {vision_model_id} 未开启多模态能力。")
    else:
        multimodal_llm_config = await llm_service.get_multimodal_llm(tenant_id=tenant_id)
    if not multimodal_llm_config:
        raise ValueError("要使用图片解析工具，请在模型配置页面配置多模态大模型。")
    multimodal_llm = create_llm(multimodal_llm_config)
    content = await aget_multimodal_analysis_from_db(image_list=image_list, video_list=video_list, question=question, multimodal_llm=multimodal_llm)
    return content

async def aget_multimodal_parser_tool(
    image_list: List[str] = [],
    video_list: List[str] = [],
    llm_service: LlmService = None,
    tenant_id: str = None,
    vision_model_id: Optional[str] = None,
):
    """
    创建 multimodal-parser 工具，用于解析上传的图片和视频内容。
    """
    assert image_list or video_list, "image_list or video_list is required"

    async def aget_multimodal_analysis_func(
        query: str = "请描述图片和视频中的内容。",
    ):
        return await aget_multimodal_analysis(
            image_list=image_list,
            video_list=video_list,
            question=query,
            llm_service=llm_service,
            tenant_id=tenant_id,
            vision_model_id=vision_model_id,
        )

    multimodal_parser_tool = FunctionTool.from_defaults(
        async_fn=aget_multimodal_analysis_func,
        name="multimodal-parser",
        description="""解析上传的图片和视频内容。适用于用户提问涉及图片和视频中的信息（如图表、文字、产品图等）。
参数：
- query: 用户的查询意图，例如"图中智能床的价格是多少？"、"请提取表格数据"等"。
返回：包含图片和视频分析结果的 JSON 对象。""",
        return_direct=False,
    )

    logger.info(f"Created multimodal-parser tool with image list length: {len(image_list)} and video list length: {len(video_list)}")
    return multimodal_parser_tool
