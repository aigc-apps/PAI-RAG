from service.knowledgebase.rag_service import RagService
from service.tool.chatapp_service import ChatappService
from service.tool.faq_config_service import FAQConfigService
from typing import Annotated, Optional
from functools import partial
from llama_index.core.tools import FunctionTool
import json
from loguru import logger
from common.knowledgebase.constants import FAQ_KNOWLEDGEBASE_NAME


async def aget_faq_result(
    query: str,
    chatapp_id: str,
    user_id: str | None = None,
    rag_service: RagService | None = None,
    chatapp_service: ChatappService | None = None,
    faq_config_service: FAQConfigService | None = None,
    tenant_id: str = None,
) -> str:
    """Get FAQ search result from FAQ knowledgebase"""
    logger.info(f"Searching FAQ with chatapp_id {chatapp_id} and user {user_id}.")

    # Try to get chatbot by id first, if not found, try by app_id
    chatbot = await chatapp_service.get_chatapp(id=chatapp_id, tenant_id=tenant_id)
    if not chatbot:
        chatbot = await chatapp_service.get_chatapp_by_app_id(
            app_id=chatapp_id,
            tenant_id=tenant_id
        )
    if not chatbot:
        raise ValueError(f"应用 '{chatapp_id}' 不存在。")


    # Get FAQ knowledgebase by name: {app_id}_{FAQ_KNOWLEDGEBASE_NAME}
    kb_name = f"{chatbot.app_id}_{FAQ_KNOWLEDGEBASE_NAME}"
    kb = await rag_service.get_knowledgebase_by_name(name=kb_name, tenant_id=tenant_id)

    if not kb:
        raise ValueError(f"FAQ知识库 '{kb_name}' 不存在。")


    records = await rag_service.aquery(
        query=query,
        user_id=user_id,
        kb_id=kb.id,
        tenant_id=tenant_id,
    )

    logger.info(
        f"Retrieved {len(records)} FAQ results for query '{query}' from knowledgebase {kb.id}."
    )

    faq_config = None
    if faq_config_service:
        try:
            faq_config = await faq_config_service.get_faq_config_by_chatbot_id(
                chatbot_id=chatbot.id, tenant_id=tenant_id
            )
        except Exception as e:
            logger.warning(f"Failed to get FAQ config: {e}, using defaults")

    question_in_response = faq_config.question_in_response if faq_config else False
    answer_in_response = faq_config.answer_in_response if faq_config else True

    records_dict = []
    for record in records:
        record_dict = record.model_dump()
        metadata = record_dict.get('metadata', {}) or {}

        question = metadata.get('question', '') or ''
        answer = metadata.get('answer', '') or ''

        content_parts = []
        if question_in_response and question:
            content_parts.append(f"问题：{question}")
        if answer_in_response and answer:
            content_parts.append(f"答案：{answer}")

        if content_parts:
            record_dict['content'] = '\n'.join(content_parts)

        records_dict.append(record_dict)

    return json.dumps({"result": records_dict}, ensure_ascii=False)


async def aget_faq_tool(
    chatapp_id: str,
    tenant_id: str,
    user_id: Optional[str] = None,
    rag_service: RagService = None,
    chatapp_service: ChatappService = None,
    faq_config_service: FAQConfigService = None,
):
    """Create a FAQ search tool for the given chatapp_id."""
    aquery_faq_func = partial(
        aget_faq_result,
        chatapp_id=chatapp_id,
        user_id=user_id,
        rag_service=rag_service,
        chatapp_service=chatapp_service,
        faq_config_service=faq_config_service,
        tenant_id=tenant_id,
    )

    chatbot = await chatapp_service.get_chatapp(id=chatapp_id, tenant_id=tenant_id)
    if not chatbot:
        chatbot = await chatapp_service.get_chatapp_by_app_id(
            app_id=chatapp_id,
            tenant_id=tenant_id
        )
    if not chatbot:
        raise ValueError(f"应用 '{chatapp_id}' 不存在。")

    kb_name = f"{chatbot.app_id}_{FAQ_KNOWLEDGEBASE_NAME}"
    kb = await rag_service.get_knowledgebase_by_name(name=kb_name, tenant_id=tenant_id)

    if not kb:
        raise ValueError(f"FAQ知识库 '{kb_name}' 不存在。")

    async def query_faq_handler(
        query: Annotated[
            str,
            "根据上下文添加必要的背景信息，改写一个新的独立问题，使问题更完整，注意指代消解、完善主语等",
        ] = "",
    ):
        return await aquery_faq_func(
            query=query,
        )

    search_faq_tool = FunctionTool.from_defaults(
        async_fn=query_faq_handler,
        name=f"search-faq-{chatapp_id}",
        description=f"根据上下文从FAQ知识库中搜索和用户查询相关的内容。\n应用ID: {chatbot.app_id}\n知识库名称: {kb.name}\n知识库描述: {kb.description}\n",
    )
    return search_faq_tool
