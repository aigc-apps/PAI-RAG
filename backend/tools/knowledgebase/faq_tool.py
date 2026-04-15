from service.knowledgebase.rag_service import RagService
from service.tool.chatapp_service import ChatappService
from service.tool.faq_config_service import FAQConfigService
from db.models.faq_config import FAQConfigCreate
from typing import Annotated, Optional
from functools import partial
from llama_index.core.tools import FunctionTool
import json
from loguru import logger


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

    chatbot = await chatapp_service.get_chatapp_by_app_id(
        app_id=chatapp_id,
        tenant_id=tenant_id
    )
    if not chatbot:
        raise ValueError(f"Chat app '{chatapp_id}' does not exist.")

    # Convert dict to FAQConfigCreate object
    kb_id = None
    if chatbot.faq_config:
        try:
            faq_config = FAQConfigCreate.model_validate(chatbot.faq_config)
            kb_id = faq_config.kb_id
        except Exception as e:
            logger.warning(f"Failed to validate FAQ config for chatbot {chatapp_id}: {e}")

    if not kb_id:
        raise ValueError(f"FAQ knowledgebase not configured for chatapp '{chatapp_id}'.")

    records = await rag_service.aquery(
        query=query,
        kb_id=kb_id,
        tenant_id=tenant_id,
    )

    logger.info(
        f"Retrieved {len(records)} FAQ results for query '{query}' from knowledgebase {kb_id}."
    )

    question_in_response = faq_config.enable_question_in_response if faq_config else True
    answer_in_response = faq_config.enable_answer_in_response if faq_config else True
    return_direct = faq_config.return_direct if faq_config else False

    records_dict = []
    for record in records:
        record_dict = record.model_dump()
        metadata = record_dict.get('metadata', {}) or {}

        question = metadata.get('question', '') or ''
        answer = metadata.get('answer', '') or ''

        content_parts = []
        if not return_direct:
            if question_in_response and question:
                content_parts.append(f"Question: {question}")
            if answer_in_response and answer:
                content_parts.append(f"Answer: {answer}")

            if content_parts:
                record_dict['content'] = '\n'.join(content_parts)
            else:
                record_dict['content'] = ''
        else:
            record_dict['content'] = answer if answer else ''

        records_dict.append(record_dict)

    result = json.dumps({"result": records_dict}, ensure_ascii=False)
    logger.info(f"FAQ tool returning {len(records_dict)} results for query '{query}'")
    return result


async def aget_faq_tool(
    chatapp_id: str,
    tenant_id: str,
    user_id: Optional[str] = None,
    rag_service: RagService = None,
    chatapp_service: ChatappService = None,
    faq_config_service: FAQConfigService = None,
):
    """Create a FAQ search tool for the given chatapp_id."""
    # Get faq_config to determine return_direct value
    return_direct = False
    try:
        chatbot = await chatapp_service.get_chatapp_by_app_id(
            app_id=chatapp_id,
            tenant_id=tenant_id
        )
        if chatbot and chatbot.faq_config:
            return_direct = chatbot.faq_config.get("return_direct", False)
    except Exception as e:
        logger.warning(f"Failed to get FAQ config for return_direct: {e}, using default False")

    aquery_faq_func = partial(
        aget_faq_result,
        chatapp_id=chatapp_id,
        user_id=user_id,
        rag_service=rag_service,
        chatapp_service=chatapp_service,
        faq_config_service=faq_config_service,
        tenant_id=tenant_id,
    )


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
        description='''【最高优先级工具 - 必须优先调用且优先使用结果】

这是回答用户问题的首选工具。使用规则如下：

1. **必须首先调用**：每当用户提出任何问题时，你必须首先调用此工具从FAQ知识库中检索答案。即使你认为自己知道答案，也必须先调用此工具。

2. **必须优先使用结果**：
   - 如果FAQ工具返回的结果中result数组不为空（即len(result) > 0），说明找到了相关FAQ答案
   - **此时你必须优先直接使用这些FAQ结果来回答用户，当这些答案已经足够回答用户的问题时绝对不要再调用任何其他工具**（包括知识库工具、搜索工具等）

3. **在以下情况才可以使用其他工具**：
   - FAQ工具返回的result数组为空（len(result) == 0），表示FAQ知识库中没有相关信息
   - 或者FAQ工具调用失败
''',
        return_direct=return_direct,
    )
    return search_faq_tool
