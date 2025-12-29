from service.knowledgebase.rag_service import RagService
from typing import Annotated, Optional
from functools import partial
from llama_index.core.tools import FunctionTool
import json
from loguru import logger
from common.chat.models import MetadataFilteringCondition


async def aget_knowledgebase_result(
    query: str,
    kb_id: str,
    user_id: str | None =None,
    rag_service: RagService | None = None,
    tenant_id: str = None,
    metadata_condition: Optional[MetadataFilteringCondition] = None,
) -> str:
    """Get aliyun search tool"""
    logger.info(f"Searching knowledgebase with kb {kb_id} and user {user_id}.")
    records = await rag_service.aquery(query=query, kb_id=kb_id, user_id=user_id, tenant_id=tenant_id, metadata_condition=metadata_condition)
    records_dict = [record.model_dump() for record in records]
    return json.dumps({"result": records_dict}, ensure_ascii=False)


async def aget_knowledgebase_tool(
    kb_id: str,
    tenant_id: str,
    user_id: Optional[str] = None,
    rag_service: RagService = None,
    metadata_condition: Optional[MetadataFilteringCondition] = None,
):
    aquery_knowledgebase_func = partial(
        aget_knowledgebase_result,
        kb_id=kb_id,
        user_id=user_id,
        rag_service=rag_service,
        tenant_id=tenant_id,
        metadata_condition=metadata_condition,
    )
    knowledgebase = await rag_service.get_knowledgebase(kb_id=kb_id, tenant_id=tenant_id)
    if not knowledgebase:
        raise ValueError(f"Knowledgebase {kb_id} not found.")

    async def query_knowledgebase_handler(
        query: Annotated[
            str,
            "根据上下文添加必要的背景信息，改写一个新的独立问题，使问题更完整，注意指代消解、完善主语等",
        ] = "",
    ):
        return await aquery_knowledgebase_func(
            query=query,
        )

    search_knowledgebase_tool = FunctionTool.from_defaults(
        async_fn=query_knowledgebase_handler,
        name=f"search-knowledgebase-{kb_id}",
        description=f"根据上下文从知识库中搜索和用户查询相关的内容。\n知识库名称: {knowledgebase.name}\n知识库描述: {knowledgebase.description}\n",
    )
    return search_knowledgebase_tool
