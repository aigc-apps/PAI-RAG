import json
from llama_index.core.tools import FunctionTool
from typing import Annotated, List
from functools import partial
from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.db_context import with_async_db_session
from pairag.db.models.knowledgebase.file import KbFileEntity
from pairag.mcp.tools.knowledgebase.knowledgebase_tool import kb_client
from pairag.chat.models import RetrievalSetting
import re
@with_async_db_session
async def aget_file_retrieve_results_from_vector_store(session: AsyncSession, file_ids: List[str], query_str: str):
    file_res = await session.exec(
        select(KbFileEntity).where(
            KbFileEntity.frontend_file_id.in_(file_ids)
        )
    )
    processed_file_entities = file_res.all()
    document_ids = [entity.id for entity in processed_file_entities]
    unique_kb_ids = list({entity.kb_id for entity in processed_file_entities})
    assert len(unique_kb_ids) == 1, "file_ids must be from the same knowledgebase"

    node_results = await kb_client.aquery_for_attachments(
        query=query_str,
        knowledge_id=unique_kb_ids[0],
        retrieval_setting=RetrievalSetting(top_k=5, score_threshold=0.1),
        document_ids=document_ids
    )
    records = []
    for score_node in node_results:
        origin_text = score_node.node.get_content()
        pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"'
        matches = re.findall(pattern, origin_text)
        score_node.node.metadata["images_info"] = [{"url":src, "desc": alt } for src, alt in matches]
        records.append({
            "title": score_node.node.metadata.get("file_name", "null"),
            "content": score_node.node.get_content(),
            "score": score_node.score,
        })
    logger.info(
        f"Retrieved {len(node_results)} for query '{query_str}' against knowledgebase {unique_kb_ids[0]}."
    )
    return records

async def aget_file_retrieve_results(
    attachments: List[dict],
    query_str: str
):
    """Get retrieve file tool"""
    file_ids = [attachment["id"] for attachment in attachments]
    results = await aget_file_retrieve_results_from_vector_store(file_ids=file_ids, query_str=query_str)
    data = {"query_str": query_str, "content": results}
    return json.dumps(data, ensure_ascii=False)


async def aget_file_searcher(attachments: List[dict]):
    get_file_retrieve_results_func = partial(aget_file_retrieve_results, attachments=attachments)
    async def file_retrieve_handler(
        query_str: Annotated[
            str,
            "用户的问题",
        ] = "",
    ):
        logger.info(
            f"File_retrieve_tool with attachments: {attachments}"
        )
        return await get_file_retrieve_results_func(
            query_str=query_str
        )

    think_tool = FunctionTool.from_defaults(
        async_fn=file_retrieve_handler,
        name="search-file",
        description="文件检索工具：如果附件中并没有直接提供关于用户问题的相关信息，请使用该工具进一步搜索附件里的更多信息。",
    )
    return think_tool
