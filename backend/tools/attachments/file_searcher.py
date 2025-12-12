import json
from llama_index.core.tools import FunctionTool
from typing import Annotated, List
from functools import partial
from loguru import logger
from common.chat.models import RetrievalSetting
from service.knowledgebase.rag_service import RagService
from common.knowledgebase.constants import ATTACHMENT_KNOWLEDGEBASE_NAME
from common.tool.search_result import SearchResult


async def aget_file_retrieve_results(
    doc_ids: List[str],
    query_str: str,
    rag_service: RagService,
):
    """Get retrieve file tool"""
    if not doc_ids:
        return []

    search_results: List[SearchResult] = await rag_service.aquery(
        query=query_str,
        kb_name=ATTACHMENT_KNOWLEDGEBASE_NAME,
        retrieval_setting=RetrievalSetting(top_k=5, score_threshold=0.1),
        document_ids=doc_ids,
    )
    records = []
    for node in search_results:
        records.append({
            "title": node.title,
            "content": node.content,
            "score": node.score,
        })
    data = {"query_str": query_str, "content": search_results}
    return json.dumps(data, ensure_ascii=False)


async def aget_file_searcher(rag_service: RagService):
    get_file_retrieve_results_func = partial(aget_file_retrieve_results, rag_service=rag_service)
    async def file_retrieve_handler(
        query_str: Annotated[
            str,
            "用户的问题",
        ] = "",
        doc_ids: Annotated[
            List[str],
            "The IDs of the documents to search.",
        ] = [],
        **kwargs
    ):
        logger.info(
            f"File_searcher_tool with doc_ids with query_str: {query_str}, kwargs: {kwargs}"
        )
        return await get_file_retrieve_results_func(
            query_str=query_str,
            doc_ids=doc_ids,
        )

    search_tool = FunctionTool.from_defaults(
        async_fn=file_retrieve_handler,
        name="search-file",
        description="""File Search Tool: Use this tool to retrieve additional information from attached documents when the initial response is truncated (e.g., contains "[truncated]") and does not directly answer the user's question.
Parameters:
- query_str (str): A clear and specific query describing what information you need from the file (e.g., "What is shown in the image?", "Find the content of Chapter 3", or "Summarize the section about climate trends").
- doc_ids (List[str]): The IDs of the documents to search.
- kwargs (dict, optional): Additional arguments for the tool.
""",
    )
    return search_tool
