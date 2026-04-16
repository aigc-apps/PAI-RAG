import traceback
from typing import List, Optional
from loguru import logger
from common.chat.models import RetrievalSetting
from common.chat.models import MetadataFilteringCondition
from pydantic import BaseModel, Field
from mcp.server.fastmcp.tools import Tool
from mcp.server.fastmcp.utilities.func_metadata import func_metadata
from db.db_context import with_async_db_session
from service.injection import get_rag_service
from sqlmodel.ext.asyncio.session import AsyncSession
from service.knowledgebase.rag_service import RagService


class NodeMetadata(BaseModel):
    file_path: str
    image_url: List[str]
    title: str
    doc_name: str

class NodeInfo(BaseModel):
    metadata: NodeMetadata
    text: str = Field(description="The text content of the node")


class NodeResult(BaseModel):
    score: float
    node: NodeInfo


class RetrievalToolResponse(BaseModel):
    status: str
    status_code: int
    message: Optional[str] = None
    total: int
    nodes: List[NodeResult] = []
    request_id: str


async def asearch_knowledgebase(
    query: str,
    knowledgebase_id: str,
    tenant_id: str,
    image_list: List[str] = [],
    user_id: Optional[str] = None,
    retrieval_setting: Optional[RetrievalSetting] = None,
    metadata_condition: Optional[MetadataFilteringCondition] = None,
    rag_service: RagService = None,
    request_id: str = None,
) -> RetrievalToolResponse:
    # TODO: Handle images if needed in the future
    # For now, images are accepted but not used in retrieval
    if image_list:
        logger.warning(f"Received {len(image_list)} images, but image-based retrieval is not yet implemented.")

    try:
        # Perform retrieval using the same strategy as retrieval.py
        node_results = await rag_service.aquery(
            query=query,
            user_id=user_id,
            kb_id=knowledgebase_id,
            retrieval_setting=retrieval_setting,
            metadata_condition=metadata_condition,
            tenant_id=tenant_id,
        )

        logger.info(
            f"Retrieved {len(node_results)} nodes for query '{query}' against knowledgebase {knowledgebase_id}."
        )

        # Transform results to the required format
        nodes = []
        for score_node in node_results:
            # Extract title - prefer title, then file_name
            title = score_node.title
            # Extract doc_name - prefer doc_name, then file_name
            doc_name = score_node.title
            # Extract file_path
            file_path = score_node.url

            # Build the node in the required format
            # Include all metadata fields but structure the required ones at the top level
            # score_node is a SearchResult object, use attribute access instead of .get()
            images = score_node.images or []
            node = NodeResult(
                score=score_node.score,
                node=NodeInfo(
                    text=score_node.content,
                    metadata=NodeMetadata(
                        file_path=file_path or "",
                        image_url=[img.get("url", "") for img in images if isinstance(img, dict) and img.get("url", "")],
                        title=title or "",
                        doc_name=doc_name or "",
                    ),
                )
            )
            nodes.append(node)

        return RetrievalToolResponse(
            status="SUCCESS",
            status_code=200,
            total=len(nodes),
            nodes=nodes,
            message=None,
            request_id=request_id
        )
    except Exception as ex:
        logger.exception(f"Retrieval tool failed: {traceback.format_exc()}")
        return RetrievalToolResponse(
            status="ERROR",
            status_code=500,
            total=0,
            nodes=[],
            message=str(ex),
            request_id=request_id
        )


def get_retrieval_tool(
    knowledgebase_id: str,
    kb_name: str,
    kb_description: str,
    tenant_id: str,
) -> Tool:
    func_name = f"search-knowledgebase-{knowledgebase_id}"
    func_description = f"从知识库中搜索和用户查询相关的内容。\n知识库名称: {kb_name}\n知识库描述: {kb_description}\n"

    @with_async_db_session
    async def asearch_knowledgebase_wrapper(
        query: str,
        image_list: List[str] = [],
        user_id: Optional[str] = None,
        retrieval_setting: Optional[RetrievalSetting] = None,
        metadata_condition: Optional[MetadataFilteringCondition] = None,
        session: AsyncSession = None,
    ) -> RetrievalToolResponse:
        """
        Search knowledgebase function.
        """
        rag_service = await get_rag_service(session=session)
        return await asearch_knowledgebase(
            query=query,
            knowledgebase_id=knowledgebase_id,
            tenant_id=tenant_id,
            image_list=image_list,
            user_id=user_id,
            retrieval_setting=retrieval_setting,
            metadata_condition=metadata_condition,
            rag_service=rag_service,
        )

    func_arg_metadata = func_metadata(
        asearch_knowledgebase_wrapper,
    skip_names=["user_id", "knowledgebase_id", "retrieval_setting", "session"],
        structured_output=True,
    )
    parameters = func_arg_metadata.arg_model.model_json_schema(by_alias=True)

    return Tool(
        fn=asearch_knowledgebase_wrapper,
        name=func_name,
        description=func_description,
        fn_metadata=func_arg_metadata,
        parameters=parameters,
        is_async=True,
    )
