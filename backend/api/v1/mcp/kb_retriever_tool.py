import traceback
import uuid
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

class NodeResult(BaseModel):
    score: float
    metadata: NodeMetadata
    text: str = Field(description="The text content of the node")


class RetrievalResult(BaseModel):
    total: int
    nodes: List[NodeResult] = []


class RetrievalToolResponse(BaseModel):
    status: str
    status_code: int
    data: RetrievalResult
    request_id: str


async def asearch_knowledgebase(
    query: str,
    knowledgebase_id: str,
    image_list: List[str] = [],
    user_id: Optional[str] = None,
    retrieval_setting: Optional[RetrievalSetting] = None,
    metadata_condition: Optional[MetadataFilteringCondition] = None,
    rag_service: RagService = None,
) -> RetrievalToolResponse:
    request_id = str(uuid.uuid4())

    # TODO: Handle images if needed in the future
    # For now, images are accepted but not used in retrieval
    if image_list:
        logger.warning(f"Received {len(image_list)} images, but image-based retrieval is not yet implemented.")

    try:
        # Perform retrieval using the same strategy as retrieval.py
        node_results = await rag_service.aquery(
            query=query,
            user_id=user_id,
            knowledge_id=knowledgebase_id,
            retrieval_setting=retrieval_setting,
            metadata_condition=metadata_condition,
        )

        logger.info(
            f"Retrieved {len(node_results)} nodes for query '{query}' against knowledgebase {knowledgebase_id}."
        )

        # Transform results to the required format
        nodes = []
        for score_node in node_results:
            # Extract title - prefer title, then file_name
            title = score_node.get("title", "")
            # Extract doc_name - prefer doc_name, then file_name
            doc_name = score_node.get("title")
            # Extract file_path
            file_path = score_node.get("url", "")

            # Build the node in the required format
            # Include all metadata fields but structure the required ones at the top level
            node = NodeResult(
                score=score_node.get("score", 0),
                metadata=NodeMetadata(
                    file_path=file_path,
                    image_url=[img.get("url", "") for img in score_node.get("images", []) if img.get("url", "")],
                    title=title,
                    doc_name=doc_name,
                ),
                text=score_node.get("content", ""),
            )
            nodes.append(node)

        return RetrievalToolResponse(
            status="SUCCESS",
            status_code=200,
            data=RetrievalResult(
                total=len(nodes),
                nodes=nodes
            ),
            request_id=request_id
        )

    except Exception:
        logger.exception(f"Retrieval tool failed: {traceback.format_exc()}")
        return RetrievalToolResponse(
            status="ERROR",
            status_code=500,
            data=RetrievalResult(
                total=0,
                nodes=[]
            ),
            request_id=request_id
        )


def get_retrieval_tool(
    knowledgebase_id: str,
    kb_name: str,
    kb_description: str,
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
            image_list=image_list,
            user_id=user_id,
            retrieval_setting=retrieval_setting,
            metadata_condition=metadata_condition,
            rag_service=rag_service,
        )

    func_arg_metadata = func_metadata(
        asearch_knowledgebase_wrapper,
    skip_names=["user_id", "metadata_condition", "knowledgebase_id", "retrieval_setting", "session"],
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
