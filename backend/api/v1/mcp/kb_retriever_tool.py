import uuid
from functools import partial
from typing import List, Optional
from loguru import logger
from common.chat.models import RetrievalSetting
from db.models.knowledgebase.metadata_filter import MetadataFilteringCondition
from tools.knowledgebase.knowledgebase_tool import kb_tool
from pydantic import BaseModel, Field
from mcp.server.fastmcp.tools import Tool
from mcp.server.fastmcp.utilities.func_metadata import func_metadata


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
) -> RetrievalToolResponse:
    request_id = str(uuid.uuid4())

    # TODO: Handle images if needed in the future
    # For now, images are accepted but not used in retrieval
    if image_list:
        logger.warning(f"Received {len(image_list)} images, but image-based retrieval is not yet implemented.")

    try:
        # Perform retrieval using the same strategy as retrieval.py
        node_results = await kb_tool.aquery(
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
            node_metadata = score_node.node.metadata.copy()

            # Extract image_url from metadata
            # Check for images_info first (from knowledgebase_tool processing)
            image_url = []
            if "images_info" in node_metadata and isinstance(node_metadata["images_info"], list):
                # Extract URLs from images_info list
                image_url = [img.get("url", "") for img in node_metadata["images_info"] if isinstance(img, dict) and "url" in img]
            elif "image_url" in node_metadata:
                # Fallback to direct image_url field
                image_url_value = node_metadata.get("image_url", [])
                if isinstance(image_url_value, list):
                    image_url = image_url_value
                elif image_url_value:
                    image_url = [image_url_value]

            # Extract title - prefer title, then file_name
            title = node_metadata.get("title") or node_metadata.get("file_name", "")
            # Extract doc_name - prefer doc_name, then file_name
            doc_name = node_metadata.get("doc_name") or node_metadata.get("file_name", "")
            # Extract file_path
            file_path = node_metadata.get("file_url", node_metadata.get("file_path", ""))

            # Build the node in the required format
            # Include all metadata fields but structure the required ones at the top level
            node = NodeResult(
                score=score_node.score,
                metadata=NodeMetadata(
                    file_path=file_path,
                    image_url=image_url,
                    title=title,
                    doc_name=doc_name,
                ),
                text=score_node.node.text,
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

    except Exception as e:
        logger.exception(f"Retrieval tool failed: {e}")
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

    func_arg_metadata = func_metadata(
        asearch_knowledgebase,
        skip_names=["user_id", "metadata_condition", "knowledgebase_id", "retrieval_setting"],
        structured_output=True,
    )
    parameters = func_arg_metadata.arg_model.model_json_schema(by_alias=True)

    fn = partial(asearch_knowledgebase, knowledgebase_id=knowledgebase_id)


    return Tool(
        fn=fn,
        name=func_name,
        description=func_description,
        fn_metadata=func_arg_metadata,
        parameters=parameters,
        is_async=True,
    )
