from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from common.chat.models import RetrievalSetting
from api.v1.mcp.kb_retriever_tool import asearch_knowledgebase
from common.chat.models import MetadataFilteringCondition
from service.knowledgebase.rag_service import RagService
from service.injection import get_rag_service, get_tenant_id
from loguru import logger
from extensions.trace.context import get_request_id

retrieval_tool_router = APIRouter()


class RetrievalToolRequest(BaseModel):
    query: str
    image_list: Optional[List[str]] = []
    user_id: Optional[str] = None
    retrieval_setting: Optional[RetrievalSetting] = None
    metadata_condition: Optional[MetadataFilteringCondition] = None


@retrieval_tool_router.post("/{knowledgebase_id}")
async def mcp_retrieval(
    knowledgebase_id: str,
    request: RetrievalToolRequest,
    tenant_id: str = Depends(get_tenant_id),
    rag_service: RagService = Depends(get_rag_service),
):
    """
    Retrieval tool interface with different input/output format.
    Input: {"query": "xxx", "images": ["1.jpg", "2.jpg"]}
    Output: {"status": "SUCCESS", "status_code": 200, "data": {"total": 2, "nodes": [...]}, "request_id": "..."}
    """
    request_id = get_request_id()
    logger.info(f"Retrieval tool request: {request}, knowledgebase_id: {knowledgebase_id}, tenant_id: {tenant_id}, request_id: {request_id}")

    result = await asearch_knowledgebase(
        query=request.query,
        image_list=request.image_list,
        knowledgebase_id=knowledgebase_id,
        retrieval_setting=request.retrieval_setting,
        metadata_condition=request.metadata_condition,
        rag_service=rag_service,
        tenant_id=tenant_id,
        request_id=request_id,
    )
    result_data = result.model_dump()
    logger.info(f"Retrieval tool response: {result_data}, request_id: {request_id}")
    return JSONResponse(status_code=result.status_code, content=result_data)
