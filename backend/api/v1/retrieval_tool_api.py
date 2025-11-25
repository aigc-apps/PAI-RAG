from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from sqlmodel.ext.asyncio.session import AsyncSession
from db.db_context import get_session
from common.chat.models import RetrievalSetting
from api.v1.mcp.kb_retriever_tool import asearch_knowledgebase
from db.models.knowledgebase.metadata_filter import MetadataFilteringCondition

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
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieval tool interface with different input/output format.
    Input: {"query": "xxx", "images": ["1.jpg", "2.jpg"]}
    Output: {"status": "SUCCESS", "status_code": 200, "data": {"total": 2, "nodes": [...]}, "request_id": "..."}
    """

    result = await asearch_knowledgebase(
        query=request.query,
        image_list=request.image_list,
        knowledgebase_id=knowledgebase_id,
        retrieval_setting=request.retrieval_setting,
        metadata_condition=request.metadata_condition,
    )
    return JSONResponse(status_code=result.status_code, content=result.model_dump())
