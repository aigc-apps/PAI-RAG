from fastapi import APIRouter, Depends
from common.chat.response_model import ResponseModel
from api.api_exception import handle_api_exceptions
from common.chat.models import DocRecord, NewRetrievalResponse, RetrievalRequest
from service.injection import get_rag_service, get_tenant_id
from service.knowledgebase.rag_service import RagService
from typing import List
from common.tool.search_result import SearchResult
from fastapi.responses import JSONResponse
from extensions.trace.context import get_request_id
from loguru import logger


retrieval_router = APIRouter()

@retrieval_router.post(
    "", response_model=ResponseModel[NewRetrievalResponse]
)
@handle_api_exceptions(action="retrieve")
async def retrieval(
    retrieval_request: RetrievalRequest,
    tenant_id: str = Depends(get_tenant_id),
    rag_service: RagService = Depends(get_rag_service),
):
    request_id = get_request_id()
    logger.info(f"Retrieval request: {retrieval_request}, tenant_id: {tenant_id}, request_id: {request_id}")
    search_results: List[SearchResult] = await rag_service.aquery(
        query=retrieval_request.query,
        user_id=retrieval_request.user_id,
        kb_id=retrieval_request.knowledge_id,
        kb_id_list=retrieval_request.knowledge_id_list,
        retrieval_setting=retrieval_request.retrieval_setting,
        metadata_condition=retrieval_request.metadata_condition,
        tenant_id=tenant_id,
    )
    logger.info(
        f"Retrieved {len(search_results)} for query '{retrieval_request.query}', request_id: {request_id}"
    )
    records = []
    for node in search_results:
        records.append(DocRecord(
            content=node.content,
            score=node.score,
            title=node.title,
            url=node.url,
            metadata=node.metadata,
        ))
    # 使用统一的响应格式
    retrieval_response = NewRetrievalResponse(records=records)
    retrieval_response_data = retrieval_response.model_dump()
    logger.info(f"Retrieval response: {retrieval_response_data}, request_id: {request_id}")

    return JSONResponse(status_code=200, content=retrieval_response_data)
