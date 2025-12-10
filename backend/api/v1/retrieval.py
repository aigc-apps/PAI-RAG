from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from common.chat.response_model import ResponseModel, to_dict
from api.api_exception import ApiException
from db.db_context import get_db_session
from common.chat.models import DocRecord, NewRetrievalResponse, RetrievalRequest
from sqlmodel.ext.asyncio.session import AsyncSession
from service.injection import get_rag_service, get_tenant_id
from service.knowledgebase.rag_service import RagService
from typing import List
from common.tool.search_result import SearchResult
import traceback
from loguru import logger


retrieval_router = APIRouter()

@retrieval_router.post(
    "", response_model=ResponseModel[NewRetrievalResponse]
)
async def retrieval(
    retrieval_request: RetrievalRequest,
    session: AsyncSession = Depends(get_db_session),
    tenant_id: str = Depends(get_tenant_id),
    rag_service: RagService = Depends(get_rag_service),
):
    try:
        search_results: List[SearchResult] = await rag_service.aquery(
            query=retrieval_request.query,
            user_id=retrieval_request.user_id,
            knowledge_id=retrieval_request.knowledge_id,
            retrieval_setting=retrieval_request.retrieval_setting,
            metadata_condition=retrieval_request.metadata_condition,
            tenant_id=tenant_id,
        )
        logger.info(
            f"Retrieved {len(search_results)} for query '{retrieval_request.query}' against knowledgebase {retrieval_request.knowledge_id}."
        )
        records = []
        for node in search_results:
            records.append(DocRecord(
                content=node.get("content", ""),
                score=node.get("score", 0),
                title=node.get("title", ""),
                metadata=node.get("metadata", {}),
            ))
        return JSONResponse(status_code=200, content={"records": to_dict(records)})
    except ValueError as e:
        logger.error(f"Failed to retrieve: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"Failed to retrieve: {e}")
    except Exception as e:
        logger.error(f"Failed to retrieve: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Failed to retrieve: {e}")
