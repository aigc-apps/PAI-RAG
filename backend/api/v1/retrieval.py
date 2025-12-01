from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from api.response_model import ResponseModel, error_response, to_dict
from db.db_context import get_session
from db.models.knowledgebase.knowledgebase import KbEntity
from common.chat.models import DocRecord, NewRetrievalResponse, RetrievalRequest
from sqlmodel.ext.asyncio.session import AsyncSession
from tools.knowledgebase.knowledgebase_tool import kb_tool
from loguru import logger


retrieval_router = APIRouter()

@retrieval_router.post(
    "", response_model=ResponseModel[NewRetrievalResponse]
)
async def retrieval(
    retrieval_request: RetrievalRequest, session: AsyncSession = Depends(get_session)
):
    knowledgebase = await session.get(KbEntity, retrieval_request.knowledge_id)
    if knowledgebase is None:
        return error_response(
            code=404, message=f"找不到知识库{retrieval_request.knowledge_id}"
        )
    try:
        node_results = await kb_tool.aquery(
            query=retrieval_request.query,
            user_id=retrieval_request.user_id,
            knowledge_id=retrieval_request.knowledge_id,
            retrieval_setting=retrieval_request.retrieval_setting,
            metadata_condition=retrieval_request.metadata_condition,
        )
        logger.info(
            f"Retrieved {len(node_results)} for query '{retrieval_request.query}' against knowledgebase {retrieval_request.knowledge_id}."
        )
        records = []
        for score_node in node_results:
            records.append(DocRecord(
                content=score_node.node.get_content(),
                score=score_node.score,
                title=score_node.node.metadata.get("file_name", "null"),
                metadata=score_node.node.metadata,
            ))
        return JSONResponse(status_code=200, content={"records": to_dict(records)})
    except Exception as e:
        logger.error(f"Failed to retrieve: {e}")
        return error_response(
            code=500, message=f"Failed to retrieve: {e}"
        )
