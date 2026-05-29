from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException, handle_api_exceptions
from db.db_context import get_db_session
from common.chat.models import DocRecord, NewRetrievalResponse, RetrievalSetting, MetadataFilteringCondition
from sqlmodel.ext.asyncio.session import AsyncSession
from service.injection import get_rag_service, get_tenant_id, get_chatapp_service, get_faq_config_service
from service.knowledgebase.rag_service import RagService
from service.tool.chatapp_service import ChatappService
from service.tool.faq_config_service import FAQConfigService
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from common.knowledgebase.constants import DEFAULT_FAQ_SIMILARITY_THRESHOLD
from common.knowledgebase.types import VectorIndexRetrievalType
from common.tool.search_result import SearchResult
from loguru import logger


faq_retrieval_router = APIRouter()


class FAQRetrievalRequest(BaseModel):
    chatapp_id: str  # chatbot.id
    query: str  # query content
    user_id: Optional[str] = None
    retrieval_setting: Optional[RetrievalSetting] = None
    metadata_condition: Optional[MetadataFilteringCondition] = None


@faq_retrieval_router.post(
    "", response_model=ResponseModel[NewRetrievalResponse]
)
@handle_api_exceptions(action="retrieve FAQ")
async def faq_retrieval(
    retrieval_request: FAQRetrievalRequest,
    session: AsyncSession = Depends(get_db_session),
    tenant_id: str = Depends(get_tenant_id),
    rag_service: RagService = Depends(get_rag_service),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    faq_config_service: FAQConfigService = Depends(get_faq_config_service),
):
    logger.info(f"FAQ Retrieval request: chatapp_id={retrieval_request.chatapp_id}, query={retrieval_request.query}, tenant_id={tenant_id}")
    chatbot = await chatapp_service.get_chatapp_by_app_id(
        app_id=retrieval_request.chatapp_id,
        tenant_id=tenant_id
    )
    if not chatbot:
        raise ApiException(code=404, message=f"Application '{retrieval_request.chatapp_id}' does not exist.")

    # Get FAQ config to get similarity_threshold
    faq_config = chatbot.faq_config
    if not faq_config:
        raise ApiException(code=404, message=f"FAQ configuration '{retrieval_request.chatapp_id}' does not exist.")


    kb_id = faq_config.kb_id
    knowledgebase_service = KnowledgebaseService(session)
    kb = await knowledgebase_service.get_knowledgebase(kb_id, tenant_id=tenant_id)

    if not kb:
        raise ApiException(code=404, message=f"FAQ knowledge base '{kb_id}' does not exist.")

    # Set default retrieval_setting if not provided, or merge with defaults
    default_similarity_threshold = faq_config.similarity_threshold if faq_config else DEFAULT_FAQ_SIMILARITY_THRESHOLD

    if retrieval_request.retrieval_setting is None:
        retrieval_setting = RetrievalSetting(
            retrieval_mode=VectorIndexRetrievalType.vector,
            top_k=1,
            enable_rerank=False,
            rerank_top_k=None,
            vector_weight=1.0,
            similarity_threshold=default_similarity_threshold,
        )
    else:
        # Merge user-provided settings with defaults
        retrieval_setting = RetrievalSetting(
            retrieval_mode=retrieval_request.retrieval_setting.retrieval_mode or VectorIndexRetrievalType.vector,
            top_k=retrieval_request.retrieval_setting.top_k if retrieval_request.retrieval_setting.top_k is not None else 1,
            enable_rerank=retrieval_request.retrieval_setting.enable_rerank if retrieval_request.retrieval_setting.enable_rerank is not None else False,
            rerank_top_k=retrieval_request.retrieval_setting.rerank_top_k,
            rerank_model=retrieval_request.retrieval_setting.rerank_model,
            rerank_provider_name=retrieval_request.retrieval_setting.rerank_provider_name,
            vector_weight=retrieval_request.retrieval_setting.vector_weight if retrieval_request.retrieval_setting.vector_weight is not None else 1.0,
            similarity_threshold=retrieval_request.retrieval_setting.similarity_threshold if retrieval_request.retrieval_setting.similarity_threshold is not None else default_similarity_threshold,
            score_threshold=retrieval_request.retrieval_setting.score_threshold,
        )

    search_results: List[SearchResult] = await rag_service.aquery(
        query=retrieval_request.query,
        user_id=retrieval_request.user_id,
        kb_id=kb.id,
        kb_id_list=None,
        retrieval_setting=retrieval_setting,
        metadata_condition=None,
        tenant_id=tenant_id,
    )

    logger.info(
        f"Retrieved {len(search_results)} FAQ results for query '{retrieval_request.query}' from knowledgebase {kb.id}."
    )

    records = []
    for node in search_results:
        records.append(DocRecord(
            content=node.content,
            score=node.score,
            title=node.title,
            metadata=node.metadata,
        ))

    # Use unified response format
    retrieval_response = NewRetrievalResponse(records=records)
    return success_response(data=retrieval_response, message="FAQ retrieval succeeded.")
