from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException
from db.db_context import get_db_session
from common.chat.models import DocRecord, NewRetrievalResponse, RetrievalSetting, MetadataFilteringCondition
from sqlmodel.ext.asyncio.session import AsyncSession
from service.injection import get_rag_service, get_tenant_id, get_chatapp_service, get_faq_config_service
from service.knowledgebase.rag_service import RagService
from service.tool.chatapp_service import ChatappService
from service.tool.faq_config_service import FAQConfigService
from service.knowledgebase.knowledgebase_service import KnowledgebaseService
from common.knowledgebase.constants import FAQ_KNOWLEDGEBASE_NAME
from common.knowledgebase.types import VectorIndexRetrievalType
from common.tool.search_result import SearchResult
import traceback
from loguru import logger


faq_retrieval_router = APIRouter()


class FAQRetrievalRequest(BaseModel):
    chatapp_id: str  # ChatApp ID (can be chatbot.id or app_id)
    query: str  # 查询内容
    user_id: Optional[str] = None
    retrieval_setting: Optional[RetrievalSetting] = None
    metadata_condition: Optional[MetadataFilteringCondition] = None


@faq_retrieval_router.post(
    "", response_model=ResponseModel[NewRetrievalResponse]
)
async def faq_retrieval(
    retrieval_request: FAQRetrievalRequest,
    session: AsyncSession = Depends(get_db_session),
    tenant_id: str = Depends(get_tenant_id),
    rag_service: RagService = Depends(get_rag_service),
    chatapp_service: ChatappService = Depends(get_chatapp_service),
    faq_config_service: FAQConfigService = Depends(get_faq_config_service),
):
    logger.info(f"FAQ Retrieval request: chatapp_id={retrieval_request.chatapp_id}, query={retrieval_request.query}, tenant_id={tenant_id}")
    try:
        # Try to get chatbot by id first, if not found, try by app_id
        chatbot = await chatapp_service.get_chatapp(id=retrieval_request.chatapp_id, tenant_id=tenant_id)
        if not chatbot:
            chatbot = await chatapp_service.get_chatapp_by_app_id(
                app_id=retrieval_request.chatapp_id,
                tenant_id=tenant_id
            )
        if not chatbot:
            raise ApiException(code=404, message=f"应用 '{retrieval_request.chatapp_id}' 不存在。")

        # Get FAQ config to get similarity_threshold
        faq_config = await faq_config_service.get_faq_config_by_chatbot_id(
            chatbot_id=chatbot.id, tenant_id=tenant_id
        )

        # Get FAQ knowledgebase by name: {app_id}_{FAQ_KNOWLEDGEBASE_NAME}
        kb_name = f"{chatbot.app_id}_{FAQ_KNOWLEDGEBASE_NAME}"
        knowledgebase_service = KnowledgebaseService(session)
        kb = await knowledgebase_service.get_knowledgebase_by_name(kb_name, tenant_id=tenant_id)

        if not kb:
            raise ApiException(code=404, message=f"FAQ知识库 '{kb_name}' 不存在。")

        # Set default retrieval_setting if not provided, or merge with defaults
        default_similarity_threshold = faq_config.similarity_threshold if faq_config else 0.9

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

        # 使用统一的响应格式
        retrieval_response = NewRetrievalResponse(records=records)
        return success_response(data=retrieval_response, message="FAQ检索成功")
    except ApiException:
        raise
    except ValueError as e:
        logger.error(f"Failed to retrieve FAQ: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"FAQ检索失败: {e}")
    except Exception as e:
        logger.error(f"Failed to retrieve FAQ: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"FAQ检索失败: {e}")
