### Embedding configuration API ###

import traceback
from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.embedding import (
    EmbeddingModelCreate,
    EmbeddingModelRead,
    EmbeddingType,
)
from db.db_context import get_db_session
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException
from service.model.embedding_service import EmbeddingService
from service.injection import get_embedding_service, get_tenant_id
from loguru import logger
from common.i18n import i18n

embedding_router = APIRouter()


@embedding_router.post("", response_model=ResponseModel[EmbeddingModelRead])
async def create_embedding(
    embedding_create: EmbeddingModelCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    try:
        embedding = await embedding_service.create_embedding(embedding_create, tenant_id=tenant_id)
        await session.refresh(embedding)

        if embedding.type == EmbeddingType.LOCAL:
            import app.worker as background_worker
            background_worker.download_model.delay(id=embedding.id, model_name=embedding.model_name)
        return success_response(data=embedding, message=i18n.t("api.embedding.create_success"))
    except ValueError as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to create embedding: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.embedding.create_failed", error=str(e)))


@embedding_router.get("/providers")
async def get_embedding_providers(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Get distinct provider names for embeddings."""
    try:
        providers = await embedding_service.get_provider_names(tenant_id=tenant_id)
        return success_response(data=providers, message=i18n.t("api.embedding.providers_success"))
    except Exception as e:
        logger.error(f"Failed to get embedding providers: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.embedding.providers_failed", error=str(e)))


@embedding_router.get("")
async def get_embeddings(
    model_name: str = None,
    provider_name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    try:
        if not model_name:
            embedding_models = await embedding_service.list_embeddings(tenant_id=tenant_id, page=page, size=size, provider_name=provider_name)
            return success_response(
                data=embedding_models,
                message=i18n.t("api.embedding.list_success")
            )
        else:
            embedding_model = await embedding_service.get_embedding_by_model_name(model_name=model_name, tenant_id=tenant_id)
            if not embedding_model:
                raise ApiException(
                    code=404, message=i18n.t("api.embedding.query_failed", model=model_name)
                )
            return success_response(data=embedding_model, message=i18n.t("api.embedding.query_success"))
    except Exception as e:
        logger.error(f"Failed to get embeddings: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.embedding.list_failed", error=str(e)))


@embedding_router.put("/{emb_id}", response_model=ResponseModel[EmbeddingModelRead])
async def update_embedding(
    emb_id: str,
    update_data: EmbeddingModelCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    try:
        embedding_model = await embedding_service.update_embedding(emb_id=emb_id, update_data=update_data, tenant_id=tenant_id)
        await session.refresh(embedding_model)
        logger.info(f"Embedding {emb_id} updated to {embedding_model}.")
        return success_response(data=embedding_model, message=i18n.t("api.embedding.update_success"))
    except ValueError as e:
        logger.error(f"Failed to update embedding: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to update embedding: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.embedding.update_failed", error=str(e)))


@embedding_router.delete("/{emb_id}")
async def delete_embedding(
    emb_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    try:
        await embedding_service.delete_embedding(emb_id=emb_id, tenant_id=tenant_id)
        logger.info(f"Embedding {emb_id} deleted.")
        return success_response(message=i18n.t("api.embedding.delete_success", id=emb_id))
    except ValueError as e:
        logger.error(f"Failed to delete embedding: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete embedding: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.embedding.delete_failed", error=str(e)))
