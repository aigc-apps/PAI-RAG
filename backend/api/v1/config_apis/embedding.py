### Embedding configuration API ###

from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.embedding import (
    EmbeddingModelCreate,
    EmbeddingModelRead,
    EmbeddingType,
)
from db.db_context import get_db_session
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException, handle_api_exceptions
from service.model.embedding_service import EmbeddingService
from service.injection import get_embedding_service, get_tenant_id
from loguru import logger
from common.i18n import i18n

embedding_router = APIRouter()


@embedding_router.post("", response_model=ResponseModel[EmbeddingModelRead])
@handle_api_exceptions(
    action="create embedding",
    i18n_error_key="api.embedding.create_failed",
    default_code=400,
)
async def create_embedding(
    embedding_create: EmbeddingModelCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    embedding = await embedding_service.create_embedding(embedding_create, tenant_id=tenant_id)
    await session.refresh(embedding)

    if embedding.type == EmbeddingType.LOCAL:
        import app.worker as background_worker
        background_worker.download_model.delay(id=embedding.id, model_name=embedding.model_name)
    return success_response(data=embedding, message=i18n.t("api.embedding.create_success"))


@embedding_router.get("/providers")
@handle_api_exceptions(
    action="get embedding providers",
    i18n_error_key="api.embedding.providers_failed",
    default_code=400,
)
async def get_embedding_providers(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    """Get distinct provider names for embeddings."""
    providers = await embedding_service.get_provider_names(tenant_id=tenant_id)
    return success_response(data=providers, message=i18n.t("api.embedding.providers_success"))


@embedding_router.get("")
@handle_api_exceptions(
    action="get embeddings",
    i18n_error_key="api.embedding.list_failed",
    default_code=400,
)
async def get_embeddings(
    model_name: str = None,
    provider_name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
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


@embedding_router.put("/{emb_id}", response_model=ResponseModel[EmbeddingModelRead])
@handle_api_exceptions(
    action="update embedding",
    i18n_error_key="api.embedding.update_failed",
    default_code=400,
)
async def update_embedding(
    emb_id: str,
    update_data: EmbeddingModelCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    embedding_model = await embedding_service.update_embedding(emb_id=emb_id, update_data=update_data, tenant_id=tenant_id)
    await session.refresh(embedding_model)
    logger.info(f"Embedding {emb_id} updated to {embedding_model}.")
    return success_response(data=embedding_model, message=i18n.t("api.embedding.update_success"))


@embedding_router.delete("/{emb_id}")
@handle_api_exceptions(
    action="delete embedding",
    i18n_error_key="api.embedding.delete_failed",
    default_code=400,
)
async def delete_embedding(
    emb_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    await embedding_service.delete_embedding(emb_id=emb_id, tenant_id=tenant_id)
    logger.info(f"Embedding {emb_id} deleted.")
    return success_response(message=i18n.t("api.embedding.delete_success", id=emb_id))
