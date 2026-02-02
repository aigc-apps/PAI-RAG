### Reranker configuration API ###

from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.knowledgebase.reranker import (
    RerankerModelCreate,
    RerankerModelRead,
)
from db.db_context import get_db_session
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException
from service.model.reranker_service import RerankerService
from service.injection import get_reranker_service, get_tenant_id
import traceback
from loguru import logger
from common.i18n import i18n

reranker_router = APIRouter()


@reranker_router.post("", response_model=ResponseModel[RerankerModelRead])
async def create_reranker(
    reranker_data: RerankerModelCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    try:
        reranker = await reranker_service.create_reranker(reranker_data=reranker_data, tenant_id=tenant_id)
        return success_response(data=reranker, message=i18n.t("api.reranker.create_success"))
    except ValueError as e:
        logger.error(f"Failed to create reranker model: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.reranker.create_failed", error=str(e)))
    except Exception as e:
        logger.error(f"Failed to create reranker model: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.reranker.create_failed", error=str(e)))


@reranker_router.get("/providers")
async def get_reranker_providers(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    """Get distinct provider names for rerankers."""
    try:
        providers = await reranker_service.get_provider_names(tenant_id=tenant_id)
        return success_response(data=providers, message=i18n.t("api.reranker.providers_success"))
    except Exception as e:
        logger.error(f"Failed to get reranker providers: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.reranker.providers_failed", error=str(e)))


@reranker_router.get("")
async def get_rerankers(
    model_name: str = None,
    provider_name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    try:
        if not model_name:
            reranker_models = await reranker_service.list_rerankers(tenant_id=tenant_id, page=page, size=size, provider_name=provider_name)
            return success_response(data=reranker_models, message=i18n.t("api.reranker.list_success"))
        else:
            reranker_model = await reranker_service.get_reranker_by_model_name(model_name=model_name, tenant_id=tenant_id)
            return success_response(data=reranker_model, message=i18n.t("api.reranker.query_success"))
    except ValueError as e:
        logger.error(f"Failed to get reranker model: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.reranker.query_failed", error=str(e)))
    except Exception as e:
        logger.error(f"Failed to get reranker model: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.reranker.query_failed", error=str(e)))


@reranker_router.put("/{reranker_id}", response_model=ResponseModel[RerankerModelRead])
async def update_reranker(
    reranker_id: str,
    new_reranker: RerankerModelCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    try:
        reranker_model = await reranker_service.update_reranker(reranker_id=reranker_id, update_data=new_reranker, tenant_id=tenant_id)
        return success_response(data=reranker_model, message=i18n.t("api.reranker.update_success"))
    except ValueError as e:
        logger.error(f"Failed to update reranker model: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.reranker.update_failed", error=str(e)))
    except Exception as e:
        logger.error(f"Failed to update reranker model: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.reranker.update_failed", error=str(e)))


@reranker_router.delete("/{reranker_id}")
async def delete_reranker(
    reranker_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    reranker_service: RerankerService = Depends(get_reranker_service)
):
    try:
        await reranker_service.delete_reranker(reranker_id=reranker_id, tenant_id=tenant_id)
        return success_response(message=i18n.t("api.reranker.delete_success"))
    except ValueError as e:
        logger.error(f"Failed to delete reranker model: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.reranker.delete_failed", error=str(e)))
    except Exception as e:
        logger.error(f"Failed to delete reranker model: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.reranker.delete_failed", error=str(e)))
