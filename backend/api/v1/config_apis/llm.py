import traceback
from typing import Optional
import os
import openai
from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.llm import LlmModelCreate, LlmModelRead, LlmModelEntity
from db.db_context import get_db_session
from common.encrypt_utils import encrypt_key
from common.chat.response_model import success_response, ResponseModel
from service.model.llm_service import LlmService
from service.injection import get_llm_service, get_tenant_id
from api.api_exception import ApiException
from loguru import logger
from common.llm.models import llm_url_to_model_provider_id_map, model_provider_map
from common.i18n import i18n

### LLM Configuration API ###
llm_router = APIRouter()



def try_get_initial_model_from_env():
    endpoint = os.environ.get("PAIRAG_RAG__LLM__endpoint", "").rstrip("/")
    if not endpoint:
        return None

    if not endpoint.endswith("/v1"):
        endpoint = endpoint.rstrip("/") + "/v1"

    token = os.environ.get("PAIRAG_RAG__LLM__token") or "abc"

    client = openai.OpenAI(api_key=token, base_url=endpoint)
    try:
        logger.info(f"Try to load models from {endpoint}:{token}.")
        models = client.models.list()
        if len(models.data) > 0:
            logger.info(f"Loaded default llm model {models.data[0].id}")
            return LlmModelEntity.model_validate({
                "base_url": endpoint,
                "encrypted_api_key": encrypt_key(token),
                "model": models.data[0].id,
                "provider_name": "openai_like",
                "model_name": models.data[0].id,
                "model_id": models.data[0].id,
                "source": "OpenAI-Compatible",
            })
    except Exception as ex:
        logger.warning(f"Load model list failed: {ex}")
        pass

    return None


@llm_router.post("", response_model=ResponseModel[LlmModelRead])
async def create_llm(
    llm_data: LlmModelCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    llm_service: LlmService = Depends(get_llm_service),
):
    logger.info(f"Creating LLM: {llm_data}.")
    try:
        llm_entity = await llm_service.create_llm(llm_data=llm_data, tenant_id=tenant_id)

        return success_response(data=llm_entity, message=i18n.t("api.llm.create_success"))
    except ValueError as e:
        logger.error(f"Failed to create llm: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.llm.create_failed", error=str(e)))
    except Exception as e:
        logger.error(f"Failed to create llm: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.llm.create_failed", error=str(e)))



@llm_router.get("/groups")
async def get_llm_groups(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    llm_service: LlmService = Depends(get_llm_service),
):
    logger.info("Getting LLM groups.")
    try:
        llm_entities = await llm_service.get_all_llms(tenant_id=tenant_id)

        if len(llm_entities) == 0:
            logger.info("trying to load default llms.")
            default_model = try_get_initial_model_from_env()

            if default_model:
                await llm_service.create_llm(default_model=default_model, tenant_id=tenant_id)
                llm_entities = [default_model]

        grouped_results = {}
        for llm in llm_entities:
            if not llm.model and not llm.model_name:
                continue

            model_provider_id = llm.provider_name or llm_url_to_model_provider_id_map.get(llm.base_url, "openai_like")
            if model_provider_id not in model_provider_map:
                continue
            provider_label = model_provider_map[model_provider_id].label

            if provider_label not in grouped_results:
                grouped_results[provider_label] = {
                    "id": len(grouped_results),
                    "label": provider_label,
                    "models": [],
                }
            grouped_results[provider_label]["models"].append(llm)

        return success_response(data={"groups": list(grouped_results.values())}, message=i18n.t("api.llm.groups_success"))
    except Exception as e:
        logger.error(f"Failed to get llm groups: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.llm.groups_failed", error=str(e)))


@llm_router.get("/providers")
async def get_llm_providers(
    vision_support: Optional[bool] = Query(default=None, description="过滤支持vision的多模态大模型"),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    llm_service: LlmService = Depends(get_llm_service),
):
    """Get distinct provider names for LLMs."""
    try:
        providers = await llm_service.get_provider_names(tenant_id=tenant_id, vision_support=vision_support)
        return success_response(data=providers, message=i18n.t("api.llm.providers_success"))
    except Exception as e:
        logger.error(f"Failed to get LLM providers: {traceback.format_exc()}")
        raise ApiException(code=400, message=i18n.t("api.llm.providers_failed", error=str(e)))


@llm_router.get("")
async def get_llms(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    provider_name: str = Query(default=None, description="过滤LLM模型提供商，None表示不过滤"),
    vision_support: Optional[bool] = Query(default=None, description="过滤支持vision的多模态大模型，None表示不过滤"),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    llm_service: LlmService = Depends(get_llm_service),
):
    logger.info(f"Getting LLMs with page: {page}, size: {size}, vision_support: {vision_support}.")
    try:
        llm_entities = await llm_service.list_llms(tenant_id=tenant_id, page=page, size=size, vision_support=vision_support, provider_name=provider_name)
        return success_response(data=llm_entities, message=i18n.t("api.llm.list_success"))
    except ApiException as e:
        logger.warning(f"Failed to get llms: {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to get llms: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.llm.list_failed", error=str(e)))


@llm_router.get("/{llm_id}", response_model=ResponseModel[LlmModelRead])
async def read_llm(
    llm_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    llm_service: LlmService = Depends(get_llm_service),
):
    logger.info(f"Getting LLM: {llm_id}.")
    try:
        llm_entity = await llm_service.get_llm(llm_id=llm_id, tenant_id=tenant_id)
        if not llm_entity:
            raise ApiException.not_found(llm_id, "LLM")

        llm_entity.provider_name = llm_entity.provider_name or llm_url_to_model_provider_id_map.get(llm_entity.base_url, "openai_like")
        if not llm_entity:
            raise ApiException.not_found(llm_id, "LLM")
        return success_response(data=llm_entity, message=i18n.t("api.llm.query_success"))
    except ApiException as e:
        logger.warning(f"Failed to get llm: {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to get llm: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.llm.query_failed", error=str(e)))


@llm_router.put("/{llm_id}", response_model=ResponseModel[LlmModelRead])
async def update_llm(
    llm_id: str,
    update_llm: LlmModelCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    llm_service: LlmService = Depends(get_llm_service),
):
    logger.info(f"Updating LLM: {llm_id} with data: {update_llm}.")
    try:
        if update_llm.provider_name is None:
            update_llm.provider_name = llm_url_to_model_provider_id_map.get(update_llm.base_url, "openai_like")
        llm_entity = await llm_service.update_llm(llm_id=llm_id, update_data=update_llm, tenant_id=tenant_id)
        return success_response(data=llm_entity, message=i18n.t("api.llm.update_success"))
    except Exception as e:
        logger.error(f"Failed to update llm: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.llm.update_failed", error=str(e)))


@llm_router.delete("/{llm_id}")
async def delete_llm(
    llm_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    llm_service: LlmService = Depends(get_llm_service),
):
    logger.info(f"Deleting LLM: {llm_id}.")
    try:
        await llm_service.delete_llm(llm_id=llm_id, tenant_id=tenant_id)
        return success_response(message=i18n.t("api.llm.delete_success"))
    except Exception as e:
        logger.error(f"Failed to delete llm: {traceback.format_exc()}")
        raise ApiException(code=500, message=i18n.t("api.llm.delete_failed", error=str(e)))
