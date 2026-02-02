### Guardrail configuration API ###

import traceback
from typing import List
from common.chat.response_model import ResponseModel, success_response
from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.guardrail import (
    GuardrailConfigRead,
    GuardrailConfigCreate,
)
from db.db_context import get_db_session
from service.tool.guardrail_service import GuardrailService
from service.injection import get_guardrail_service, get_tenant_id
from api.api_exception import ApiException
from loguru import logger


guardrail_router = APIRouter()


@guardrail_router.post("", response_model=ResponseModel[GuardrailConfigRead])
async def add_guardrail_config(
    new_guardrail_config: GuardrailConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
):
    logger.info(f"Adding guardrail config: {new_guardrail_config}.")
    try:
        guardrail_entity = await guardrail_service.create_or_update_guardrail_config(config_data=new_guardrail_config, tenant_id=tenant_id)
        return success_response(data=guardrail_entity, message="Add guardrail config success.")
    except Exception as e:
        logger.error(f"Failed to add guardrail config: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"Add guardrail config failed: '{e}'.")


@guardrail_router.get("", response_model=ResponseModel[List[GuardrailConfigRead]])
async def list_guardrail_configs(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
):
    logger.info("Listing guardrail configs.")
    try:
        guardrail_entities = await guardrail_service.get_all_guardrail_configs(tenant_id=tenant_id)
        return success_response(data=guardrail_entities, message="List guardrail configs success.")
    except Exception as e:
        logger.error(f"Failed to list guardrail configs: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"List guardrail configs failed: '{e}'.")
