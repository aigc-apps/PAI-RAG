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
from service.injection import get_guardrail_service
from api.api_exception import ApiException
from loguru import logger


guardrail_router = APIRouter()


@guardrail_router.post("", response_model=ResponseModel[GuardrailConfigRead])
async def add_guardrail_config(
    new_guardrail_config: GuardrailConfigCreate,
    session: AsyncSession = Depends(get_db_session),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
):
    logger.info(f"Adding guardrail config: {new_guardrail_config}.")
    try:
        guardrail_entity = await guardrail_service.create_guardrail(new_guardrail_config)
        return success_response(data=guardrail_entity, message="添加安全护栏配置成功.")
    except Exception as e:
        logger.error(f"Failed to add guardrail config: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"添加安全护栏配置失败: '{e}'.")


@guardrail_router.get("", response_model=ResponseModel[List[GuardrailConfigRead]])
async def list_guardrail_configs(
    session: AsyncSession = Depends(get_db_session),
    guardrail_service: GuardrailService = Depends(get_guardrail_service),
):
    logger.info("Listing guardrail configs.")
    try:
        guardrail_entities = await guardrail_service.get_all_guardrail_configs()
        return success_response(data=guardrail_entities, message="查询护栏配置成功。")
    except Exception as e:
        logger.error(f"Failed to list guardrail configs: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"查询护栏配置失败: '{e}'.")
