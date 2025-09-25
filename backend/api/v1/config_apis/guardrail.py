### Guardrail configuration API ###

import traceback
from typing import List
from api.response_model import ResponseModel, error_response, success_response
from fastapi import APIRouter, Depends, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.guardrail import (
    GuardrailConfigRead,
    GuardrailConfigCreate,
    GuardrailConfigEntity,
)
from db.db_context import get_session
from common.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from config.providers.guardrail_provider import guardrail_provider
from loguru import logger


guardrail_router = APIRouter()


@guardrail_router.post("", response_model=ResponseModel[GuardrailConfigRead])
async def add_guardrail_config(
    new_guardrail_config: GuardrailConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    encrypted_access_key_id = encrypt_key(new_guardrail_config.access_key_id)
    encrypted_access_key_secret = encrypt_key(new_guardrail_config.access_key_secret)

    guardrail_config = (await session.exec(select(GuardrailConfigEntity))).first()
    if guardrail_config is None:
        logger.info("Adding guardrail config.")

        guardrail_config = GuardrailConfigEntity.model_validate(
            new_guardrail_config,
            update={
                "encrypted_access_key_id": encrypted_access_key_id,
                "encrypted_access_key_secret": encrypted_access_key_secret,
            },
        )
    else:
        guardrail_config.encrypted_access_key_id = encrypted_access_key_id
        guardrail_config.encrypted_access_key_secret = encrypted_access_key_secret
        guardrail_config.endpoint = new_guardrail_config.endpoint or guardrail_config.endpoint
        guardrail_config.region_id = new_guardrail_config.region_id or guardrail_config.region_id
        guardrail_config.region_name = new_guardrail_config.region_name or guardrail_config.region_name

    session.add(guardrail_config)
    try:
        await session.commit()
        await session.refresh(guardrail_config)
        guardrail_provider.update(guardrail_config)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.GUARDRAIL,
            source_id=guardrail_config.id,
            event_type=ChangeEventType.UPDATE,
        )

        return success_response(data=guardrail_config, message="保存安全护栏配置成功.")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add guardrail config: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"添加护栏配置失败: {e}")
    except Exception as e:
        logger.error(f"Failed to add guardrail config: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=500, message=f"添加护栏配置失败: {e}")


@guardrail_router.get("", response_model=ResponseModel[List[GuardrailConfigRead]])
async def list_search_config(
    session: AsyncSession = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    guardrail_config_results = await session.exec(
        select(GuardrailConfigEntity).offset(offset).limit(limit)
    )
    return success_response(data=guardrail_config_results.all(), message="查询护栏配置成功。")
