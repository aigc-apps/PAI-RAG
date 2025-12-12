### Code sandbox configuration API ###

from typing import List
import traceback
from fastapi import APIRouter, Depends, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.code_sandbox import (
    CodeSandboxConfigRead,
    CodeSandboxConfigCreate,
    CodeSandboxConfigEntity,
)
from api.response_model import success_response, error_response, ResponseModel
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from config.providers.code_sandbox_provider import codesandbox_provider
from loguru import logger


code_sandbox_router = APIRouter()


@code_sandbox_router.post("", response_model=ResponseModel[CodeSandboxConfigRead])
async def add_code_sandbox_config(
    new_code_sandbox_config: CodeSandboxConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    if new_code_sandbox_config.type not in ["aliyun-fc"]:
        return error_response(code=400, message="不支持的code sandbox类型，仅支持aliyun-fc")


    aliyun_id = new_code_sandbox_config.aliyun_id
    interpreter_id = new_code_sandbox_config.interpreter_id
    interpreter_name = new_code_sandbox_config.interpreter_name
    type = new_code_sandbox_config.type
    enabled = new_code_sandbox_config.enabled


    statement = select(CodeSandboxConfigEntity)
    code_sandbox_config = (await session.exec(statement)).first()
    if code_sandbox_config is None:
        logger.info(f"Adding new code sandbox config for type {new_code_sandbox_config.type}")

        code_sandbox_config = CodeSandboxConfigEntity.model_validate(
            new_code_sandbox_config
        )
    else:
        logger.info("Updating code sandbox config")
        code_sandbox_config.aliyun_id = aliyun_id or code_sandbox_config.aliyun_id
        code_sandbox_config.interpreter_id = interpreter_id or code_sandbox_config.interpreter_id
        code_sandbox_config.interpreter_name = interpreter_name or code_sandbox_config.interpreter_name
        code_sandbox_config.type = type or code_sandbox_config.type
        code_sandbox_config.enabled = enabled


    session.add(code_sandbox_config)
    try:
        await session.commit()
        await session.refresh(code_sandbox_config)
        codesandbox_provider.update(code_sandbox_config)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.CODESANDBOX,
            source_id=code_sandbox_config.id,
            event_type=ChangeEventType.UPDATE,
        )

        return success_response(data=code_sandbox_config, message="添加代码沙盒配置成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add code sandbox config: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"Failed to add code sandbox config: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to add code sandbox config: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"Failed to add code sandbox config: {str(e)}")


@code_sandbox_router.get("", response_model=ResponseModel[List[CodeSandboxConfigRead]])
async def list_code_sandbox_config(
    session: AsyncSession = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    code_sandbox_config_result = (await session.exec(
        select(CodeSandboxConfigEntity).offset(offset).limit(limit)
    )).first()
    code_sandbox_config = None

    if code_sandbox_config_result:
        code_sandbox_config = CodeSandboxConfigRead(
            type=code_sandbox_config_result.type,
            aliyun_id=code_sandbox_config_result.aliyun_id,
            interpreter_id=code_sandbox_config_result.interpreter_id,
            interpreter_name=code_sandbox_config_result.interpreter_name,
            enabled=code_sandbox_config_result.enabled,
            id=code_sandbox_config_result.id,
        )
    else:
        logger.warning("No code sandbox config found.")

    return success_response(data=[code_sandbox_config], message="查询代码沙盒配置成功。")
