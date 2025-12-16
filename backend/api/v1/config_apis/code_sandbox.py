### Code sandbox configuration API ###

from typing import List
import traceback
from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.code_sandbox import (
    CodeSandboxConfigRead,
    CodeSandboxConfigCreate,
)
from common.chat.response_model import success_response, ResponseModel
from db.db_context import get_db_session
from service.tool.codesandbox_service import CodesandboxService
from service.injection import get_codesandbox_service, get_tenant_id
from api.api_exception import ApiException
from loguru import logger


code_sandbox_router = APIRouter()


@code_sandbox_router.post("", response_model=ResponseModel[CodeSandboxConfigRead])
async def add_code_sandbox_config(
    new_code_sandbox_config: CodeSandboxConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    codesandbox_service: CodesandboxService = Depends(get_codesandbox_service),
):
    if new_code_sandbox_config.type.lower() not in ["aliyun-fc"]:
        logger.error(f"不支持的code sandbox类型{new_code_sandbox_config.type}，仅支持aliyun-fc")
        raise ApiException(code=400, message=f"不支持的code sandbox类型{new_code_sandbox_config.type}，仅支持aliyun-fc")



    try:
        code_sandbox_config = await codesandbox_service.create_or_update_codesandbox_config(
            new_code_sandbox_config, tenant_id=tenant_id
        )
        await session.refresh(code_sandbox_config)
        return success_response(data=code_sandbox_config, message="添加代码沙盒配置成功。")
    except ValueError as e:
        logger.error(f"Failed to add code sandbox config: {str(e)}")
        raise ApiException(code=400, message=f"添加代码沙盒配置失败: {e}")
    except Exception as e:
        logger.error(f"Failed to add code sandbox config: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"Failed to add code sandbox config: {str(e)}")


@code_sandbox_router.get("", response_model=ResponseModel[List[CodeSandboxConfigRead]])
async def list_code_sandbox_config(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    codesandbox_service: CodesandboxService = Depends(get_codesandbox_service),
):

    try:
        configs = await codesandbox_service.get_all_codesandbox_configs(tenant_id=tenant_id)
        if configs:
            code_sandbox_config_read = CodeSandboxConfigRead(
                type=configs[0].type,
                aliyun_id=configs[0].aliyun_id,
                interpreter_id=configs[0].interpreter_id,
                interpreter_name=configs[0].interpreter_name,
                enabled=configs[0].enabled,
                timeout_default=configs[0].timeout_default,
                id=configs[0].id,
            )
            return success_response(data=[code_sandbox_config_read], message="查询代码沙盒配置成功。")
        else:
            return success_response(data=[], message="查询代码沙盒配置成功。")
    except Exception as e:
        logger.error(f"Failed to list code sandbox config: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"查询代码沙盒配置失败: {str(e)}")
