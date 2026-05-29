### Code sandbox configuration API ###

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
from api.api_exception import ApiException, handle_api_exceptions
from common.encrypt_utils import decrypt_key
from loguru import logger
from common.i18n import i18n


code_sandbox_router = APIRouter()


@code_sandbox_router.post("", response_model=ResponseModel[CodeSandboxConfigRead])
@handle_api_exceptions(action="add code sandbox config", i18n_error_key="api.code_sandbox.add_failed", default_code=400)
async def add_code_sandbox_config(
    new_code_sandbox_config: CodeSandboxConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    codesandbox_service: CodesandboxService = Depends(get_codesandbox_service),
):
    if new_code_sandbox_config.type.lower() not in ["aliyun-fc"]:
        logger.error(f"不支持的code sandbox类型{new_code_sandbox_config.type}，仅支持aliyun-fc")
        raise ApiException(code=400, message=i18n.t("api.code_sandbox.unsupported_type", type=new_code_sandbox_config.type))

    code_sandbox_config = await codesandbox_service.create_or_update_codesandbox_config(
        new_code_sandbox_config, tenant_id=tenant_id
    )
    await session.commit()
    await session.refresh(code_sandbox_config)
    return success_response(data=code_sandbox_config, message=i18n.t("api.code_sandbox.add_success"))


@code_sandbox_router.get("", response_model=ResponseModel[CodeSandboxConfigRead])
@handle_api_exceptions(action="list code sandbox config", i18n_error_key="api.code_sandbox.query_failed", default_code=400)
async def list_code_sandbox_config(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    codesandbox_service: CodesandboxService = Depends(get_codesandbox_service),
):
    config = await codesandbox_service.get_codesandbox_config_or_create(tenant_id=tenant_id)
    if not config:
        logger.warning("No code sandbox config found.")
        return success_response(data=CodeSandboxConfigRead(), message=i18n.t("api.code_sandbox.query_success"))
    code_sandbox_config_read = CodeSandboxConfigRead(
        type=config.type,
        aliyun_id=config.aliyun_id,
        interpreter_id=config.interpreter_id,
        interpreter_name=config.interpreter_name,
        enabled=config.enabled,
        timeout_default=config.timeout_default,
        api_key=decrypt_key(config.encrypted_api_key) if config.encrypted_api_key else None,
        id=config.id,
    )
    return success_response(data=code_sandbox_config_read, message=i18n.t("api.code_sandbox.query_success"))
