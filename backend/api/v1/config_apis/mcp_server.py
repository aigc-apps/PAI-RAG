### MCP Configuration API ###

from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from common.chat.response_model import PagedResult, ResponseModel, success_response
from db.models.mcp import McpServerRead, McpServerCreate
from db.db_context import get_db_session
from service.tool.mcpserver_service import McpserverService
from service.injection import get_mcpserver_service, get_tenant_id
from api.api_exception import ApiException, handle_api_exceptions
from common.i18n import i18n

mcp_router = APIRouter()


@mcp_router.post("", response_model=McpServerRead)
@handle_api_exceptions(action="create mcp", i18n_error_key="api.mcp.create_failed")
async def create_mcp(
    mcp_data: McpServerCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    mcp_entity = await mcp_server_service.create_mcpserver(mcp_data=mcp_data, tenant_id=tenant_id)
    return success_response(data=mcp_entity, message=i18n.t("api.mcp.create_success"))


@mcp_router.get("", response_model=ResponseModel[PagedResult])
@handle_api_exceptions(action="list mcps", i18n_error_key="api.mcp.query_failed")
async def list_mcps(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    if name:
        mcp_entity = await mcp_server_service.get_mcpserver_by_name(name=name, tenant_id=tenant_id)
        if not mcp_entity:
            raise ApiException.not_found(name, "MCP")
        return success_response(data=mcp_entity, message=i18n.t("api.mcp.query_success"))
    else:
        mcp_entities = await mcp_server_service.list_mcpservers(tenant_id=tenant_id, page=page, size=size)
        return success_response(data=mcp_entities, message=i18n.t("api.mcp.list_success"))


@mcp_router.get("/{mcp_id}", response_model=McpServerRead)
@handle_api_exceptions(action="read mcp", i18n_error_key="api.mcp.query_failed")
async def read_mcp(
    mcp_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    mcp_entity = await mcp_server_service.get_mcpserver(mcp_id=mcp_id, tenant_id=tenant_id)
    if not mcp_entity:
        raise ApiException.not_found(mcp_id, "MCP")
    return success_response(data=mcp_entity, message=i18n.t("api.mcp.query_success"))


@mcp_router.put("/{mcp_id}", response_model=McpServerRead)
@handle_api_exceptions(action="update mcp", i18n_error_key="api.mcp.update_failed")
async def update_mcp(
    mcp_id: str,
    update_mcp: McpServerCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    mcp_entity = await mcp_server_service.update_mcpserver(mcp_id=mcp_id, update_data=update_mcp, tenant_id=tenant_id)
    return success_response(data=mcp_entity, message=i18n.t("api.mcp.update_success"))



@mcp_router.delete("/{mcp_id}")
@handle_api_exceptions(action="delete mcp", i18n_error_key="api.mcp.delete_failed")
async def delete_mcp(
    mcp_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    await mcp_server_service.delete_mcpserver(mcp_id=mcp_id, tenant_id=tenant_id)
    return success_response(message=i18n.t("api.mcp.delete_success"))
