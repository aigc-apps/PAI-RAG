### MCP Configuration API ###

import traceback
from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession
from common.chat.response_model import PagedResult, ResponseModel, success_response
from db.models.mcp import McpServerRead, McpServerCreate
from db.db_context import get_db_session
from service.tool.mcpserver_service import McpserverService
from service.injection import get_mcpserver_service
from api.api_exception import ApiException
from loguru import logger

mcp_router = APIRouter()


@mcp_router.post("", response_model=McpServerRead)
async def create_mcp(
    mcp_data: McpServerCreate,
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    try:
        mcp_entity = await mcp_server_service.create_mcpserver(mcp_data)
        return success_response(data=mcp_entity, message="创建MCP配置成功。")
    except ValueError as e:
        logger.error(f"Failed to create mcp: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"创建MCP配置失败: '{e}'.")
    except Exception as e:
        logger.error(f"Failed to create mcp: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"创建MCP配置失败: '{e}'.")

@mcp_router.get("", response_model=ResponseModel[PagedResult])
async def list_mcps(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    try:
        if name:
            mcp_entity = await mcp_server_service.get_mcpserver_by_name(name=name)
            if not mcp_entity:
                raise ApiException.not_found(name, "MCP")
            return success_response(data=mcp_entity, message="查询MCP配置成功。")
        else:
            mcp_entities = await mcp_server_service.list_mcpservers(page=page, size=size)
            return success_response(data=mcp_entities, message="查询MCP配置列表成功。")
    except Exception as e:
        logger.error(f"Failed to list mcps: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"查询MCP配置失败: '{e}'.")


@mcp_router.get("/{mcp_id}", response_model=McpServerRead)
async def read_mcp(
    mcp_id: str,
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    try:
        mcp_entity = await mcp_server_service.get_mcpserver(mcp_id=mcp_id)
        if not mcp_entity:
            raise ApiException.not_found(mcp_id, "MCP")
        return success_response(data=mcp_entity, message="查询MCP配置成功。")
    except Exception as e:
        logger.error(f"Failed to read mcp: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"查询MCP配置失败: '{e}'.")


@mcp_router.put("/{mcp_id}", response_model=McpServerRead)
async def update_mcp(
    mcp_id: str,
    update_mcp: McpServerCreate,
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    try:
        mcp_entity = await mcp_server_service.update_mcpserver(mcp_id=mcp_id, update_data=update_mcp)
        return success_response(data=mcp_entity, message="更新MCP配置成功。")
    except ValueError as e:
        logger.error(f"Failed to update mcp: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"更新MCP配置失败: '{e}'.")
    except Exception as e:
        logger.error(f"Failed to update mcp: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"更新MCP配置失败: '{e}'.")



@mcp_router.delete("/{mcp_id}")
async def delete_mcp(
    mcp_id: str,
    session: AsyncSession = Depends(get_db_session),
    mcp_server_service: McpserverService = Depends(get_mcpserver_service)
):
    try:
        await mcp_server_service.delete_mcpserver(mcp_id=mcp_id)
        return success_response(message="删除MCP配置成功。")
    except ValueError as e:
        logger.error(f"Failed to delete mcp: {traceback.format_exc()}")
        raise ApiException(code=400, message=f"删除MCP配置失败: '{e}'.")
    except Exception as e:
        logger.error(f"Failed to delete mcp: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"删除MCP配置失败: '{e}'.")
