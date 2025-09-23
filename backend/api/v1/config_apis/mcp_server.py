### MCP Configuration API ###

import traceback
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from api.response_model import PagedResult, ResponseModel, success_response, error_response
from api.v1.utils.paginate import get_pagination_meta
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.mcp import McpServerRead, McpServerCreate, McpServerEntity
from db.db_context import get_session
from common.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from config.providers.mcp_tool_provider import mcp_provider

from loguru import logger

mcp_router = APIRouter()


@mcp_router.post("", response_model=McpServerRead)
async def create_mcp(
    mcp_data: McpServerCreate, session: AsyncSession = Depends(get_session)
):
    encrypted_auth_token = None
    if mcp_data.auth_token:
        encrypted_auth_token = encrypt_key(mcp_data.auth_token)
    mcp = McpServerEntity.model_validate(
        mcp_data, update={"encrypted_auth_token": encrypted_auth_token}
    )
    session.add(mcp)
    try:
        await session.commit()
        await session.refresh(mcp)
        mcp_provider.add(mcp)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.MCP,
            source_id=mcp.id,
            event_type=ChangeEventType.ADD,
        )
        return success_response(data=mcp, message="创建MCP配置成功.")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add mcp: {traceback.format_exc()}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return error_response(code=400, message="创建MCP配置失败: mcp已存在.")

        return error_response(code=400, message=f"创建MCP配置失败: '{e}'.")
    except Exception as e:
        logger.error(f"Exception occurred when add mcp: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"创建MCP配置失败: '{e}'.")


@mcp_router.get("", response_model=ResponseModel[PagedResult])
async def list_mcps(
    name: str = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    if not name:
        total_results = await session.exec(
            select(func.count()).select_from(
                select(McpServerEntity)
            )
        )
        total_num = total_results.one_or_none()
        pagination = get_pagination_meta(page, size, total_num)
        sql_results = await session.exec(select(McpServerEntity).offset(pagination.offset).limit(size))
        mcp_entities = sql_results.all()
        mcp_models = [
            McpServerRead.model_validate(entity)
            for entity in mcp_entities
        ]

        return success_response(
            data=PagedResult(
                items=mcp_models,
                total=pagination.total,
                pages=pagination.pages,
                page=pagination.page,
                size=pagination.size,
            ),message="查询mcp配置列表成功")
    else:
        statement = select(McpServerEntity).where(
            McpServerEntity.name == name
        )
        mcp_model = (await session.exec(statement)).first()
        if not mcp_model:
            return error_response(
                    code=404, message=f"查询MCP配置失败: MCP '{name}'不存在。"
                )

        return success_response(data=mcp_model, message="查询MCP配置成功。")



@mcp_router.get("/{mcp_id}", response_model=McpServerRead)
async def read_mcp(mcp_id: str, session: AsyncSession = Depends(get_session)):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        return error_response(code=404, message=f"MCP {mcp_id} not found.")

    return success_response(data=mcp, message="查询mcp成功。")


@mcp_router.put("/{mcp_id}", response_model=McpServerRead)
async def update_mcp(
    mcp_id: str,
    update_mcp: McpServerCreate,
    session: AsyncSession = Depends(get_session),
):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        return error_response(code=404, message=f"MCP {mcp_id} not found.")

    mcp.name = update_mcp.name or mcp.name
    mcp.enabled = update_mcp.enabled
    mcp.encrypted_auth_token = (
        encrypt_key(update_mcp.auth_token)
        if update_mcp.auth_token
        else mcp.encrypted_auth_token
    )
    mcp.type = update_mcp.type or mcp.type
    mcp.url = update_mcp.url or mcp.url

    session.add(mcp)
    await session.commit()
    await session.refresh(mcp)

    mcp_provider.update(mcp)
    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.MCP,
        source_id=mcp.id,
        event_type=ChangeEventType.UPDATE,
    )

    logger.info(f"MCP {mcp_id} updated to {mcp}.")

    return success_response(data=mcp, message="更新mcp成功。")


@mcp_router.delete("/{mcp_id}")
async def delete_mcp(
    mcp_id: str,
    session: AsyncSession = Depends(get_session),
):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        return error_response(code=404, message=f"MCP {mcp_id} not found.")

    await session.delete(mcp)
    await session.commit()

    mcp_provider.delete(mcp_id)
    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.MCP,
        source_id=mcp_id,
        event_type=ChangeEventType.DELETE,
    )

    logger.info(f"MCP {mcp_id} has been deleted.")

    return success_response(data=mcp, message="删除mcp成功。")
