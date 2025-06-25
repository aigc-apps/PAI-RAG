### MCP Configuration API ###

import asyncio
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models.mcp import McpServerRead, McpServerCreate, McpServerEntity
from pairag.db.db_context import get_session
from pairag.db.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from pairag.mcp.providers.mcp_tool_provider import mcp_provider

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
        asyncio.create_task(mcp_provider.refresh())
        return mcp
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add mcp: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            raise HTTPException(
                status_code=400, detail=f"Mcp name {mcp.name} already exists."
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Failed to add mcp config: {str(e)}"
            )
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=400, detail=f"Failed to add mcp config: {str(e)}"
        )


@mcp_router.get("", response_model=List[McpServerRead])
async def list_mcps(
    session: AsyncSession = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    mcp_results = await session.exec(
        select(McpServerEntity).offset(offset).limit(limit)
    )
    return mcp_results.all()


@mcp_router.get("/{mcp_id}", response_model=McpServerRead)
async def read_mcp(mcp_id: str, session: AsyncSession = Depends(get_session)):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        raise HTTPException(status_code=404, detail=f"MCP {mcp_id} not found.")

    return mcp


@mcp_router.patch("/{mcp_id}", response_model=McpServerRead)
async def update_mcp(
    mcp_id: str,
    update_mcp: McpServerCreate,
    session: AsyncSession = Depends(get_session),
):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        raise HTTPException(status_code=404, detail=f"MCP {mcp_id} not found.")

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

    asyncio.create_task(mcp_provider.refresh())

    logger.info(f"MCP {mcp_id} updated to {mcp}.")

    return mcp


@mcp_router.delete("/{mcp_id}")
async def delete_mcp(
    mcp_id: str,
    session: AsyncSession = Depends(get_session),
):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        raise HTTPException(status_code=404, detail=f"MCP {mcp_id} not found.")

    await session.delete(mcp)
    await session.commit()

    asyncio.create_task(mcp_provider.refresh())

    logger.info(f"MCP {mcp_id} has been deleted.")

    return {"message": f"MCP {mcp_id} has been deleted."}
