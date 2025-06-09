from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models import (
    LlmModelEntity,
    LlmModelCreate,
    LlmModelRead,
    McpServerEntity,
    McpServerCreate,
    McpServerRead,
    WebSearchConfigCreate,
    WebSearchConfigEntity,
    WebSearchConfigRead,
)
from pairag.db.db_context import db_context
from pairag.db.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from loguru import logger

config_router = APIRouter()


### Configuration API ###


@config_router.post("/llms/", response_model=LlmModelRead)
async def create_llm(
    llm_data: LlmModelCreate, session: AsyncSession = Depends(db_context.get_session)
):
    encrypted_api_key = encrypt_key(llm_data.api_key)
    llm = LlmModelEntity.model_validate(
        llm_data, update={"encrypted_api_key": encrypted_api_key}
    )
    session.add(llm)
    try:
        await session.commit()
        await session.refresh(llm)
        return llm
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add llm: {e.orig}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            raise HTTPException(
                status_code=400, detail=f"Model_id {llm_data.model_id} already exists."
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Failed to add llm config: {str(e)}"
            )
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=400, detail=f"Failed to add llm config: {str(e)}"
        )


@config_router.get("/llms/", response_model=List[LlmModelRead])
async def get_llms(
    session: AsyncSession = Depends(db_context.get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    llm_results = await session.exec(select(LlmModelEntity).offset(offset).limit(limit))
    return llm_results.all()


@config_router.get("/llms/{llm_id}", response_model=LlmModelRead)
async def read_llm(
    llm_id: int, session: AsyncSession = Depends(db_context.get_session)
):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {llm_id} not found.")

    return llm


@config_router.patch("/llms/{llm_id}", response_model=LlmModelRead)
async def update_llm(
    llm_id: int,
    update_llm: LlmModelCreate,
    session: AsyncSession = Depends(db_context.get_session),
):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {llm_id} not found.")

    llm.base_url = update_llm.base_url or llm.base_url
    llm.context_window = update_llm.context_window or llm.context_window
    llm.model = update_llm.model or llm.model
    llm.temperature = update_llm.temperature or llm.temperature
    llm.encrypted_api_key = (
        encrypt_key(update_llm.api_key) if update_llm.api_key else llm.encrypted_api_key
    )

    session.add(llm)
    await session.commit()
    await session.refresh(llm)

    return llm


@config_router.delete("/llms/{llm_id}")
async def delete_llm(
    llm_id: int,
    session: AsyncSession = Depends(db_context.get_session),
):
    llm = await session.get(LlmModelEntity, llm_id)
    if not llm:
        raise HTTPException(status_code=404, detail=f"LLM {llm_id} not found.")
    await session.delete(llm)
    await session.commit()
    return {"message": f"LLM {llm_id} deleted."}


# MCP CRUD


@config_router.post("/mcps/", response_model=McpServerRead)
async def create_mcp(
    mcp_data: McpServerCreate, session: AsyncSession = Depends(db_context.get_session)
):
    encrypted_auth_token = encrypt_key(mcp_data.auth_token)
    mcp = McpServerEntity.model_validate(
        mcp_data, update={"encrypted_auth_token": encrypted_auth_token}
    )
    session.add(mcp)
    try:
        await session.commit()
        await session.refresh(mcp)
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


@config_router.get("/mcps/", response_model=List[McpServerRead])
async def list_mcps(
    session: AsyncSession = Depends(db_context.get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    mcp_results = await session.exec(
        select(McpServerEntity).offset(offset).limit(limit)
    )
    return mcp_results.all()


@config_router.get("/mcps/{mcp_id}", response_model=McpServerRead)
async def read_mcp(
    mcp_id: int, session: AsyncSession = Depends(db_context.get_session)
):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        raise HTTPException(status_code=404, detail=f"MCP {mcp_id} not found.")

    return mcp


@config_router.patch("/mcps/{mcp_id}", response_model=McpServerRead)
async def update_mcp(
    mcp_id: int,
    update_mcp: McpServerCreate,
    session: AsyncSession = Depends(db_context.get_session),
):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        raise HTTPException(status_code=404, detail=f"MCP {mcp_id} not found.")

    mcp.name = update_mcp.name or mcp.name
    mcp.active = update_mcp.active
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

    logger.info(f"MCP {mcp_id} updated to {mcp}.")

    return mcp


@config_router.delete("/mcps/{mcp_id}")
async def delete_mcp(
    mcp_id: int,
    session: AsyncSession = Depends(db_context.get_session),
):
    mcp = await session.get(McpServerEntity, mcp_id)
    if not mcp:
        raise HTTPException(status_code=404, detail=f"MCP {mcp_id} not found.")

    await session.delete(mcp)
    await session.commit()

    logger.info(f"MCP {mcp_id} has been deleted.")

    return {"message": f"MCP {mcp_id} has been deleted."}


@config_router.post("/search_engines/", response_model=WebSearchConfigRead)
async def add_search_config(
    new_search_config: WebSearchConfigCreate,
    session: AsyncSession = Depends(db_context.get_session),
):
    encrypted_access_key_id = encrypt_key(new_search_config.access_key_id)
    encrypted_access_key_secret = encrypt_key(new_search_config.access_key_secret)

    statement = select(WebSearchConfigEntity).where(
        WebSearchConfigEntity.type == new_search_config.type
    )
    search_config = (await session.exec(statement)).first()
    if search_config is None:
        logger.info(f"Adding new search config for type {new_search_config.type}")

        search_config = WebSearchConfigEntity.model_validate(
            new_search_config,
            update={
                "encrypted_access_key_id": encrypted_access_key_id,
                "encrypted_access_key_secret": encrypted_access_key_secret,
            },
        )
    else:
        search_config.encrypted_access_key_id = encrypted_access_key_id
        search_config.encrypted_access_key_secret = encrypted_access_key_secret
        search_config.endpoint = new_search_config.endpoint or search_config.endpoint

    session.add(search_config)
    try:
        await session.commit()
        await session.refresh(search_config)
        return search_config
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add search config: {e.orig}")
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=400, detail=f"Failed to add search config: {str(e)}"
        )


@config_router.get("/search_engines/", response_model=List[WebSearchConfigRead])
async def list_search_config(
    session: AsyncSession = Depends(db_context.get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    search_config_results = await session.exec(
        select(WebSearchConfigEntity).offset(offset).limit(limit)
    )
    return search_config_results.all()
