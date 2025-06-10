import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from openai import AsyncOpenAI
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.models import (
    LlmModelEntity,
    LlmModelCreate,
    LlmModelRead,
    McpServerEntity,
    McpServerCreate,
    McpServerRead,
    TraceConfig,
    TraceConfigEntity,
    WebSearchConfigCreate,
    WebSearchConfigEntity,
    WebSearchConfigRead,
)
from pairag.db.db_context import db_context
from pairag.db.encrypt_utils import decrypt_key, encrypt_key
from sqlalchemy.exc import IntegrityError
from loguru import logger
from openai.types.chat import ChatCompletionSystemMessageParam
from pairag.mcp.chat_mcp import handle_chat, process_mcp_tools
from pairag.mcp.prompts import SYSTEM_PROMPT
from pairag.mcp.utils.message_utils import convert_to_openai_messages
from pairag.mcp.utils.time_utils import get_prompt_current_time_str
from pairag.mcp.websearch.aliyun_search_tool import aget_aliyun_search_tool

config_router = APIRouter()


### Configuration API ###
llm_url_group_map = {
    "https://dashscope.aliyuncs.com/compatible-mode/v1": "通义千问",
    "https://api.openai.com/v1": "OpenAI",
}


@config_router.post("/llms", response_model=LlmModelRead)
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


@config_router.get("/llm_groups")
async def get_llm_groups(
    session: AsyncSession = Depends(db_context.get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    llm_results = await session.exec(select(LlmModelEntity).offset(offset).limit(limit))
    llms = llm_results.all()

    grouped_results = {}
    for llm in llms:
        if not llm.model:
            continue

        group_name = llm_url_group_map.get(llm.base_url, "OpenAI-Compatible")
        if group_name not in grouped_results:
            grouped_results[group_name] = {
                "id": len(grouped_results),
                "label": group_name,
                "models": [],
            }

        grouped_results[group_name]["models"].append(llm)

    return {"groups": list(grouped_results.values())}


@config_router.get("/llms", response_model=List[LlmModelRead])
async def get_llms(
    session: AsyncSession = Depends(db_context.get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    sql_results = await session.exec(select(LlmModelEntity).offset(offset).limit(limit))
    llm_entities = sql_results.all()
    llm_models = [
        LlmModelRead.model_validate(
            llm,
            update={"source": llm_url_group_map.get(llm.base_url, "OpenAI-Compatible")},
        )
        for llm in llm_entities
    ]

    return llm_models


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


@config_router.post("/mcps", response_model=McpServerRead)
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


@config_router.get("/mcps", response_model=List[McpServerRead])
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


@config_router.post("/websearch", response_model=WebSearchConfigRead)
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


@config_router.get("/websearch", response_model=List[WebSearchConfigRead])
async def list_search_config(
    session: AsyncSession = Depends(db_context.get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    search_config_results = await session.exec(
        select(WebSearchConfigEntity).offset(offset).limit(limit)
    )
    return search_config_results.all()


@config_router.post("/trace", response_model=TraceConfigEntity)
async def set_trace_config(
    new_trace_config: TraceConfig,
    session: AsyncSession = Depends(db_context.get_session),
):
    trace_config = (await session.exec(select(TraceConfigEntity))).first()
    if trace_config is None:
        logger.info(f"Adding new trace config {trace_config}")

        trace_config = WebSearchConfigEntity.model_validate(
            new_trace_config,
        )
    else:
        trace_config.endpoint = new_trace_config.endpoint or trace_config.endpoint
        trace_config.active = new_trace_config.active or trace_config.active
        trace_config.token = new_trace_config.token or trace_config.token
        trace_config.service_name = (
            new_trace_config.service_name or trace_config.service_name
        )

    session.add(trace_config)
    try:
        await session.commit()
        await session.refresh(trace_config)
        return trace_config
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add search config: {e.orig}")
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=400, detail=f"Failed to add search config: {str(e)}"
        )


@config_router.get("/trace", response_model=WebSearchConfigRead)
async def get_trace_config(
    session: AsyncSession = Depends(db_context.get_session),
):
    search_config_results = await session.exec(select(WebSearchConfigEntity))
    return search_config_results.first()


@config_router.post("/api/chat")
async def chat(
    request: Request, session: AsyncSession = Depends(db_context.get_session)
):
    try:
        # 解析请求体
        data = await request.json()
        messages = data.get("messages", [])
        system_prompt = SYSTEM_PROMPT.format(
            current_datetime=get_prompt_current_time_str()
        )
        system = data.get("system", system_prompt)

        # 从 headers 中获取模型参数
        model_id = request.headers.get("X-Model-Id")
        sql_result = await session.exec(
            select(LlmModelEntity).where(LlmModelEntity.model_id == model_id)
        )
        model_entity = sql_result.first()

        if model_entity is None:
            logger.error(f"Model id {model_id} not exists.")
            return None

        api_key = decrypt_key(model_entity.encrypted_api_key)
        model = AsyncOpenAI(
            api_key=api_key,
            base_url=model_entity.base_url,
        ).chat.completions
        logger.info(f"[Model] model_name: {model_entity.model}")

        x_options = (
            request.headers.get("X-Options").split(",")
            if request.headers.get("X-Options")
            else []
        )

        openai_tools = []
        tools_name_to_fn = {}
        if "search" in x_options:
            sql_result = await session.exec(select(WebSearchConfigEntity))
            search_entity = sql_result.first()

            if search_entity is None:
                logger.error("Search config not exists.")
                raise

            os.environ["WEBSEARCH_ACCESS_KEY_ID"] = decrypt_key(
                search_entity.encrypted_access_key_id
            )
            os.environ["WEBSEARCH_ACCESS_KEY_SECRET"] = decrypt_key(
                search_entity.encrypted_access_key_secret
            )
            (
                search_openai_tools,
                search_tools_name_to_fn,
            ) = await aget_aliyun_search_tool()
            openai_tools.extend(search_openai_tools)
            tools_name_to_fn.update(search_tools_name_to_fn)
        if "mcp" in x_options:
            mcp_openai_tools, mcp_tools_name_to_fn = await process_mcp_tools()
            openai_tools.extend(mcp_openai_tools)
            tools_name_to_fn.update(mcp_tools_name_to_fn)

        logger.info(f"[Tool] openai_tools: {openai_tools}")
        logger.info(f"[Tool] tools_name_to_fn: {tools_name_to_fn}")

        # 构建openai_messages
        full_messages = [
            ChatCompletionSystemMessageParam(role="system", content=system)
        ] + messages
        full_messages = convert_to_openai_messages(full_messages)

        return await handle_chat(
            model=model,
            model_name=model_entity.model,
            messages=full_messages,
            tools=openai_tools,
            tools_name_to_fn=tools_name_to_fn,
        )
    except Exception as e:
        logger.exception(f"Error in /api/chat: {str(e)}")
        return Response(content="Internal Server Error", status_code=500)
