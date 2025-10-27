### Web search configuration API ###

from typing import List
from fastapi import APIRouter, Depends, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.websearch import (
    WebSearchConfigRead,
    WebSearchConfigCreate,
    WebSearchConfigEntity,
)
from api.response_model import ResponseModel, error_response, success_response
from db.db_context import get_session
from common.encrypt_utils import encrypt_key
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from config.providers.websearch_provider import websearch_provider
from loguru import logger


websearch_router = APIRouter()


@websearch_router.post("", response_model=ResponseModel[WebSearchConfigRead])
async def add_search_config(
    new_search_config: WebSearchConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    if new_search_config.type not in ["tavily", "aliyun"]:
        return error_response(code=400, message="不支持的搜索引擎类型，仅支持tavily和aliyun")

    encrypted_access_key_id = encrypt_key(new_search_config.access_key_id)
    encrypted_access_key_secret = encrypt_key(new_search_config.access_key_secret)
    encrypted_tavily_api_key = encrypt_key(new_search_config.tavily_api_key)

    statement = select(WebSearchConfigEntity)
    search_config = (await session.exec(statement)).first()
    if search_config is None:
        logger.info(f"Adding new search config for type {new_search_config.type}")

        search_config = WebSearchConfigEntity.model_validate(
            new_search_config,
            update={
                "encrypted_access_key_id": encrypted_access_key_id,
                "encrypted_access_key_secret": encrypted_access_key_secret,
                "encrypted_tavily_api_key": encrypted_tavily_api_key,
            },
        )
    else:
        search_config.encrypted_access_key_id = encrypted_access_key_id or search_config.encrypted_access_key_id
        search_config.encrypted_access_key_secret = encrypted_access_key_secret or search_config.encrypted_access_key_secret
        search_config.encrypted_tavily_api_key = encrypted_tavily_api_key or search_config.encrypted_tavily_api_key
        search_config.search_count = new_search_config.search_count
        search_config.endpoint = new_search_config.endpoint or search_config.endpoint
        search_config.type = new_search_config.type

    session.add(search_config)
    try:
        await session.commit()
        await session.refresh(search_config)
        websearch_provider.update(search_config)
        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.WEBSEARCH,
            source_id=search_config.id,
            event_type=ChangeEventType.UPDATE,
        )

        return success_response(data=search_config, message="更新搜索配置成功。")
    except IntegrityError as e:
        logger.error(f"IntegrityError occurred when add search config: {e.orig}")
        await session.rollback()
        raise error_response(
            status_code=400, detail=f"Failed to add search config: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to add search config: {str(e)}")
        await session.rollback()
        raise error_response(
            status_code=400, detail=f"Failed to add search config: {str(e)}"
        )


@websearch_router.get("", response_model=ResponseModel[List[WebSearchConfigRead]])
async def list_search_config(
    session: AsyncSession = Depends(get_session),
    offset: int = 0,
    limit: int = Query(default=10, lte=1000),
):
    search_config_result = (await session.exec(
        select(WebSearchConfigEntity).offset(offset).limit(limit)
    )).first()

    if not search_config_result:
        return success_response(
            data=[WebSearchConfigRead(
                type="aliyun",
                endpoint="",
                id="",
                is_aliyun_empty=True,
                is_tavily_empty=True,
            )],
        message="查询检索配置成功")

    websearch_config = WebSearchConfigRead(
        type=search_config_result.type or "aliyun",
        endpoint=search_config_result.endpoint,
        search_count=search_config_result.search_count,
        id=search_config_result.id,
        is_aliyun_empty=not search_config_result.encrypted_access_key_id,
        is_tavily_empty=not search_config_result.encrypted_tavily_api_key,
    )

    return success_response(data=[websearch_config], message="查询检索配置成功")
