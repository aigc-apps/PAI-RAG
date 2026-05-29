### Web search configuration API ###

from typing import List
from fastapi import APIRouter, Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.websearch import (
    WebSearchConfigRead,
    WebSearchConfigCreate,
)
from common.chat.response_model import ResponseModel, success_response
from api.api_exception import ApiException, handle_api_exceptions
from db.db_context import get_db_session
from service.injection import get_websearch_service, get_tenant_id
from service.tool.websearch_service import WebsearchService


websearch_router = APIRouter()


@websearch_router.post("", response_model=ResponseModel[WebSearchConfigRead])
@handle_api_exceptions(action="update websearch config")
async def add_search_config(
    new_search_config: WebSearchConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    websearch_service: WebsearchService = Depends(get_websearch_service),
):
    """
    Create or update web search configuration.

    Args:
        new_search_config: Web search configuration data
        session: Database session
        websearch_service: WebsearchService instance

    Returns:
        ResponseModel with WebSearchConfigRead
    """
    # API layer input validation
    if new_search_config.type and new_search_config.type not in ["tavily", "aliyun"]:
        raise ApiException(code=400, message="不支持的搜索引擎类型，仅支持tavily和aliyun")

    # Use service layer for business logic
    search_config = await websearch_service.create_or_update_websearch_config(
        config_data=new_search_config,
        tenant_id=tenant_id
    )

    # Convert to read model
    websearch_config_read = WebSearchConfigRead(
        type=search_config.type or "aliyun",
        endpoint=search_config.endpoint,
        search_count=search_config.search_count,
        id=search_config.id,
        is_aliyun_empty=not search_config.encrypted_access_key_id,
        is_tavily_empty=not search_config.encrypted_tavily_api_key,
    )

    return success_response(
        data=websearch_config_read, message="Update websearch config success."
    )


@websearch_router.get("", response_model=ResponseModel[List[WebSearchConfigRead]])
@handle_api_exceptions(action="list websearch config")
async def list_search_config(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    websearch_service: WebsearchService = Depends(get_websearch_service),
):
    """
    List web search configurations.

    Args:
        session: Database session
        websearch_service: WebsearchService instance
        offset: Pagination offset
        limit: Pagination limit

    Returns:
        ResponseModel with list of WebSearchConfigRead
    """
    # Use service layer to get all configs
    configs = await websearch_service.get_all_websearch_configs(tenant_id=tenant_id)

    if not configs:
        # Return default empty config if none exists
        return success_response(
            data=[
                WebSearchConfigRead(
                    type="aliyun",
                    endpoint="",
                    id="",
                    is_aliyun_empty=True,
                    is_tavily_empty=True,
                )
            ],
            message="List websearch config success.",
        )

    # Convert to read models
    websearch_configs = []
    for config in configs:
        websearch_config_read = WebSearchConfigRead(
            type=config.type or "aliyun",
            endpoint=config.endpoint,
            search_count=config.search_count,
            id=config.id,
            is_aliyun_empty=not config.encrypted_access_key_id,
            is_tavily_empty=not config.encrypted_tavily_api_key,
        )
        websearch_configs.append(websearch_config_read)

    return success_response(
        data=websearch_configs, message="List websearch config success."
    )
