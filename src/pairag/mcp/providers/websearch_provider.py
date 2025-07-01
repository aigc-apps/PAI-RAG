import os
from sqlmodel import select
from pairag.db.encrypt_utils import decrypt_key
from pairag.db.models.websearch import WebSearchConfigEntity
from pairag.db.db_context import with_async_db_session
from pairag.mcp.tools.search.aliyun_search_tool import aget_aliyun_search_result
from sqlmodel.ext.asyncio.session import AsyncSession
from llama_index.core.tools import FunctionTool
from loguru import logger


@with_async_db_session
async def fetch_websearch_config(session: AsyncSession):
    logger.info("[WebSearchProvider] Start fetching search config.")
    sql_result = await session.exec(select(WebSearchConfigEntity))
    search_entity = sql_result.first()

    if search_entity is None:
        logger.warning("Search config not exists.")
        return False

    os.environ["WEBSEARCH_ACCESS_KEY_ID"] = decrypt_key(
        search_entity.encrypted_access_key_id
    )
    os.environ["WEBSEARCH_ACCESS_KEY_SECRET"] = decrypt_key(
        search_entity.encrypted_access_key_secret
    )

    return True


class WebSearchProvider:
    def __init__(self):
        self.is_available = (
            False  # might be a set of available search engines in the future
        )

    async def refresh(self):
        logger.info("[WebSearchProvider] Start refreshing search configs.")
        self.is_available = await fetch_websearch_config()
        logger.info(
            f"[WebSearchProvider] refreshed websearch configs, available: {self.is_available}."
        )

    def get_search_tools(self):
        assert self.is_available, "There is no available websearch configs."
        aliyun_search_tool = FunctionTool.from_defaults(
            async_fn=aget_aliyun_search_result,
            name="search-web",
            description="从阿里云搜索引擎中搜索给定查询的最新内容。",
        )
        return [aliyun_search_tool]


websearch_provider = WebSearchProvider()
