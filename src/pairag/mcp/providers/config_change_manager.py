

import asyncio
import traceback
from loguru import logger
from pairag.db.db_context import with_async_db_session
from pairag.db.models.change_event import ChangeEvent, ChangeEventSource, ChangeEventType
from pairag.mcp.providers.base_provider import BaseConfigProvider
from pairag.utils.time_utils import get_timestamp
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.mcp.providers.embedding_provider import embedding_provider
from pairag.mcp.providers.llm_provider import llm_provider
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider


class ConfigChangeManager:
    def __init__(self, worker_mode: bool = False):
        self.last_change_tick = -1 # 表示状态未初始化，不会扫描ChangeEvent表
        self.initialized = False
        self.worker_mode = worker_mode # 只需要管理embedding/llm/kb

    async def init_configuration(self):
        if self.initialized:
            return

        current_ts = get_timestamp()
        from pairag.db.db_context import init_db

        await init_db()
        logger.info("Initialized databases for MCP.")

        if not self.worker_mode:
            from pairag.mcp.providers.mcp_tool_provider import mcp_provider
            from pairag.mcp.providers.websearch_provider import websearch_provider
            await mcp_provider.full_load_from_db_async()
            logger.info("Initialized mcp tools.")
            await websearch_provider.full_load_from_db_async()
            logger.info("Initialized websearch configs.")

        await llm_provider.full_load_from_db_async()
        logger.info("Initialized llm models.")
        await embedding_provider.full_load_from_db_async()
        logger.info("Initialized embedding models.")
        await knowledgebase_provider.full_load_from_db_async()
        logger.info("Initialized knowledgebases.")

        self.initialized = True
        self.last_change_tick = current_ts

    @with_async_db_session
    async def notify_change_async(
        self,
        session: AsyncSession,
        event_source: ChangeEventSource,
        source_id: str,
        event_type: ChangeEventType,
    ):
        event = ChangeEvent(
            source_id=source_id,
            event_type=event_type,
            event_source=event_source,
        )

        session.add(event)
        await session.commit()


    @with_async_db_session
    async def monitor_changes(self):
        while True:
            if self.last_change_tick > 0:
                try:
                    # list changes
                    await self.process_change()
                except Exception:
                    logger.error(
                        f"Error when processing changes. Details:{traceback.format_exc()}"
                    )
            await asyncio.sleep(10)

    async def process_change(
        self,
        event_source: ChangeEventSource,
        source_id: str,
        event_type: ChangeEventType,
    ):
        config_provider = self._get_config_provider(event_source)
        await config_provider.process_event(
            event_type=event_type,
            source_id=source_id,
        )
        logger.info(f"Applied change event {event_type} for {event_source} {source_id}.")

    def _get_config_provider(
        self,
        event_source: ChangeEventSource) -> BaseConfigProvider:
        match event_source:
            case ChangeEventSource.EMBEDDING:
                return embedding_provider
            case ChangeEventSource.LLM:
                return llm_provider
            case ChangeEventSource.KNOWLEDGEBASE:
                return knowledgebase_provider
            case ChangeEventSource.MCP:
                from pairag.mcp.providers.mcp_tool_provider import mcp_provider
                return mcp_provider
            case ChangeEventSource.WEBSEARCH:
                from pairag.mcp.providers.websearch_provider import websearch_provider
                return websearch_provider
            case _:
                raise ValueError(f"Unknown event source: {event_source}")
