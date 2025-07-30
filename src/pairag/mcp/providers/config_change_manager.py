

from datetime import datetime, timezone
import asyncio
from sqlmodel import select
import traceback
from loguru import logger
from tenacity import retry, stop_after_attempt
from pairag.db.db_context import with_async_db_session
from pairag.db.models.change_event import ChangeEvent, ChangeEventSource, ChangeEventType
from pairag.mcp.providers.base_provider import BaseConfigProvider
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.mcp.providers.embedding_provider import embedding_provider
from pairag.mcp.providers.llm_provider import llm_provider
from pairag.mcp.providers.knowledgebase_provider import knowledgebase_provider

class ConfigChangeManager:
    def __init__(self, worker_mode: bool = False):
        self.last_change_dt = None # 表示状态未初始化，不会扫描ChangeEvent表
        self.initialized = False
        self.worker_mode = worker_mode # 只需要管理embedding/llm/kb

    async def init_configuration(self):
        if self.initialized:
            return

        current_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        from pairag.db.db_context import init_db

        await init_db()
        logger.info("Initialized databases for MCP.")

        if not self.worker_mode:
            from pairag.mcp.providers.mcp_tool_provider import mcp_provider
            from pairag.mcp.providers.websearch_provider import websearch_provider
            from pairag.mcp.providers.reranker_provider import reranker_provider
            await mcp_provider.full_load_from_db_async()
            logger.info("Initialized mcp tools.")
            await websearch_provider.full_load_from_db_async()
            logger.info("Initialized websearch configs.")
            await reranker_provider.full_load_from_db_async()
            logger.info("Initialized reranker configs.")

        await llm_provider.full_load_from_db_async()
        logger.info("Initialized llm models.")
        await embedding_provider.full_load_from_db_async()
        logger.info("Initialized embedding models.")
        await knowledgebase_provider.full_load_from_db_async()
        logger.info("Initialized knowledgebases.")

        self.initialized = True
        self.last_change_dt = current_dt
        logger.info(f"ConfigManager inited with worker_mode {self.worker_mode}, timestamp {self.last_change_dt}")

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
        logger.info(f"Submitting change event: {event}")

        session.add(event)
        await session.commit()
        logger.info(f"Notified change event: {event}")


    @with_async_db_session
    async def monitor_changes_async(self, session: AsyncSession):
        while True:
            if self.last_change_dt is not None:
                try:
                    event = (await session.exec(
                        select(ChangeEvent)
                        .where(ChangeEvent.created_at > self.last_change_dt)
                        .order_by(ChangeEvent.created_at.asc())
                    )).first()

                    if not event:
                        await asyncio.sleep(10)
                    else:
                        logger.info(f"Found change event {event}.")
                        await self.process_change(event)
                except Exception:
                    logger.error(
                        f"Error when processing changes. Details:{traceback.format_exc()}"
                    )
                finally:
                    if event:
                        logger.info(f"Updated change timestamp from {self.last_change_dt} to {event.created_at}")
                        self.last_change_dt = event.created_at

    @retry(stop=stop_after_attempt(3))
    async def process_change(
        self,
        event: ChangeEvent,
    ):
        config_provider = self._get_config_provider(event.event_source)
        await config_provider.process_event(
            event_type=event.event_type,
            source_id=event.source_id,
        )
        logger.info(f"Applied change event {event.event_type} for {event}.")

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
            case ChangeEventSource.RERANK:
                from pairag.mcp.providers.reranker_provider import reranker_provider
                return reranker_provider
            case _:
                raise ValueError(f"Unknown event source: {event_source}")

config_change_manager = ConfigChangeManager()
