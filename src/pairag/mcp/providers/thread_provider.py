from typing import Dict
from sqlmodel import select
from pairag.db.models.thread import ThreadEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger


@with_async_db_session
async def fetch_threads(session: AsyncSession):
    logger.info("[ThreadProvider] Start fetching threads.")
    sql_results = await session.exec(select(ThreadEntity))
    thread_results = sql_results.all()

    logger.info(f"[ThreadProvider] fetched {len(thread_results)} threads.")
    return {thread.id: thread for thread in thread_results}


class ThreadProvider:
    def __init__(self):
        self.thread_map: Dict[str, ThreadEntity] = {}

    async def refresh(self):
        logger.info("[ThreadProvider] Start refreshing threads.")
        self.thread_map = await fetch_threads()
        logger.info(f"[ThreadProvider] refreshed {len(self.thread_map)} threads.")

    def get_thread(self, thread_id: str) -> ThreadEntity:
        assert thread_id in self.thread_map, f"Thread '{thread_id}' not found."
        return self.thread_map[thread_id]


thread_provider = ThreadProvider()
