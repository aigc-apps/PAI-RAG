from typing import Dict
from sqlmodel import select
from pairag.db.models.knowledgebase.knowledgebase import KnowledgebaseEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger


@with_async_db_session
async def fetch_knowledgebases(session: AsyncSession):
    logger.info("[KnowledgebaseProvider] Start fetching knowledgebases.")
    sql_results = await session.exec(select(KnowledgebaseEntity))
    knowledgebase_results = sql_results.all()

    logger.info(
        f"[KnowledgebaseProvider] fetched {len(knowledgebase_results)} knowledges."
    )
    return {
        knowledgebase.name: knowledgebase for knowledgebase in knowledgebase_results
    }


class KnowledgebaseProvider:
    def __init__(self):
        self.knowledgebase_map: Dict[str, KnowledgebaseEntity] = {}

    async def refresh(self):
        logger.info("[KnowledgebaseProvider] Start refreshing knowledgebases.")
        self.knowledgebase_map = await fetch_knowledgebases()
        logger.info(
            f"[KnowledgebaseProvider] refreshed {len(self.knowledgebase_map)} knowledgebases."
        )

    def get_knowledgebase(self, knowledgebase_name: str) -> KnowledgebaseEntity:
        assert (
            knowledgebase_name in self.knowledgebase_map
        ), f"Knowledgebase '{knowledgebase_name}' not found."
        return self.knowledgebase_map[knowledgebase_name]


knowledgebase_provider = KnowledgebaseProvider()
