from typing import Dict, List
from sqlmodel import select
from pairag.db.models.knowledgebase.knowledgebase import KbEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from loguru import logger


@with_async_db_session
async def fetch_knowledgebases(session: AsyncSession) -> List[KbEntity]:
    logger.info("[KnowledgebaseProvider] Start fetching knowledgebases.")
    sql_results = await session.exec(select(KbEntity))
    knowledgebase_results = sql_results.all()

    logger.info(
        f"[KnowledgebaseProvider] fetched {len(knowledgebase_results)} knowledges."
    )
    return knowledgebase_results


class KnowledgebaseProvider:
    def __init__(self):
        self.knowledgebase_map: Dict[str, KbEntity] = {}

    async def refresh(self):
        logger.info("[KnowledgebaseProvider] Start refreshing knowledgebases.")
        knowledgebases = await fetch_knowledgebases()
        self.knowledgebase_map = {
            knowledgebase.id: knowledgebase for knowledgebase in knowledgebases
        }
        self.knowledgebase_name_map = {
            knowledgebase.name: knowledgebase for knowledgebase in knowledgebases
        }
        logger.info(
            f"[KnowledgebaseProvider] refreshed {len(self.knowledgebase_map)} knowledgebases."
        )

    def get_knowledgebase(self, knowledgebase_id: str) -> KbEntity:
        assert (
            knowledgebase_id in self.knowledgebase_map
        ), f"Knowledgebase {knowledgebase_id} not found."
        return self.knowledgebase_map[knowledgebase_id]

    def get_knowledgebase_by_name(self, knowledgebase_name: str) -> KbEntity:
        assert (
            knowledgebase_name in self.knowledgebase_name_map
        ), f"Knowledgebase '{knowledgebase_name}' not found."
        return self.knowledgebase_name_map[knowledgebase_name]


knowledgebase_provider = KnowledgebaseProvider()
