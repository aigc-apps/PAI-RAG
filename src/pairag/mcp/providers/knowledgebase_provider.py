from typing import Dict
from sqlmodel import Field
from pairag.db.models.knowledgebase.knowledgebase import KbEntity
from pairag.db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.mcp.providers.base_provider import BaseConfigProvider

@with_async_db_session
async def fetch_knowledgebases_by_id(session: AsyncSession, kb_id: str) -> KbEntity:
    kb = await session.get(KbEntity, kb_id)
    return kb


class KnowledgebaseProvider(BaseConfigProvider):
    name_to_entry_id: Dict[str, str] = Field(default={})

    def _load_entries(self, entries):
        super()._load_entries(entries)
        for entry_id, entry in self.config_map.items():
            self.name_to_entry_id[entry.name] = entry_id

    async def aget_knowledgebase(self, knowledgebase_id: str) -> KbEntity:
        if knowledgebase_id not in self.config_map:
            kb = await fetch_knowledgebases_by_id(knowledgebase_id)
            if kb is None:
                raise ValueError(f"Knowledgebase {knowledgebase_id} not found.")

            self._load_entries([kb])
            return kb
        return self.config_map[knowledgebase_id]

    def get_knowledgebase_by_name(self, knowledgebase_name: str) -> KbEntity:
        assert (
            knowledgebase_name in self.name_to_entry_id
        ), f"Knowledgebase '{knowledgebase_name}' not found."
        return self.config_map[self.name_to_entry_id[knowledgebase_name]]


knowledgebase_provider = KnowledgebaseProvider()
