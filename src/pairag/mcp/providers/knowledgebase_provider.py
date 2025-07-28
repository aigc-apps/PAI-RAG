import traceback
from typing import Dict, Type
from loguru import logger
from sqlmodel import Field, SQLModel
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
    entity_class: Type[SQLModel] = KbEntity

    def add(self, entry: KbEntity):
        super().add(entry)
        self.name_to_entry_id[entry.name] = entry.id

    def update(self, entry: KbEntity):
        super().update(entry)
        self.name_to_entry_id[entry.name] = entry.id

    def delete(self, entry_id: str):
        super().delete(entry_id)
        try:
            for k, v in self.name_to_entry_id.items():
                if v == entry_id:
                    del self.name_to_entry_id[k]
                    break
        except Exception:
            logger.warning(f"Failed to delete entry with entry_id {entry_id}. error: {traceback.format_exc()}.")

    def _load_entries(self, entries):
        super()._load_entries(entries)
        for entry_id, entry in self.config_map.items():
            self.name_to_entry_id[entry.name] = entry_id

    async def aget_knowledgebase(self, knowledgebase_id: str) -> KbEntity:
        if knowledgebase_id not in self.config_map:
            kb = await fetch_knowledgebases_by_id(kb_id=knowledgebase_id)
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
