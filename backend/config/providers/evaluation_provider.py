import traceback
from typing import Dict, Type
from loguru import logger
from sqlmodel import Field, SQLModel
from db.models.evaluation.evaluation import EvalEntity
from db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from config.providers.base_provider import BaseConfigProvider

@with_async_db_session
async def fetch_evaluation_by_id(session: AsyncSession, eval_id: str) -> EvalEntity:
    eval = await session.get(EvalEntity, eval_id)
    return eval


class EvaluationProvider(BaseConfigProvider):
    name_to_entry_id: Dict[str, str] = Field(default={})
    entity_class: Type[SQLModel] = EvalEntity

    def add(self, entry: EvalEntity):
        super().add(entry)
        self.name_to_entry_id[entry.name] = entry.id

    def update(self, entry: EvalEntity):
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

    async def aget_evaluation(self, evaluation_id: str) -> EvalEntity:
        if evaluation_id not in self.config_map:
            eval = await fetch_evaluation_by_id(eval_id=evaluation_id)
            if eval is None:
                raise ValueError(f"Knowledgebase {evaluation_id} not found.")

            self._load_entries([eval])
            return eval
        return self.config_map[evaluation_id]

    def get_evaluation_by_name(self, evaluation_name: str) -> EvalEntity:
        if evaluation_name not in self.name_to_entry_id:
            logger.info(f"Knowledgebase '{evaluation_name}' not found.")
            return None
        else:
            return self.config_map[self.name_to_entry_id[evaluation_name]]


evaluation_provider = EvaluationProvider()
