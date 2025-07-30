from typing import Any, Dict, List, Type
from loguru import logger
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from pairag.db.db_context import with_async_db_session
from pairag.db.models.change_event import ChangeEventType
from pairag.utils.constants import try_get_int_env
from pairag.utils.lru_cache import LruCache

MAX_OBJECT_CACHE_SIZE = try_get_int_env("MAX_OBJECT_CACHE_SIZE", 300)
DEFAULT_ID_FIELD = "id"


class BaseConfigProvider(BaseModel):
    config_map: Dict[str, BaseModel] = Field(default={})
    instance_map: Any = Field(default=None)
    entity_class: Type[SQLModel] = Field(default=None)


    def __init__(self, entries: List[BaseModel] = []):
        super().__init__()
        self.config_map = {}
        self.instance_map = LruCache(maxsize=MAX_OBJECT_CACHE_SIZE)
        self._load_entries(entries)

    def _load_entries(self, entries: List[BaseModel]):
        for entry in entries:
            if hasattr(entry, DEFAULT_ID_FIELD):
                self.config_map[entry.id] = entry
            else:
                raise ValueError(f"{DEFAULT_ID_FIELD} attribute not found in object {entry}.")

    @with_async_db_session
    async def full_load_from_db_async(self, session: AsyncSession):
        entries = (await session.exec(select(self.entity_class))).all()
        self._load_entries(entries)

    @with_async_db_session
    async def process_event(
        self,
        session: AsyncSession,
        event_type: ChangeEventType,
        source_id: str,
    ):
        if event_type == ChangeEventType.DELETE:
            self.delete(source_id)
        elif event_type == ChangeEventType.UPDATE:
            entity = await session.get(self.entity_class, source_id)
            await session.refresh(entity)
            self.update(entity)
        elif event_type == ChangeEventType.ADD:
            entity = await session.get(self.entity_class, source_id)
            self.add(entity)
        else:
            raise ValueError(f"Invalid event type: {event_type}")

    ## Config related api
    def add(self, entry: BaseModel):
        if hasattr(entry, DEFAULT_ID_FIELD):
            self.config_map[entry.id] = entry
            logger.info(f"`{entry.id}` added to config_map.")
        else:
            raise ValueError(f"{DEFAULT_ID_FIELD} attribute not found in object {entry}.")

    def update(self, entry: BaseModel):
        self.delete_instance_if_exist(entry_id=entry.id)
        self.add(entry=entry) # default update和add没有区别

    def delete(self, entry_id: str):
        if entry_id not in self.config_map:
            logger.warning(f"`{entry_id}` not found in config_map. {self.config_map}")
            return

        del self.config_map[entry_id]
        self.delete_instance_if_exist(entry_id)
        logger.info(f"Deleted `{entry_id}` from config_map.")


    ## Instance related
    def _create_instance(self, config: BaseModel) -> Any:
        raise NotImplementedError


    def delete_instance_if_exist(self, entry_id: str):
        if entry_id in self.instance_map:
            self.instance_map.delete(entry_id)
            logger.info(f"Removed instance for {entry_id}.")


    def get_instance(self, entry_id: str):
        instance = self.instance_map.get(entry_id)
        if instance is None:
            if entry_id not in self.config_map:
                raise ValueError(f"`{entry_id}` not found in instance_map.")
            instance = self._create_instance(self.config_map[entry_id])
            self.instance_map.put(entry_id, instance)
        return instance
