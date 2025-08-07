import traceback
from typing import Dict, List, Type

from sqlmodel import SQLModel
from config.providers.base_provider import BaseConfigProvider
from db.models.chatbot import ChatBotEntity
from pydantic import Field
from loguru import logger


class ChatBotProvider(BaseConfigProvider):
    config_map: Dict[str, ChatBotEntity] = Field(default={})
    app_id_to_entry_id: Dict[str, str] = Field(default={})
    entity_class: Type[SQLModel] = ChatBotEntity


    def _load_entries(self, entries: List[ChatBotEntity]):
        super()._load_entries(entries)
        for entry in entries:
            self.app_id_to_entry_id[entry.app_id] = entry.id

    def add(self, entry: ChatBotEntity):
        super().add(entry)
        self.app_id_to_entry_id[entry.app_id] = entry.id

    def update(self, entry: ChatBotEntity):
        super().update(entry)
        self.app_id_to_entry_id[entry.app_id] = entry.id

    def delete(self, entry_id: str):
        super().delete(entry_id)
        try:
            for k, v in self.app_id_to_entry_id.items():
                if v == entry_id:
                    del self.app_id_to_entry_id[k]
                    break
        except Exception:
            logger.warning(f"Failed to delete entry with entry_id {entry_id}. error: {traceback.format_exc()}.")

    def _load_entries(self, entries):
        super()._load_entries(entries)
        for entry_id, entry in self.config_map.items():
            self.app_id_to_entry_id[entry.app_id] = entry_id

    def get_chatbot(self, app_id: str):
        assert app_id in self.app_id_to_entry_id, f"`{app_id}` not found. {self.app_id_to_entry_id}"
        entry_id = self.app_id_to_entry_id[app_id]
        return self.config_map[entry_id]


chatbot_provider = ChatBotProvider()
