from typing import Any, Type

from pydantic import Field
from sqlmodel import SQLModel
from common.encrypt_utils import decrypt_key
from db.models.chatdb.chatdb import ChatDbConfigEntity
from config.providers.base_provider import BaseConfigProvider
from config.providers.llm_provider import llm_provider
from tools.chatdb.xiyan_client import XiyanClient
from llama_index.core.tools import FunctionTool
from loguru import logger


class ChatDbProvider(BaseConfigProvider):
    entity_class: Type[SQLModel] = ChatDbConfigEntity
    client: Any = Field(default=None)


    def _load(self, entry: ChatDbConfigEntity):
        try:
            llm_model = llm_provider.get_llm_model(model_id=entry.model_id)
        except Exception as e:
            logger.error(f"Error getting llm model {entry.model_id} for chatdb: {e}")
            return

        self.client = XiyanClient(
            dialect=entry.dialect,
            host=entry.host,
            port=entry.port,
            db_name=entry.db_name,
            username=entry.username,
            password=decrypt_key(entry.encrypted_password),
            llm=llm_model,
        )

    def _load_entries(self, entries):
        super()._load_entries(entries)
        if len(entries) > 0:
            self._load(entries[0])

    def add(self, entry):
        super().add(entry)
        self._load(entry)

    def update(self, entry):
        super().update(entry)
        self._load(entry)

    def get_db_tools(self):
        assert len(self.config_map) > 0, "There is no available chat_db configs."
        assert self.client is not None, "There is no available chat_db tools."

        db_tool = FunctionTool.from_defaults(
                async_fn=self.client.execute_async,
                name="chat-db",
                description="使用自然语言从给定的数据库中获取数据。输入参数: query(str类型),表示用户的查询意图，需结合上下文信息生成。",
            )
        return [db_tool]

chatdb_provider = ChatDbProvider()
