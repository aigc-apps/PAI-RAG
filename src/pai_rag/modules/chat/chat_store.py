import os
from typing import Dict, List, Any
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.storage.chat_store.redis import RedisChatStore
from pai_rag.utils.store_utils import read_chat_store_state
from llama_index.core.memory import ChatMemoryBuffer
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

CHAT_STORE_FILE = "chat_store.json"


class ChatStoreModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        self.config = new_params[MODULE_PARAM_CONFIG]
        self.persist_path = self.config.persist_path
        self.chat_store_type = self.config.get("type", "Local")
        self.chat_store = None
        self.is_chat_empty = (
            read_chat_store_state(self.persist_path, CHAT_STORE_FILE) is None
        )
        self.init_chat_store()
        return self

    def init_chat_store(self):
        if not self.chat_store:
            self.chat_store = self._get_or_create_chat_store(self.is_chat_empty)
            self.is_chat_empty = False
            print("Create chat store successfully.")

    def _get_or_create_chat_store(self, is_empty):
        if self.chat_store_type == "Local":
            if is_empty:
                chat_store = SimpleChatStore()
            else:
                chat_store = SimpleChatStore().from_persist_path(
                    os.path.join(self.persist_path, CHAT_STORE_FILE)
                )
            return chat_store
        elif self.chat_store_type == "Aliyun-Redis":
            self.redis_host = self.config.get("host", "localhost")
            self.redis_pwd = self.config.get("password", "pwd")
            chat_store = RedisChatStore(
                redis_url=f"redis://{self.redis_host}:6379", password=self.redis_pwd
            )
            print("Initialize Redis chat store successfully.")
            return chat_store
        else:
            raise ValueError(f"Unknown chat_store type '{self.chat_store_type}'.")

    def persist(self):
        if self.chat_store_type == "Local":
            self.chat_store.persist(
                persist_path=os.path.join(self.persist_path, CHAT_STORE_FILE)
            )

    def get_chat_memory_buffer(self, store_key):
        chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=self.chat_store,
            chat_store_key=store_key,
        )
        return chat_memory
