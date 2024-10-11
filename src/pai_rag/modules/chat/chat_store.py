import os
from typing import Dict, List, Any
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.storage.chat_store.redis import RedisChatStore
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG

CHAT_STORE_FILE = "chat_store.json"


class ChatStoreModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return []

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        persist_path = config.get("persist_path", "localdata/chat")
        chat_store_type = config.get("type", "Local")

        if chat_store_type == "Local":
            chat_file = os.path.join(persist_path, CHAT_STORE_FILE)
            if not os.path.exists(chat_file):
                chat_store = SimpleChatStore()
            else:
                chat_store = SimpleChatStore().from_persist_path(chat_file)
            return chat_store
        elif chat_store_type == "Aliyun-Redis":
            redis_host = config.get("host", "localhost")
            redis_pwd = config.get("password", "pwd")
            chat_store = RedisChatStore(
                redis_url=f"redis://{redis_host}:6379", password=redis_pwd
            )
            print("Initialize Redis chat store successfully.")
            return chat_store
        else:
            raise ValueError(f"Unknown chat_store type '{chat_store_type}'.")
