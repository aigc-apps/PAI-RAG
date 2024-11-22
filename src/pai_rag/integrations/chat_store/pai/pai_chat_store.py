from enum import Enum
import os
from typing import List, Any, Literal, Optional
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.storage.chat_store.base import BaseChatStore
from pydantic import BaseModel
from llama_index.core.bridge.pydantic import PrivateAttr
from loguru import logger

CHAT_STORE_FILE = "chat_store.json"
DEFAULT_LOCAL_STORAGE_PATH = "./localdata/storage/"


class ChatStoreType(str, Enum):
    """Chat store types."""

    local = "local"
    redis = "redis"


class BaseChatStoreConfig(BaseModel):
    type: ChatStoreType = ChatStoreType.local


class LocalChatStoreConfig(BaseChatStoreConfig):
    type: Literal[ChatStoreType.local] = ChatStoreType.local
    persist_path: str = DEFAULT_LOCAL_STORAGE_PATH


class RedisChatStoreConfig(BaseChatStoreConfig):
    type: Literal[ChatStoreType.redis] = ChatStoreType.redis
    host: str = "localhost"
    password: str = "usr:pwd"


def create_chat_store(chat_store_config: BaseChatStoreConfig) -> BaseChatStore:
    if isinstance(chat_store_config, LocalChatStoreConfig):
        chat_file = os.path.join(chat_store_config.persist_path, CHAT_STORE_FILE)
        if not os.path.exists(chat_file):
            return SimpleChatStore()
        else:
            return SimpleChatStore().from_persist_path(chat_file)
    elif isinstance(chat_store_config, RedisChatStoreConfig):
        redis_chat_store = RedisChatStore(
            redis_url=f"redis://{chat_store_config.host}:6379",
            password=chat_store_config.password,
        )
        logger.info(
            "Adding Redis chat store to 'redis://{chat_store_config.host}:6379'."
        )
        return redis_chat_store
    else:
        raise ValueError(f"Unknown chat store config: {chat_store_config}")


class PaiChatStore(BaseChatStore):
    _chat_store: Any = PrivateAttr()

    def __init__(self, chat_store_config: BaseChatStoreConfig):
        self._chat_store = create_chat_store(chat_store_config)
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "PaiChatStore"

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key."""
        self._chat_store.set_messages(key, messages)
        return

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        return self._chat_store.get_messages(key)

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        self._chat_store.add_message(key, message)
        return

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete messages for a key."""
        return self._chat_store.delete_messages(key)

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        return self._chat_store.delete_message(key, idx)

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        return self._chat_store.delete_last_message(key)

    def get_keys(self) -> List[str]:
        """Get all keys."""
        return self._chat_store.get_keys()
