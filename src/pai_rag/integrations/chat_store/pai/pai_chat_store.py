from enum import Enum
from threading import Lock
from typing import List, Any, Literal, Optional
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.core.bridge.pydantic import Field
from pydantic import BaseModel
from llama_index.core.bridge.pydantic import PrivateAttr
from loguru import logger
from collections import OrderedDict, deque

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
    ttl: int = 86400


def create_chat_store(chat_store_config: BaseChatStoreConfig) -> BaseChatStore:
    if isinstance(chat_store_config, LocalChatStoreConfig):
        logger.info("Adding local LRU chat store.")
        return LruSimpleChatStore()

    elif isinstance(chat_store_config, RedisChatStoreConfig):
        redis_chat_store = RedisChatStore(
            redis_url=f"redis://{chat_store_config.host}:6379",
            password=chat_store_config.password,
            ttl=chat_store_config.ttl,
        )
        logger.info(
            f"Adding Redis chat store to 'redis://{chat_store_config.host}:6379'."
        )
        return redis_chat_store
    else:
        raise ValueError(f"Unknown chat store config: {chat_store_config}")


class LruSimpleChatStore(SimpleChatStore):
    max_message_per_session: int = Field(default=20)
    max_session_num: int = Field(default=1000)
    _cache: OrderedDict = PrivateAttr()
    _lock: Lock = PrivateAttr()

    def __init__(self, max_message_per_session: int = 20, max_session_num: int = 1000):
        super().__init__()

        self.max_session_num = max_session_num
        self.max_message_per_session = max_message_per_session
        self._cache = OrderedDict()
        self._lock = Lock()

    def add_message(self, key, message, idx=None):
        """Add a message for a key."""
        if idx is None:
            self.store.setdefault(key, []).append(message)
        else:
            self.store.setdefault(key, []).insert(idx, message)

        self.check_store_capacity(key)

    def get_messages(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)

        return self.store.get(key, [])

    def set_messages(self, key, messages):
        self.store[key] = messages[-self.max_message_per_session :]
        self.check_store_capacity(key)

    def check_store_capacity(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = True

        if len(self.store[key]) > self.max_message_per_session:
            self.store[key] = self.store[key][-self.max_message_per_session :]

        if len(self.store) > self.max_session_num:
            with self._lock:
                session_key, _ = self._cache.popitem(last=False)
                del self.store[session_key]


class PaiChatStore(BaseChatStore):
    _chat_store: Any = PrivateAttr()

    def __init__(self, chat_store_config: BaseChatStoreConfig):
        super().__init__()
        self._chat_store = create_chat_store(chat_store_config)

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
        default_messages = self._chat_store.get_messages(key)
        recent_messages = deque(default_messages[-6:], maxlen=6)
        ret_messages = []
        for msg in reversed(recent_messages):
            msg.content = msg.content[:600]
            ret_messages.append(msg)

        ret_messages.reverse()
        return ret_messages

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
